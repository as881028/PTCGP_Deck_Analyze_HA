import os
import json
import logging
import re
import time
import requests
import pandas as pd
import openai
from b_score import calculate_b_score
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'MIN_COUNT': 20,
    'MIN_WIN_RATE': 45.0,
    'TOP_N': 10,
    'CACHE_DURATION': 3600,  # seconds (1 hour)
    'DECK_CACHE': 'deck_cache.json',
    'MATCHUP_CACHE': 'matchup_cache.json',
    'NAME_CACHE': 'pokemon_name_cache.json',
    'DEBUG': False,
    'MIN_MATCH_COUNT': 5,
    'FORCE_FRESH': True,  # 強制重新抓取資料，忽略緩存
    # 分批撈取對戰資料以降低伺服器負擔
    'BATCH_FETCH': True,
    'BATCH_SIZE': 3,   # 每批處理的牌組數量
    'BATCH_DELAY': 2,  # 每批之間暫停秒數
    'PROMPT_LOG_DIR': 'prompt_logs',
    'CHATGPT': {
        'MODEL': 'o3-mini',
        'PROMPT_FILES': {
            'SYSTEM': 'system_prompt.txt'
        },
        'USER_PROMPT_TEMPLATE': """
以下為 TOP{deck_count} 牌組對所有對手的對戰矩陣 (JSON)。
請僅使用 JSON 中資料依 system 演算法計算，禁止引用任何外部數字。
請「僅使用下方 JSON 數據」依 system 指令計算，不得引用表外或模型記憶的資料。

### 對戰矩陣 (JSON) - 包含全部對局
```json
{matchup_matrix}
```

### B-Score 計算結果 (JSON)
```json
{b_score_results}
```

請依系統算法輸出三段結果：
1. 計算表（包含 B-Score 結果）
2. 判斷依據（參考 B-Score 結果）
3. 結論（包含最有利牌組和有缺陷牌組）
"""
    }
}

# OpenAI API key – set environment variable OPENAI_API_KEY=<your key>
openai_api_key = os.getenv("OPENAI_API_KEY")

# openai 庫自 v1 起不再讀取全域變數，因此僅記錄是否載入金鑰
if openai_api_key:
    logging.info("已載入 OPENAI_API_KEY")
else:
    logging.info("未提供 OPENAI_API_KEY，ChatGPT 將被跳過")

logging.basicConfig(
    level=logging.DEBUG if CONFIG['DEBUG'] else logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

# ============================================================
# Utility: generic JSON cache wrapper
# ============================================================
class JsonCache:
    """Simple timestamp‑based JSON cache."""

    def __init__(self, path: str, duration_sec: int):
        self.path = path
        self.duration = duration_sec

    def load(self) -> Optional[Any]:
        if not os.path.exists(self.path):
            return None
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            ts = datetime.fromisoformat(data.get('timestamp'))
            if datetime.now() - ts < timedelta(seconds=self.duration):
                return data['data']
        except Exception as e:
            logging.warning(f"Cache read error ({self.path}): {e}")
        return None

    def save(self, data: Any):
        tmp_path = f"{self.path}.bak"
        if os.path.exists(self.path):
            try:
                os.replace(self.path, tmp_path)
            except Exception:
                pass
        try:
            payload = {'timestamp': datetime.now().isoformat(), 'data': data}
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Cache save error: {e}")
            if os.path.exists(tmp_path):
                os.replace(tmp_path, self.path)

# ============================================================
# Pokemon Chinese name resolver (w/ cache)
# ============================================================
class NameResolver:
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.cache = self._load_cache()
        self.skip_lookup = False  # 若無法查詢中文名則跳過後續查詢

    def _load_cache(self) -> Dict[str, str]:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def save_cache(self):
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Name cache save error: {e}")

    def get(self, english_name: str) -> str:
        if english_name in self.cache:
            return self.cache[english_name]
        if self.skip_lookup:
            return english_name
        # 處理 Tapu 後接空白的寶可夢名稱
        english_name = re.sub(r'Tapu\s+(\w+)', r'Tapu\1', english_name)
        
        # 分割牌組名稱（通常是兩隻寶可夢）
        parts = english_name.split(' ')
        zh_parts: List[str] = []
        
        for part in parts:
            if part.lower() == 'ex':
                if zh_parts:
                    zh_parts[-1] += ' EX'
            else:
                zh_parts.append(self._query_wiki(part))
        
        # 用加號連接兩隻寶可夢的名稱
        zh_name = ' + '.join(zh_parts)
        self.cache[english_name] = zh_name
        self.save_cache()
        return zh_name

    def _query_wiki(self, name: str) -> str:
        if self.skip_lookup:
            return name

        url = f"http://wiki.52poke.com/wiki/{name}"
        try:
            res = requests.get(url, timeout=8)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, 'html.parser')
            h1 = soup.find('h1', {'id': 'firstHeading'})
            if h1:
                return h1.text.strip()
        except Exception as e:
            logging.info(
                f"Wiki lookup failed for {name}, skip further lookups: {e}")
            self.skip_lookup = True
        # 若查詢失敗則直接回傳英文名稱
        return name

# ============================================================
# Scrape top decks list from Limitless
# ============================================================

def fetch_decks() -> pd.DataFrame:
    url = "https://play.limitlesstcg.com/decks?game=pocket"
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; PocketDeckBot/1.0)'
    }
    res = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(res.text, 'html.parser')
    table = soup.find('table')
    rows = table.find_all('tr')[1:] if table else []
    decks: List[Dict[str, Any]] = []
    for row in rows:
        cols = row.find_all('td')
        if len(cols) < 7:
            continue
        try:
            deck_name = cols[2].get_text(strip=True)
            count = int(cols[3].get_text(strip=True))
            share = float(cols[4].get_text(strip=True).replace('%', ''))
            win_pct_raw = cols[6].get_text(strip=True).replace('%', '')
            win_pct = float(win_pct_raw) if win_pct_raw and win_pct_raw != 'NaN' else 0.0
            decks.append({
                'Deck Name': deck_name,
                'Count': count,
                'Share': share,
                'Win %': win_pct
            })
        except Exception as e:
            logging.debug(f"Parse deck row error: {e}")
    df = pd.DataFrame(decks)
    df = df[(df['Count'] > CONFIG['MIN_COUNT']) & (df['Win %'] > CONFIG['MIN_WIN_RATE'])]
    df = df.sort_values('Win %', ascending=False).head(CONFIG['TOP_N']).copy()
    return df

# ============================================================
# Fetch matchup table for a given deck
# ============================================================

def fetch_matchups(deck_name: str, resolver: NameResolver) -> List[Dict[str, Any]]:
    list_url = "https://play.limitlesstcg.com/decks?game=pocket"
    soup = BeautifulSoup(requests.get(list_url, timeout=8).text, 'html.parser')
    table = soup.find('table')
    matchup_url: Optional[str] = None
    for row in table.find_all('tr')[1:] if table else []:
        cols = row.find_all('td')
        if len(cols) >= 3 and cols[2].get_text(strip=True).lower() == deck_name.lower():
            link = row.find('a', href=lambda x: x and 'matchups' in x)
            if link:
                matchup_url = f"https://play.limitlesstcg.com{link['href']}"
                break
    if not matchup_url:
        return []

    res = requests.get(matchup_url, timeout=8)
    soup = BeautifulSoup(res.text, 'html.parser')
    table = soup.find('table')
    rows = table.find_all('tr')[1:] if table else []
    result: List[Dict[str, Any]] = []
    for row in rows:
        cols = row.find_all('td')
        if len(cols) < 4:
            continue
        try:
            opp = cols[1].get_text(strip=True)
            match_cnt = int(cols[2].get_text(strip=True))
            score = cols[3].get_text(strip=True)
            if match_cnt < CONFIG['MIN_MATCH_COUNT']:
                continue
            if '-' in score:  # W-L-T 格式
                wins, losses, *_ = (int(x) for x in score.split('-'))
                win_rate = (wins / match_cnt) * 100 if match_cnt else 0
            else:  # 百分比格式
                win_rate = float(score.replace('%', ''))
            result.append({
                'opponent': opp,
                'opponent_chinese': resolver.get(opp),
                'win_rate': win_rate,
                'raw_score': score,
                'total_matches': match_cnt
            })
        except Exception as e:
            logging.debug(f"Parse matchup row error: {e}")
    return result

# ============================================================
# Build matchup matrix for top decks
# ============================================================

def analyze_top_matchups(df: pd.DataFrame, resolver: NameResolver, cache: JsonCache) -> Dict[str, Dict[str, Dict[str, Any]]]:
    top_decks = df['Deck Name'].tolist()
    matrix: Dict[str, Dict[str, Dict[str, Any]]] = {}

    cached_data = cache.load() or {}
    for idx, deck in enumerate(top_decks):
        if deck in cached_data:
            matchups = cached_data[deck]
        else:
            matchups = fetch_matchups(deck, resolver)
            cached_data[deck] = matchups
            cache.save(cached_data)
        if deck not in matrix:
            matrix[deck] = {}

        for m in matchups:
            opp = m['opponent']
            matrix[deck][opp] = {
                'matches': m['total_matches'],
                'wins': int(m['total_matches'] * m['win_rate'] / 100),
                'win_rate': m['win_rate']
            }

            if opp in top_decks:
                if opp not in matrix:
                    matrix[opp] = {}
                matrix[opp][deck] = {
                    'matches': m['total_matches'],
                    'wins': m['total_matches'] - int(m['total_matches'] * m['win_rate'] / 100),
                    'win_rate': 100 - m['win_rate']
                }

        # 分批處理時在每批之後暫停
        if CONFIG.get('BATCH_FETCH') and CONFIG.get('BATCH_SIZE'):
            if (idx + 1) % CONFIG['BATCH_SIZE'] == 0 and (idx + 1) < len(top_decks):
                time.sleep(CONFIG.get('BATCH_DELAY', 1))

    # 補零：確保 TOP 彼此都有紀錄
    for deck in top_decks:
        if deck not in matrix:
            matrix[deck] = {}
        for opp in top_decks:
            matrix[deck].setdefault(opp, {'matches': 0, 'wins': 0, 'win_rate': 0.0})
    
    return matrix

def build_matrix_json(matrix: Dict[str, Dict[str, Dict[str, Any]]], df: pd.DataFrame, resolver: NameResolver) -> str:
    """
    產生包含所有對局、並附 total_matches 的 JSON，
    供 ChatGPT 計算。鏡像對局不計入。
    使用中文牌組名稱。
    """
    # 建立英文到中文的映射
    name_map = {row['Deck Name']: row['Chinese Name'] for _, row in df.iterrows()}
    
    clean = {}
    for deck, opps in matrix.items():
        deck_cn = name_map.get(deck, resolver.get(deck))
        formatted_opps = {}
        for o, v in opps.items():
            opp_cn = name_map.get(o, resolver.get(o))
            formatted_opps[opp_cn] = v
        total = sum(v['matches'] for v in formatted_opps.values())
        clean[deck_cn] = {
            "total_matches": total,
            "opponents": formatted_opps
        }
    return json.dumps(clean, ensure_ascii=False, indent=2)


def build_b_score_data(matrix: Dict[str, Dict[str, Dict[str, Any]]], df: pd.DataFrame, resolver: NameResolver) -> List[Dict[str, Any]]:
    """將對戰矩陣轉換為 B-Score 函式所需的資料格式。"""
    name_map = {row['Deck Name']: row['Chinese Name'] for _, row in df.iterrows()}

    result: List[Dict[str, Any]] = []
    for deck, opps in matrix.items():
        deck_cn = name_map.get(deck, resolver.get(deck))
        opponents = []
        total = 0
        for o, v in opps.items():
            if deck == o:
                continue
            opp_cn = name_map.get(o, resolver.get(o))
            matches = v['matches']
            opponents.append({
                'name': opp_cn,
                'games': matches,
                'win_rate': v['win_rate'] / 100
            })
            total += matches
        result.append({
            'deck': deck_cn,
            'total_matches': total,
            'opponents': opponents
        })

    return result


# ============================================================
# Helper: DataFrame -> Markdown table
# ============================================================

def df_to_markdown(df: pd.DataFrame) -> str:
    header = '| ' + ' | '.join(df.columns) + ' |'
    divider = '| ' + ' | '.join(['---'] * len(df.columns)) + ' |'
    rows = ['| ' + ' | '.join(map(str, r)) + ' |' for r in df.values]
    return '\n'.join([header, divider] + rows)

# ============================================================
# Compute deck scores locally (no ChatGPT)
# ============================================================

def compute_scores(df: pd.DataFrame, matrix: Dict[str, Dict[str, Dict[str, Any]]], b_results: List[Dict[str, Any]]):
    """Calculate detailed scores for each deck based on predefined rules."""

    en_to_cn = {row['Deck Name']: row['Chinese Name'] for _, row in df.iterrows()}
    cn_to_en = {v: k for k, v in en_to_cn.items()}

    results: Dict[str, Any] = {
        'top_decks': [],
        'defective_decks': []
    }

    for item in b_results:
        deck_cn = item['deck']
        deck_en = cn_to_en.get(deck_cn, deck_cn)
        deck_matrix = matrix.get(deck_en, {})

        total_matches = sum(m.get('games', 0) for m in deck_matrix.values())

        # Sample size score
        if total_matches >= 80:
            sample_score = 2
        elif total_matches >= 50:
            sample_score = 1
        elif total_matches < 30:
            sample_score = -1
        else:
            sample_score = 0

        advantages: List[Dict[str, Any]] = []
        disadvantages: List[Dict[str, Any]] = []
        adv_score = 0

        for opp_en, matchup in deck_matrix.items():
            if not matchup:
                continue
            opp_cn = en_to_cn.get(opp_en, opp_en)
            win_rate = round(matchup.get('win_rate', 0), 2)
            games = matchup.get('matches', matchup.get('games', 0))

            entry = {
                'opponent': opp_cn,
                'win_rate': win_rate,
                'games': games
            }

            if win_rate >= 65 and games >= 20:
                entry['points'] = 2
                adv_score += 2
                advantages.append(entry)
            elif win_rate >= 60 and games >= 15:
                entry['points'] = 1
                adv_score += 1
                advantages.append(entry)
            elif win_rate <= 35 and games >= 20:
                entry['points'] = -2
                adv_score -= 2
                disadvantages.append(entry)
            elif win_rate <= 40 and games >= 15:
                entry['points'] = -1
                adv_score -= 1
                disadvantages.append(entry)

        b_score_val = item['B-Score']
        if b_score_val >= 55:
            b_score_score = 2
        elif b_score_val >= 50:
            b_score_score = 1
        elif b_score_val <= 45:
            b_score_score = -1
        else:
            b_score_score = 0

        total_score = sample_score + adv_score + b_score_score

        deck_data = {
            'deck': deck_cn,
            'B-Score': b_score_val,
            'total_matches': total_matches,
            'sample_score': sample_score,
            'advantage_score': adv_score,
            'b_score_score': b_score_score,
            'total_score': total_score,
            'advantages': advantages,
            'disadvantages': disadvantages,
        }

        is_defective = (
            b_score_val < 45 or
            total_matches < 30 or
            len(disadvantages) > 0
        )

        if is_defective:
            results['defective_decks'].append(deck_data)
        else:
            results['top_decks'].append(deck_data)

    results['top_decks'].sort(key=lambda x: x['total_score'], reverse=True)
    results['defective_decks'].sort(key=lambda x: x['total_score'])
    return results

# ============================================================
# Ask ChatGPT (o3) which deck is currently strongest
# ============================================================

def load_prompt(filename: str) -> str:
    """Load prompt from file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logging.error(f"Error loading prompt file {filename}: {e}")
        return ""

def ask_chatgpt(df, matrix, resolver, b_results):
    """Ask ChatGPT to analyze the data and provide a verdict"""
    try:
        # 檢查 API key
        if not openai_api_key:
            logging.warning("未找到 OPENAI_API_KEY，請在環境變數或 config.json 設定 openai_api_key")
            return "[未設定 OPENAI_API_KEY，跳過 ChatGPT 推論]"

        # 準備要發送給 GPT 的數據
        data = {
            'top_decks': [],
            'defective_decks': []
        }
        
        # 處理每個牌組的數據
        for item in b_results:
            deck_name = item['deck']
            deck_matrix = matrix.get(deck_name, {})
            
            # 計算總場次
            total_matches = sum(matchup.get('games', 0) for matchup in deck_matrix.values())
            
            deck_data = {
                'deck': deck_name,
                'B-Score': item['B-Score'],
                'total_matches': total_matches,
                'advantages': [],
                'disadvantages': []
            }
            
            # 分析對戰數據
            for opp_name, matchup in deck_matrix.items():
                if not matchup:  # 跳過空數據
                    continue
                    
                matchup_data = {
                    'opponent': opp_name,
                    'win_rate': round(matchup.get('win_rate', 0), 2),
                    'games': matchup.get('games', 0)
                }
                
                # 根據勝率和場次計算得分
                win_rate = matchup_data['win_rate']
                games = matchup_data['games']
                
                if win_rate >= 0.65 and games >= 20:
                    matchup_data['points'] = 2
                    deck_data['advantages'].append(matchup_data)
                elif win_rate >= 0.60 and games >= 15:
                    matchup_data['points'] = 1
                    deck_data['advantages'].append(matchup_data)
                elif win_rate <= 0.35 and games >= 20:
                    matchup_data['points'] = -2
                    deck_data['disadvantages'].append(matchup_data)
                elif win_rate <= 0.40 and games >= 15:
                    matchup_data['points'] = -1
                    deck_data['disadvantages'].append(matchup_data)
            
            # 根據 B-Score 和樣本量決定是否為缺陷牌組
            is_defective = (
                item['B-Score'] < 45 or  # B-Score 過低
                total_matches < 30 or  # 樣本量過少
                len(deck_data['disadvantages']) > 0  # 有明顯劣勢對戰
            )
            
            if is_defective:
                data['defective_decks'].append(deck_data)
            else:
                data['top_decks'].append(deck_data)
        
        # 排序牌組
        data['top_decks'].sort(key=lambda x: x['B-Score'], reverse=True)
        data['defective_decks'].sort(key=lambda x: x['B-Score'])
        
        # 讀取 system prompt
        try:
            with open('system_prompt.txt', 'r', encoding='utf-8') as f:
                system_prompt = f.read()
        except Exception as e:
            logging.error(f"讀取 system_prompt.txt 失敗: {e}")
            return f"[讀取 system_prompt.txt 失敗: {e}]"
        
        # 準備發送給 GPT 的完整提示
        prompt = f"{system_prompt}\n\n請分析以下數據：\n{json.dumps(data, ensure_ascii=False, indent=2)}"
        
        # 記錄提示內容
        logging.debug(f"發送給 GPT 的提示：\n{prompt}")
        
        # 調用 GPT API
        try:
            client = openai.OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(data, ensure_ascii=False, indent=2)}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"ChatGPT API 錯誤: {e}")
            return f"[ChatGPT API 錯誤: {e}]"
            
    except Exception as e:
        logging.error(f"ask_chatgpt 函數錯誤: {e}")
        return f"[ask_chatgpt 函數錯誤: {e}]"

# ============================================================
# Main entry
# ============================================================

def main():
    deck_cache = JsonCache(CONFIG['DECK_CACHE'], CONFIG['CACHE_DURATION'])
    matchup_cache = JsonCache(CONFIG['MATCHUP_CACHE'], CONFIG['CACHE_DURATION'])
    resolver = NameResolver(CONFIG['NAME_CACHE'])

    # 1) Load or scrape deck list
    cached_decks = None if CONFIG['FORCE_FRESH'] else deck_cache.load()
    if cached_decks:
        df = pd.DataFrame(cached_decks)
    else:
        df = fetch_decks()
        df['Chinese Name'] = df['Deck Name'].apply(resolver.get)
        deck_cache.save(df.to_dict('records'))

    # 2) Build matchup matrix
    matrix = analyze_top_matchups(df, resolver, matchup_cache)

    # 3) Pretty print deck basics
    name_w = 15
    print("\n=== TOP DECK 基本數據 ===")
    print(f"{'中文名稱':<{name_w}} {'卡表數量':<8} {'Share':<8} {'Win %':<8}")
    print("-" * 40)
    for _, row in df.iterrows():
        print(f"{row['Chinese Name']:<{name_w}} {row['Count']:<8} {row['Share']:<8.1f} {row['Win %']:<8.1f}")

    def print_matrix(title: str, value_fn):
        print(f"\n{title}:")
        print("對手".ljust(name_w), end='')
        for deck in df['Deck Name']:
            print(f"{row if (row := df[df['Deck Name'] == deck]['Chinese Name'].iloc[0]) else deck}"[:name_w-2].ljust(name_w), end='')
        print()
        for deck in df['Deck Name']:
            deck_cn = df[df['Deck Name'] == deck]['Chinese Name'].iloc[0]
            print(deck_cn[:name_w-2].ljust(name_w), end='')
            for opp in df['Deck Name']:
                if deck == opp:
                    # 顯示對戰自己的數據
                    val = value_fn(matrix[deck][opp])
                    print(str(val).center(name_w), end='')
                else:
                    val = value_fn(matrix[deck][opp])
                    print(str(val).center(name_w), end='')
            print()

    print_matrix("對戰次數矩陣", lambda x: x['matches'])
    print_matrix("勝率矩陣", lambda x: f"{x['win_rate']:.1f}%")

    # 4) Calculate B-Score
    b_score_data = build_b_score_data(matrix, df, resolver)
    b_results = calculate_b_score(b_score_data)
    print("\n=== B-Score 結果 ===")
    for item in b_results:
        print(f"{item['deck']}: {item['B-Score']}%")

    # 5) Calculate deck scores locally
    scores = compute_scores(df, matrix, b_results)
    print("\n=== 計分結果 ===")
    for deck in scores['top_decks']:
        print(f"{deck['deck']} 總分:{deck['total_score']} (樣本量{deck['sample_score']}, "+
              f"對戰優勢{deck['advantage_score']}, B-Score{deck['b_score_score']})")
        if deck['advantages']:
            print("  加分:")
            for adv in deck['advantages']:
                print(f"    +{adv['points']} {adv['opponent']} {adv['win_rate']:.1f}% ({adv['games']}場)")
        if deck['disadvantages']:
            print("  扣分:")
            for dis in deck['disadvantages']:
                print(f"    {dis['points']} {dis['opponent']} {dis['win_rate']:.1f}% ({dis['games']}場)")

    if scores['defective_decks']:
        print("\n=== 有缺陷牌組 ===")
        for deck in scores['defective_decks']:
            print(f"{deck['deck']} 總分:{deck['total_score']}")
            if deck['disadvantages']:
                print("  主要缺陷:")
                for dis in deck['disadvantages']:
                    print(f"    {dis['points']} {dis['opponent']} {dis['win_rate']:.1f}% ({dis['games']}場)")

    # 6) Save scraped data to JSON file
    result = {
        'decks': df.to_dict('records'),
        'matrix': matrix,
        'scores': scores,
    }
    try:
        with open('result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logging.info('已將資料寫入 result.json')
    except Exception as e:
        logging.error(f'寫入 result.json 失敗: {e}')


if __name__ == '__main__':
    main()

def get_pokemon_chinese_name(name: str) -> str:
    """Get Chinese name for a Pokemon deck name."""
    # 處理 Tapu 後接空白的寶可夢名稱
    name = re.sub(r'Tapu\s+(\w+)', r'Tapu\1', name)
    
    # 分割牌組名稱（通常是兩隻寶可夢）
    parts = name.split(' ')
    zh_parts = []
    
    for part in parts:
        if part.lower() == 'ex':
            if zh_parts:
                zh_parts[-1] += ' EX'
        else:
            zh_parts.append(_query_wiki(part))
    
    # 用加號連接兩隻寶可夢的名稱
    return ' + '.join(zh_parts)
