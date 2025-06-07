import os
import json
import logging
import re
import time
import requests
import pandas as pd
import openai
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

請依系統算法輸出三段結果：
計算表
判斷依據
結論
"""
    }
}

# OpenAI API key – set environment variable OPENAI_API_KEY=<your key>
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

addon_options = os.getenv("HASSIO_ADDON_OPTIONS")
if addon_options:
    try:
        options = json.loads(addon_options)
        if 'openai_api_key' in options:
            openai_api_key = options['openai_api_key']
    except Exception as e:
        print("❌ 無法解析 HASSIO_ADDON_OPTIONS:", e)

# openai 庫自 v1 起不再讀取全域變數，因此需在建立客戶端時傳入
openai.api_key = openai_api_key

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
        url = f"http://wiki.52poke.com/wiki/{name}"
        try:
            res = requests.get(url, timeout=8)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, 'html.parser')
            h1 = soup.find('h1', {'id': 'firstHeading'})
            if h1:
                return h1.text.strip()
        except Exception as e:
            logging.debug(f"Wiki lookup failed for {name}: {e}")
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


# ============================================================
# Helper: DataFrame -> Markdown table
# ============================================================

def df_to_markdown(df: pd.DataFrame) -> str:
    header = '| ' + ' | '.join(df.columns) + ' |'
    divider = '| ' + ' | '.join(['---'] * len(df.columns)) + ' |'
    rows = ['| ' + ' | '.join(map(str, r)) + ' |' for r in df.values]
    return '\n'.join([header, divider] + rows)

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

def ask_chatgpt(deck_df: pd.DataFrame, matrix: Dict[str, Dict[str, Dict[str, Any]]], resolver: NameResolver) -> str:
    """Send deck stats + matchup matrix to ChatGPT o3 and get the verdict."""
    matrix_json = build_matrix_json(matrix, deck_df, resolver)

    # Load system prompt from file
    system_prompt = load_prompt(CONFIG['CHATGPT']['PROMPT_FILES']['SYSTEM'])

    # Use template from config
    user_prompt = CONFIG['CHATGPT']['USER_PROMPT_TEMPLATE'].format(
        deck_count=len(deck_df),
        matchup_matrix=matrix_json
    )

    if not openai.api_key:
        logging.warning(
            "未找到 OPENAI_API_KEY，請在環境變數或 config.json 設定 openai_api_key。跳過 ChatGPT 推論"
        )
        print("\n=== ChatGPT 推論輸入 (system) ===")
        print(system_prompt)
        print("\n=== ChatGPT 推論輸入 (user) ===")
        print(user_prompt)
        return "[未設定 OPENAI_API_KEY，跳過 ChatGPT 推論]"

    try:
        client = openai.OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model=CONFIG['CHATGPT']['MODEL'],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"ChatGPT API error: {e}")
        return f"[ChatGPT API 錯誤: {e}]"

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

    # 4) Ask ChatGPT for verdict
    verdict = ask_chatgpt(df, matrix, resolver)
    print("\n=== ChatGPT o3-mini 推論結果 ===")
    print(verdict)


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
