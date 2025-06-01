import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from datetime import datetime, timedelta
import re
import time
import os

# Configuration parameters
CONFIG = {
    'MIN_COUNT': 40,        # Minimum count threshold
    'MIN_WIN_RATE': 50.0,   # Minimum win rate threshold (%)
    'TOP_N': 10,           # Number of top decks to display
    'CACHE_FILE': 'pokemon_name_cache.json',  # Cache file path
    'DECK_CACHE_FILE': 'deck_data_cache.json',  # Deck data cache file
    'CACHE_DURATION': 3600,  # Cache duration in seconds (1 hour)
    'USE_CACHE': False,     # Whether to use cache
    'DEBUG': True,         # Whether to show detailed logs
    'MIN_MATCH_COUNT': 5    # Minimum number of matches for a matchup to be considered
}

def debug_print(*args, **kwargs):
    """Print debug messages only when DEBUG is enabled"""
    if CONFIG['DEBUG']:
        print(*args, **kwargs)

def load_name_cache():
    if os.path.exists(CONFIG['CACHE_FILE']):
        try:
            with open(CONFIG['CACHE_FILE'], 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Cache file is corrupted: {e}")
            # If cache is corrupted, create a new one
            return {}
        except Exception as e:
            print(f"Error loading cache: {e}")
            return {}
    return {}

def save_name_cache(cache):
    try:
        # Create backup of existing cache if it exists
        if os.path.exists(CONFIG['CACHE_FILE']):
            backup_file = f"{CONFIG['CACHE_FILE']}.bak"
            try:
                with open(CONFIG['CACHE_FILE'], 'r', encoding='utf-8') as src:
                    with open(backup_file, 'w', encoding='utf-8') as dst:
                        dst.write(src.read())
            except Exception as e:
                print(f"Warning: Could not create cache backup: {e}")
        
        # Save new cache
        with open(CONFIG['CACHE_FILE'], 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving cache: {e}")
        # Try to restore from backup if save failed
        if os.path.exists(f"{CONFIG['CACHE_FILE']}.bak"):
            try:
                with open(f"{CONFIG['CACHE_FILE']}.bak", 'r', encoding='utf-8') as src:
                    with open(CONFIG['CACHE_FILE'], 'w', encoding='utf-8') as dst:
                        dst.write(src.read())
            except Exception as e:
                print(f"Warning: Could not restore cache from backup: {e}")

def get_pokemon_chinese_name(english_name, name_cache):
    # Split the name into parts
    parts = english_name.split(' ')
    chinese_parts = []
    
    # Process each part
    for i, part in enumerate(parts):
        if part == 'ex':
            # If this part is 'ex', add it to the previous part's Chinese name
            if chinese_parts:
                chinese_parts[-1] += " EX"
        else:
            # Get Chinese name for this part
            chinese_name = get_single_pokemon_chinese_name(part, name_cache)
            if chinese_name:
                chinese_parts.append(chinese_name)
    
    # Join all parts with spaces
    return ' '.join(chinese_parts)

def get_single_pokemon_chinese_name(english_name, name_cache):
    # Check cache first
    if english_name in name_cache:
        return name_cache[english_name]
    
    try:
        # Construct the wiki URL
        wiki_url = f"http://wiki.52poke.com/wiki/{english_name}"
        
        # Make the request
        response = requests.get(wiki_url)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the Chinese name in the title
        title = soup.find('h1', {'id': 'firstHeading'})
        if title:
            chinese_name = title.text.strip()
            # Save to cache
            name_cache[english_name] = chinese_name
            save_name_cache(name_cache)
            return chinese_name
        
        return english_name
    except Exception as e:
        print(f"Error getting Chinese name for {english_name}: {e}")
        return english_name

def load_deck_cache():
    if os.path.exists(CONFIG['DECK_CACHE_FILE']):
        try:
            with open(CONFIG['DECK_CACHE_FILE'], 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                # Check if cache is still valid
                cache_time = datetime.fromisoformat(cache_data['timestamp'])
                if datetime.now() - cache_time < timedelta(seconds=CONFIG['CACHE_DURATION']):
                    # Convert the cached data back to DataFrame
                    return pd.DataFrame(cache_data['data'])
        except json.JSONDecodeError as e:
            print(f"Deck cache file is corrupted: {e}")
            return None
        except Exception as e:
            print(f"Error loading deck cache: {e}")
            return None
    return None

def save_deck_cache(df):
    try:
        # Create backup of existing cache if it exists
        if os.path.exists(CONFIG['DECK_CACHE_FILE']):
            backup_file = f"{CONFIG['DECK_CACHE_FILE']}.bak"
            try:
                with open(CONFIG['DECK_CACHE_FILE'], 'r', encoding='utf-8') as src:
                    with open(backup_file, 'w', encoding='utf-8') as dst:
                        dst.write(src.read())
            except Exception as e:
                print(f"Warning: Could not create deck cache backup: {e}")
        
        # Save new cache
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'data': df.to_dict('records') if hasattr(df, 'to_dict') else df
        }
        with open(CONFIG['DECK_CACHE_FILE'], 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving deck cache: {e}")
        # Try to restore from backup if save failed
        if os.path.exists(f"{CONFIG['DECK_CACHE_FILE']}.bak"):
            try:
                with open(f"{CONFIG['DECK_CACHE_FILE']}.bak", 'r', encoding='utf-8') as src:
                    with open(CONFIG['DECK_CACHE_FILE'], 'w', encoding='utf-8') as dst:
                        dst.write(src.read())
            except Exception as e:
                print(f"Warning: Could not restore deck cache from backup: {e}")

def load_matchup_cache():
    """載入對戰數據緩存"""
    cache_file = 'matchup_cache.json'
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                # 檢查緩存是否過期
                cache_time = datetime.fromisoformat(cache_data['timestamp'])
                if datetime.now() - cache_time < timedelta(seconds=CONFIG['CACHE_DURATION']):
                    return cache_data['data']
        except Exception as e:
            print(f"Error loading matchup cache: {e}")
    return {}

def save_matchup_cache(cache_data):
    """保存對戰數據緩存"""
    cache_file = 'matchup_cache.json'
    try:
        # 創建備份
        if os.path.exists(cache_file):
            backup_file = f"{cache_file}.bak"
            try:
                with open(cache_file, 'r', encoding='utf-8') as src:
                    with open(backup_file, 'w', encoding='utf-8') as dst:
                        dst.write(src.read())
            except Exception as e:
                print(f"Warning: Could not create matchup cache backup: {e}")
        
        # 保存新緩存
        cache = {
            'timestamp': datetime.now().isoformat(),
            'data': cache_data
        }
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving matchup cache: {e}")
        # 如果保存失敗，嘗試從備份恢復
        if os.path.exists(f"{cache_file}.bak"):
            try:
                with open(f"{cache_file}.bak", 'r', encoding='utf-8') as src:
                    with open(cache_file, 'w', encoding='utf-8') as dst:
                        dst.write(src.read())
            except Exception as e:
                print(f"Warning: Could not restore matchup cache from backup: {e}")

def test_matchup_data_structure(url):
    """Test function to verify the matchup data structure"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        
        if not table:
            print("No table found")
            return
            
        # Get all rows except header
        rows = table.find_all('tr')[1:]
        if not rows:
            print("No data rows found")
            return
            
        print(f"\nFound {len(rows)} rows")
        print("\nAnalyzing first 5 rows:")
        
        for i, row in enumerate(rows[:5]):
            print(f"\nRow {i+1}:")
            cols = row.find_all('td')
            for j, col in enumerate(cols):
                print(f"Column {j}: {col.get_text(strip=True)}")
                # Print HTML structure for debugging
                print(f"HTML: {col}")
                
        return True
    except Exception as e:
        print(f"Error in test: {e}")
        return False

def get_deck_matchups(deck_name, deck_cache, use_cache=True, top_decks=None):
    debug_print(f"\nProcessing deck: {deck_name}")
    
    # 檢查緩存
    if use_cache:
        matchup_cache = load_matchup_cache()
        if deck_name in matchup_cache:
            debug_print(f"Found in matchup cache: {deck_name}")
            return matchup_cache[deck_name]
    
    try:
        # First, get the deck list page to find the matchup URL
        list_url = "https://play.limitlesstcg.com/decks?game=pocket"
        
        response = requests.get(list_url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        
        if not table:
            debug_print("No deck table found")
            return None
            
        # Find the row for this deck
        matchup_url = None
        for row in table.find_all('tr')[1:]:  # Skip header row
            cols = row.find_all('td')
            if len(cols) >= 3:
                current_deck = cols[2].get_text(strip=True)
                if current_deck.lower() == deck_name.lower():
                    # Find the matchup link in the row
                    matchup_link = row.find('a', href=lambda x: x and 'matchups' in x)
                    if matchup_link:
                        matchup_url = f"https://play.limitlesstcg.com{matchup_link['href']}"
                        debug_print(f"Found matchup URL: {matchup_url}")
                        break
        
        if not matchup_url:
            debug_print(f"No matchup URL found for deck: {deck_name}")
            return None
            
        # Now fetch the matchup data
        response = requests.get(matchup_url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        
        if not table:
            debug_print("No matchup table found in the response")
            return None
            
        # Get all rows except header
        rows = table.find_all('tr')[1:]
        if not rows:
            debug_print("No matchup data rows found in the table")
            return None
        
        debug_print(f"Found {len(rows)} matchup rows")
        
        # Process all matchups
        all_matchups = []
        for row in rows:
            try:
                cols = row.find_all('td')
                if len(cols) < 3:
                    continue
                
                # Get opponent name from the first column
                opponent_cell = cols[1]
                opponent = opponent_cell.get_text(strip=True)
                debug_print(f"\nProcessing opponent: {opponent}")
                
                # Get score and matches
                matches_text = cols[2].get_text(strip=True)  # Total matches
                score_text = cols[3].get_text(strip=True)    # W-L-T record
                debug_print(f"Raw score: {score_text}")
                debug_print(f"Raw matches: {matches_text}")
                
                try:
                    # 檢查場次是否為數字
                    if not matches_text.isdigit():
                        debug_print(f"Invalid match count: {matches_text}")
                        continue
                        
                    # 解析總場次
                    total_matches = int(matches_text)
                    
                    # 如果場次少於最小場次，跳過
                    if total_matches < CONFIG['MIN_MATCH_COUNT']:
                        debug_print(f"Skipping matchup with less than {CONFIG['MIN_MATCH_COUNT']} matches: {total_matches}")
                        continue
                    
                    # 解析勝率
                    # 格式可能是 "W-L-T" 或純數字百分比
                    if '-' in score_text:
                        # 處理 W-L-T 格式
                        wins, losses, ties = map(int, score_text.split('-'))
                        win_rate = (wins / total_matches) * 100
                    else:
                        # 處理純數字百分比
                        win_rate = float(score_text.replace('%', ''))
                    
                    # Convert opponent name to Chinese
                    name_cache = load_name_cache()
                    opponent_chinese = get_pokemon_chinese_name(opponent, name_cache)
                    
                    matchup_data = {
                        'opponent': opponent,
                        'opponent_chinese': opponent_chinese,
                        'win_rate': win_rate,
                        'raw_score': score_text,
                        'total_matches': total_matches
                    }
                    
                    all_matchups.append(matchup_data)
                    debug_print(f"Added matchup: {opponent_chinese} - {total_matches} matches, {win_rate:.1f}% win rate")
                except ValueError as e:
                    debug_print(f"Error parsing data: {e}")
                    continue
                    
            except Exception as e:
                debug_print(f"Error processing row: {e}")
                continue
        
        debug_print(f"Total valid matchups found: {len(all_matchups)}")
        
        if all_matchups:
            # 只按場次排序
            all_matchups.sort(key=lambda x: x['total_matches'], reverse=True)
            
            # 保存到緩存
            if use_cache:
                matchup_cache = load_matchup_cache()
                matchup_cache[deck_name] = all_matchups
                save_matchup_cache(matchup_cache)
            
            return all_matchups
        else:
            debug_print("No valid matchup data found")
            return None
            
    except requests.RequestException as e:
        debug_print(f"Request error: {e}")
    except Exception as e:
        debug_print(f"Error processing data: {e}")
    
    return None

def analyze_top_deck_matchups(df):
    """分析TOP DECK之間的對戰數據"""
    # 獲取TOP DECK列表
    top_decks = df['Deck Name'].tolist()
    debug_print("\nTop decks list:")
    for deck in top_decks:
        debug_print(f"- {deck}")
    
    # 載入名稱緩存
    name_cache = load_name_cache()
    
    # 創建對戰矩陣
    matchup_matrix = {}
    for deck in top_decks:
        matchup_matrix[deck] = {}
        for opponent in top_decks:
            if deck != opponent:
                matchup_matrix[deck][opponent] = {
                    'matches': 0,
                    'wins': 0,
                    'win_rate': 0.0
                }
    
    # 處理每個TOP DECK的對戰數據
    for deck in top_decks:
        print(f"\n處理 {deck} 的對戰數據...")
        matchups = get_deck_matchups(deck, None, use_cache=True, top_decks=top_decks)
        
        if matchups:
            debug_print(f"\nFound {len(matchups)} matchups for {deck}")
            # 更新對戰矩陣
            for matchup in matchups:
                opponent = matchup['opponent']
                debug_print(f"\nChecking matchup: {deck} vs {opponent}")
                debug_print(f"Matchup data: {matchup}")
                # 檢查對手是否在TOP DECK列表中
                if opponent in top_decks:
                    debug_print(f"Found matchup for {deck} vs {opponent}")
                    matchup_matrix[deck][opponent] = {
                        'matches': matchup['total_matches'],
                        'wins': int(matchup['total_matches'] * matchup['win_rate'] / 100),
                        'win_rate': matchup['win_rate']
                    }
                    debug_print(f"Updated matrix: {deck} vs {opponent} - {matchup['total_matches']} matches, {matchup['win_rate']:.1f}% win rate")
                else:
                    debug_print(f"Opponent {opponent} not in top decks list")
        else:
            debug_print(f"No matchups found for {deck}")
    
    # 獲取所有牌組的中文名稱
    chinese_names = {deck: get_pokemon_chinese_name(deck, name_cache) for deck in top_decks}
    
    # 打印對戰矩陣
    print("\n=== TOP DECK 對戰矩陣 ===")
    
    # 設置固定寬度
    NAME_WIDTH = 15
    NUM_WIDTH = 8
    
    def print_matrix_header():
        print("對手".ljust(NAME_WIDTH), end="")
        for deck in top_decks:
            print(f"{chinese_names[deck][:NAME_WIDTH-2]}".ljust(NAME_WIDTH), end="")
        print()
    
    def print_matrix_row(deck, get_value):
        print(f"{chinese_names[deck][:NAME_WIDTH-2]}".ljust(NAME_WIDTH), end="")
        for opponent in top_decks:
            if deck == opponent:
                print("-".center(NAME_WIDTH), end="")
            else:
                value = get_value(matchup_matrix[deck][opponent])
                print(f"{value}".center(NAME_WIDTH), end="")
        print()
    
    print("\n對戰次數矩陣:")
    print_matrix_header()
    for deck in top_decks:
        print_matrix_row(deck, lambda x: x['matches'])
    
    print("\n勝率矩陣:")
    print_matrix_header()
    for deck in top_decks:
        print_matrix_row(deck, lambda x: f"{x['win_rate']:.1f}%")
    
    return matchup_matrix

def scrape_limitless_decks(use_cache=None):
    # Use the global config if use_cache is not specified
    if use_cache is None:
        use_cache = CONFIG['USE_CACHE']
    
    # Try to load from cache first if enabled
    if use_cache:
        df = load_deck_cache()
        if df is not None:
            debug_print("Using cached deck data...")
            return df
    
    debug_print("Fetching fresh deck data...")
    url = "https://play.limitlesstcg.com/decks?game=pocket"
    
    # Headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Load name cache
        name_cache = load_name_cache()
        
        # Make the request
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the table containing deck data
        deck_data = []
        table = soup.find('table')
        
        if table:
            rows = table.find_all('tr')[1:]  # Skip header row
            for row in rows:
                try:
                    # Get all columns
                    cols = row.find_all('td')
                    
                    # Skip if not enough columns
                    if len(cols) < 7:  # We need at least 7 columns
                        continue
                    
                    # Extract text from each column
                    deck_name = cols[2].get_text(strip=True)  # Deck name is in column 2
                    count = cols[3].get_text(strip=True)      # Count is in column 3
                    share = cols[4].get_text(strip=True).replace('%', '')  # Share is in column 4
                    win_percentage = cols[6].get_text(strip=True).replace('%', '')  # Win % is in column 6
                    
                    # Convert to appropriate types
                    deck_data.append({
                        'Deck Name': deck_name,
                        'Count': int(count),
                        'Share': float(share),
                        'Win %': float(win_percentage) if win_percentage != 'NaN' else 0.0
                    })
                    
                except (ValueError, IndexError, AttributeError) as e:
                    debug_print(f"Skipping row due to parsing error: {e}")
                    continue
        
        if not deck_data:
            debug_print("No valid deck data found in the table")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(deck_data)
        
        # Apply filters
        df = df[
            (df['Count'] > CONFIG['MIN_COUNT']) & 
            (df['Win %'] > CONFIG['MIN_WIN_RATE'])
        ]
        
        # Sort by Win %
        df = df.sort_values('Win %', ascending=False)
        
        # Get top N decks
        top_n_df = df.head(CONFIG['TOP_N']).copy()
        
        # Get list of top deck names
        top_decks = top_n_df['Deck Name'].tolist()
        
        # Add Chinese names and matchup data
        print(f"\nFetching Chinese names and matchup data for top {CONFIG['TOP_N']} decks...")
        top_n_df['Chinese Name'] = top_n_df['Deck Name'].apply(lambda x: get_pokemon_chinese_name(x, name_cache))
        top_n_df['Matchup Data'] = top_n_df['Deck Name'].apply(lambda x: get_deck_matchups(x, None, use_cache, top_decks))
        
        # Save to cache if enabled
        if use_cache:
            save_deck_cache(top_n_df)
        
        return top_n_df
        
    except requests.RequestException as e:
        debug_print(f"Error fetching data: {e}")
        return None
    except Exception as e:
        debug_print(f"Error processing data: {e}")
        return None

if __name__ == "__main__":
    # 設置配置
    CONFIG['USE_CACHE'] = False  # 強制重新獲取數據
    CONFIG['DEBUG'] = True  # 開啟調試輸出
    
    # 獲取數據
    df = scrape_limitless_decks()
    
    # 分析TOP DECK對戰數據
    matchup_matrix = analyze_top_deck_matchups(df)
    
    # 打印TOP DECK基本數據
    print("\n=== TOP DECK 基本數據 ===")
    print(f"{'中文名稱':<15} {'Count':<8} {'Share':<8} {'Win %':<8}")
    print("-" * 40)
    
    for _, row in df.iterrows():
        print(f"{row['Chinese Name']:<15} {row['Count']:<8} {row['Share']:<8.1f} {row['Win %']:<8.1f}") 