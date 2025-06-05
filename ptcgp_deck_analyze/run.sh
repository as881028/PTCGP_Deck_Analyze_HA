#!/usr/bin/env bash
source /usr/lib/bashio/bashio.sh

echo "===== Add-on 啟動中 ====="
# 讀取 HA Add-on 中使用者設定的 key
export OPENAI_API_KEY=$(bashio::config 'openai_api_key')
echo $(bashio::config 'openai_api_key')

python /app/ptcgp_deck_analyze/scraper.py
echo "===== Add-on 執行完畢 ====="
