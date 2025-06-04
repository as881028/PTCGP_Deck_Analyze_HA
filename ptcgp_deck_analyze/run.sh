#!/usr/bin/with-contenv bashio

# 讀取 HA Add-on 中使用者設定的 key
export OPENAI_API_KEY=$(bashio::config 'openai_api_key')

echo "===== Add-on 啟動中 ====="
python /app/scraper.py
echo "===== Add-on 執行完畢 ====="
