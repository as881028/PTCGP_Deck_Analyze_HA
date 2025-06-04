# PTCGP Deck Analyze 指南

以下是協助開發與維護此專案的基本指引。

## 專案簡介
本倉庫用來評估 *Pokémon TCG Pocket*（PTCGP）各牌組的勝率，
主要程式位於 `ptcgp_deck_analyze/scraper.py`，會從 Limitless 網站抓取牌組資料、產生對戰矩陣，並可選擇透過 ChatGPT 分析最有利的牌組。

## 執行方式
1. 安裝依賴：
   ```bash
   pip install -r ptcgp_deck_analyze/requirements.txt
   ```
2. （可選）設定 `OPENAI_API_KEY` 以啟用 ChatGPT 推論。
3. 執行：
   ```bash
   python ptcgp_deck_analyze/scraper.py
   ```

程式會輸出牌組基本數據、對戰矩陣及 ChatGPT 推論結果。

## 目錄結構
- `ptcgp_deck_analyze/`：主要程式與 Docker 相關檔案。
- `repository.json`：Home Assistant Add-on 的倉庫描述。

## 提交訊息
- 使用簡潔的中英文描述修改內容，例如 `新增勝率計算` 或 `Fix deck parser`。
- 每次提交前請確認 `git status` 乾淨，並附上必要的檔案。
- 提交 PR 時請記得在 `ptcgp_deck_analyze/config.json` 中遞增 `version` 號碼，確保版本更新。
- 若僅是遞增版本號，可省略詳細檢查並直接送出 PR。

## 測試
本專案目前沒有自動化測試腳本，修改後請盡量手動執行程式確認無誤。

