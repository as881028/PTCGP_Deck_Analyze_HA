ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base:3.18
FROM ${BUILD_FROM}

# 安裝 Python + 基本工具
RUN apk add --no-cache python3 py3-pip

# 如果有像 pandas / numpy 這類需要編譯的套件，需要多裝這些編譯工具
RUN apk add --no-cache gcc musl-dev libffi-dev python3-dev libc-dev make

# 你的其他前置檔案
COPY requirements.txt /app/requirements.txt

# 安裝 Python 依賴
RUN python3 -m pip install -r /app/requirements.txt

# 複製你的程式碼（如果有）
COPY . /app/ptcgp_deck_analyze
# Debug: 確認程式碼路徑是否正確
RUN ls -al /app/ptcgp_deck_analyze

# 確保啟動腳本可以執行
RUN chmod +x /app/ptcgp_deck_analyze/run.sh

# 設定工作目錄
WORKDIR /app

# 以 run.sh 為進入點，讀取使用者提供的 OPENAI_API_KEY
ENTRYPOINT ["/app/ptcgp_deck_analyze/run.sh"]
