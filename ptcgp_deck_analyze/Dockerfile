FROM python:3.11-slim

# 安裝必要套件與編譯工具
RUN apt-get update && apt-get install -y --no-install-recommends \
        bash curl wget gcc g++ make git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 安裝 bashio 以讀取 Add-on 設定
ENV BASHIO_VERSION=0.17.0
RUN wget -q https://github.com/hassio-addons/bashio/archive/v${BASHIO_VERSION}.tar.gz -O /tmp/bashio.tar.gz \
    && tar -xzf /tmp/bashio.tar.gz -C /tmp \
    && mv /tmp/bashio-${BASHIO_VERSION}/lib /usr/lib/bashio \
    && ln -s /usr/lib/bashio/bashio /usr/local/bin/bashio \
    && rm -rf /tmp/bashio.tar.gz /tmp/bashio-${BASHIO_VERSION}

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
