ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base:3.18
FROM ${BUILD_FROM}

# 安裝 Python 和依賴
RUN apk add --no-cache python3 py3-pip
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# 複製你的程式
COPY run.sh /app/run.sh
COPY your_script.py /app/your_script.py
WORKDIR /app

# 啟動命令
CMD [ "bash", "/app/run.sh" ]
