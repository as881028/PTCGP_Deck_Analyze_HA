ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base-python:3.11
FROM ${BUILD_FROM}

# 安裝必要套件 (如有其他需求可加)
RUN apk add --no-cache bash curl

# 複製檔案
COPY run.sh /run.sh
COPY my_script.py /my_script.py
COPY requirements.txt /requirements.txt
COPY system_prompt.txt /system_prompt.txt

# 安裝 Python 套件
RUN pip install --no-cache-dir -r /requirements.txt

# 給執行權限
RUN chmod a+x /run.sh

# 啟動
CMD [ "/run.sh" ]
