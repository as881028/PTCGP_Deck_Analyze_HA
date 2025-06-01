FROM alpine:3.18

# 安裝 Python + 基本工具
RUN apk add --no-cache python3 py3-pip

# 如果有像 pandas / numpy 這類需要編譯的套件，需要多裝這些編譯工具
RUN apk add --no-cache gcc musl-dev libffi-dev

# 你的其他前置檔案
COPY requirements.txt /app/requirements.txt

# 先更新 pip
RUN pip install --upgrade pip

# 安裝 Python 依賴
RUN pip install -r /app/requirements.txt

# 複製你的程式碼（如果有）
COPY . /app

# 設定工作目錄
WORKDIR /app

# 預設啟動指令（如果有 python 腳本）
ENTRYPOINT ["python", "/app/scraper.py"]
