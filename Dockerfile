FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download ffmpeg.wasm files locally so we avoid CORS/COEP issues at runtime
RUN mkdir -p /app/static/ffmpeg && \
    wget -q -O /app/static/ffmpeg/ffmpeg.js \
        "https://unpkg.com/@ffmpeg/ffmpeg@0.12.10/dist/umd/ffmpeg.js" && \
    wget -q -O /app/static/ffmpeg/util.js \
        "https://unpkg.com/@ffmpeg/util@0.12.1/dist/umd/index.js" && \
    wget -q -O /app/static/ffmpeg/ffmpeg-core.js \
        "https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd/ffmpeg-core.js" && \
    wget -q -O /app/static/ffmpeg/ffmpeg-core.wasm \
        "https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd/ffmpeg-core.wasm"

COPY . .
EXPOSE 3005
CMD ["gunicorn", "--bind", "0.0.0.0:3005", "--timeout", "3600", "--worker-class", "gthread", "--workers", "1", "--threads", "8", "app:app"]
