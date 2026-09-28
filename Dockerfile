# Masdar's chat page in a container, for hosting a demo on a public link.
#
#   docker build -t masdar .
#   docker run -p 8000:8000 --env-file .env masdar
#
# Hosting platforms set PORT and the server listens on it (8000 otherwise).
# Set MASDAR_ACCESS_CODE before exposing it: the page then asks for the code
# before the first question. Credentials are passed at run time; none is
# baked into the image.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    MASDAR_CACHE_DIR=/data/cache \
    MASDAR_WARMUP_ON_START=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY masdar ./masdar
RUN pip install --no-cache-dir ".[ai]" \
    && useradd --create-home masdar \
    && mkdir -p /data \
    && chown masdar /data
USER masdar
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT', '8000'), timeout=4)"
CMD ["masdar", "serve", "--host", "0.0.0.0", "--out", "/data/out"]
