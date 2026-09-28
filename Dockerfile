# Masdar's chat page in a container, for hosting a demo.
#
#   docker build -t masdar .
#   docker run -p 8000:8000 --env-file .env masdar
#
# Set MASDAR_ACCESS_CODE in .env before exposing it: the server then asks
# for the code before the first question. Credentials stay in .env and are
# passed at run time; none is baked into the image.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 MASDAR_CACHE_DIR=/data/cache
WORKDIR /app
COPY pyproject.toml README.md ./
COPY masdar ./masdar
RUN pip install --no-cache-dir ".[ai]" && useradd --create-home masdar \
    && mkdir -p /data && chown masdar /data
USER masdar
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"
CMD ["masdar", "serve", "--host", "0.0.0.0", "--port", "8000", "--out", "/data/out"]
