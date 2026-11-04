FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
COPY gem_learning_engine ./gem_learning_engine
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip install --no-deps .

COPY configs ./configs
COPY data ./data
COPY docs ./docs
COPY QUICK_START.md ./QUICK_START.md

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/health', timeout=3)" || exit 1

CMD ["gem-learning-engine-gui", "--workspace", "/app", "--host", "0.0.0.0", "--port", "8080", "--insecure"]
