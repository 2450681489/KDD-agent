FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:/root/.local/bin:${PATH}"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gzip \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY main.py entrypoint.sh ./

RUN uv sync --frozen --no-dev \
    && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
