# ML Automation — production image (the "web" service; the worker and API
# services in later roadmap items reuse it with a different command).
#
# slim needs no compiler: every wheel here ships manylinux builds. curl is for
# the health check; libgomp1 because LightGBM's wheel links the system OpenMP
# runtime, without which `import lightgbm` fails.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Matplotlib rebuilds its font cache when it cannot write a config dir,
    # which adds seconds to the first PowerPoint export.
    MPLCONFIGDIR=/tmp/mpl \
    PORT=8501 \
    # Behind the ALB in production; "dev" only for a local run.
    AUTH_MODE=alb

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# The lock first: this layer is cached until the dependencies change. Exact
# versions with hashes, so every build installs the same packages.
COPY requirements.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

COPY . .

# Non-root. Nothing writes to the app directory at runtime: uploads are held
# in memory and credentials arrive as environment variables.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD curl -fsS "http://localhost:${PORT}/_stcore/health" || exit 1

# fileWatcherType=none: the container never edits its own source.
CMD streamlit run app.py \
    --server.port="${PORT}" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false
