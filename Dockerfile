FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace/OmniDistill-NAS

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN python -m pip install --upgrade pip \
    && python -m pip install -e ".[dev,docs]"

CMD ["omnidistill", "--help"]
