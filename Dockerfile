FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git nodejs npm && \
    rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code@2.1.85

RUN pip install --no-cache-dir --break-system-packages \
    "numpy==2.2.6" \
    "pandas==2.3.3" \
    "scipy==1.15.3" \
    "scikit-learn==1.7.2" \
    "requests==2.32.5" \
    "pyarrow==20.0.0"

WORKDIR /root
