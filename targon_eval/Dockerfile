FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --break-system-packages \
    "numpy==2.2.6" \
    "pandas==2.3.3" \
    "scipy==1.15.3" \
    "scikit-learn==1.7.2" \
    "requests==2.32.5" \
    "pyarrow==20.0.0" \
    "fastapi==0.115.12" \
    "uvicorn[standard]==0.34.3" \
    "boto3>=1.35" \
    "bittensor>=7.0" \
    "cryptography>=42"

COPY . /app/mantis_model_iteration_tool/

WORKDIR /app

ENV PORT=8080
EXPOSE 8080

CMD ["python", "-m", "mantis_model_iteration_tool.targon_eval.server"]
