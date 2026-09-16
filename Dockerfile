FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    DATA_DIR=/app/data \
    ARTIFACT_DIR=/app/artifacts

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home appuser
COPY pyproject.toml README.md ./
COPY requirements/constraints.txt requirements/constraints.txt
COPY src/ src/
RUN python -m pip install --constraint requirements/constraints.txt . \
    && python -m pip check
COPY app/ app/
COPY .streamlit/ .streamlit/
COPY reports/model_cards/ reports/model_cards/
RUN mkdir -p data/processed artifacts && chown -R appuser:appuser /app
USER 10001:10001
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"
CMD ["python", "-m", "streamlit", "run", "app/Home.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.fileWatcherType=none"]
