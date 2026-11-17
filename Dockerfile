# =============================================================================
# Dockerfile — immagine dell'API.
#
# Ollama NON gira nel container: su Mac l'accelerazione GPU (Metal) funziona solo
# con Ollama installato nativamente. Il container lo raggiunge tramite
# host.docker.internal (vedi docker-compose.yml).
# Prima di costruire l'immagine esegui "make setup" sul Mac: dati, modello e
# indice vengono montati come volumi, così l'immagine resta leggera.
# =============================================================================
FROM python:3.12-slim

# Niente file .pyc e log non bufferizzati (utile per vedere i log subito)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Prima solo i metadati del pacchetto: se cambia solo il codice, Docker riusa
# il layer con le dipendenze già installate (build molto più veloci).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[observability]"

COPY data/manuals ./data/manuals
COPY eval ./eval

# In container i log vanno su stderr (li raccoglie Docker); ambiente di produzione:
# senza PMC_ADMIN_PASSWORD nel .env l'API si rifiuta di partire.
ENV PMC_ENV=prod

# Utente non privilegiato: buona pratica di sicurezza per i container.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
# /health/live è pubblico e leggero: dice solo "il processo risponde".
# (/health richiede l'autenticazione e controlla anche Ollama: non adatto a un healthcheck.)
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')"
CMD ["uvicorn", "pmcopilot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
