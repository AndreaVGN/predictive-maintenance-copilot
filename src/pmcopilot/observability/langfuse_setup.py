"""
langfuse_setup.py — prepara Langfuse in locale senza passaggi manuali.

IL PROBLEMA
    Avviare Langfuse richiede di: creare un account, un'organizzazione, un
    progetto, generare due chiavi API e copiarle nel .env. Facile sbagliare.

LA SOLUZIONE: "HEADLESS INITIALIZATION"
    Langfuse, al primo avvio, legge alcune variabili LANGFUSE_INIT_* e crea da
    solo organizzazione, progetto, utente e chiavi. Questo script:
      1. genera segreti casuali e scrive infra/langfuse/.env, il file che
         docker compose legge per riempire le variabili ${...} del compose
         ufficiale di Langfuse;
      2. copia le stesse chiavi nel .env del progetto, così l'app e Langfuse
         "si conoscono" già.
    Lo script è idempotente: se infra/langfuse/.env esiste, riusa i segreti.

    Uso:  make langfuse-up   (chiama "pmc langfuse-init" e poi docker compose)

SICUREZZA
    I segreti stanno in file .env esclusi da Git (.gitignore). La modalità
    "localhost" va bene per sviluppo: per un server condiviso servono HTTPS,
    segreti gestiti (vault) e Langfuse non esposto su internet.
"""

from __future__ import annotations

import re
import secrets
from pathlib import Path

from pmcopilot.config import PROJECT_ROOT

LANGFUSE_URL = "http://localhost:3000"
PROJECT_ID = "pmcopilot"
USER_EMAIL = "admin@pmcopilot.local"


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
    return values


def set_env_var(path: Path, key: str, value: str) -> None:
    """Imposta KEY=value in un file .env: sostituisce la riga se c'è, altrimenti la aggiunge."""
    text = path.read_text() if path.exists() else ""
    line = f"{key}={value}"
    pattern = re.compile(rf"^#?\s*{re.escape(key)}=.*$", re.MULTILINE)
    if pattern.search(text):
        text = pattern.sub(line, text, count=1)
    else:
        text = text.rstrip("\n") + ("\n" if text else "") + line + "\n"
    path.write_text(text)


def init_local_langfuse(project_root: Path = PROJECT_ROOT) -> dict[str, str]:
    """Crea (o riusa) la configurazione e restituisce URL e credenziali di accesso."""
    infra_env = project_root / "infra" / "langfuse" / ".env"
    infra_env.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_env(infra_env)

    if not existing:
        # token_hex(32) = 64 caratteri esadecimali: è il formato richiesto da ENCRYPTION_KEY.
        values = {
            # --- segreti interni dei servizi (sostituiscono i "CHANGEME" del compose)
            "NEXTAUTH_SECRET": secrets.token_urlsafe(32),
            "SALT": secrets.token_urlsafe(24),
            "ENCRYPTION_KEY": secrets.token_hex(32),
            "POSTGRES_PASSWORD": secrets.token_urlsafe(16),
            "CLICKHOUSE_PASSWORD": secrets.token_urlsafe(16),
            "REDIS_AUTH": secrets.token_urlsafe(16),
            "MINIO_ROOT_PASSWORD": secrets.token_urlsafe(16),
            "TELEMETRY_ENABLED": "false",
            # --- inizializzazione automatica (headless)
            "LANGFUSE_INIT_ORG_ID": "pmcopilot-org",
            "LANGFUSE_INIT_ORG_NAME": "Predictive Maintenance",
            "LANGFUSE_INIT_PROJECT_ID": PROJECT_ID,
            "LANGFUSE_INIT_PROJECT_NAME": "Predictive Maintenance Copilot",
            "LANGFUSE_INIT_PROJECT_PUBLIC_KEY": f"pk-lf-{secrets.token_hex(8)}",
            "LANGFUSE_INIT_PROJECT_SECRET_KEY": f"sk-lf-{secrets.token_hex(16)}",
            "LANGFUSE_INIT_USER_EMAIL": USER_EMAIL,
            "LANGFUSE_INIT_USER_NAME": "Admin",
            "LANGFUSE_INIT_USER_PASSWORD": secrets.token_urlsafe(12),
        }
        # Il compose ufficiale usa le stesse password in più punti (es. DATABASE_URL):
        # le ricostruiamo in modo coerente.
        values["DATABASE_URL"] = f"postgresql://postgres:{values['POSTGRES_PASSWORD']}@postgres:5432/postgres"
        for key in ("LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY", "LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY",
                    "LANGFUSE_S3_BATCH_EXPORT_SECRET_ACCESS_KEY"):
            values[key] = values["MINIO_ROOT_PASSWORD"]
        header = ("# Generato da 'pmc langfuse-init'. Contiene segreti: NON metterlo su Git.\n"
                  "# Docker compose lo legge per riempire le variabili ${...} del compose di Langfuse.\n")
        infra_env.write_text(header + "\n".join(f"{k}={v}" for k, v in values.items()) + "\n")
        existing = values

    # Collega l'app: stesse chiavi nel .env del progetto (creato da .env.example se manca).
    app_env = project_root / ".env"
    if not app_env.exists() and (project_root / ".env.example").exists():
        app_env.write_text((project_root / ".env.example").read_text())
    set_env_var(app_env, "PMC_LANGFUSE_ENABLED", "true")
    set_env_var(app_env, "PMC_LANGFUSE_PROJECT_ID", existing["LANGFUSE_INIT_PROJECT_ID"])
    set_env_var(app_env, "LANGFUSE_PUBLIC_KEY", existing["LANGFUSE_INIT_PROJECT_PUBLIC_KEY"])
    set_env_var(app_env, "LANGFUSE_SECRET_KEY", existing["LANGFUSE_INIT_PROJECT_SECRET_KEY"])
    set_env_var(app_env, "LANGFUSE_HOST", LANGFUSE_URL)

    return {
        "url": LANGFUSE_URL,
        "email": existing["LANGFUSE_INIT_USER_EMAIL"],
        "password": existing["LANGFUSE_INIT_USER_PASSWORD"],
        "infra_env": str(infra_env),
    }
