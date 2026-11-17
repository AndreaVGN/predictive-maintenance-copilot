"""
auth.py — autenticazione, ruoli e limiti d'uso dell'API.

UTENTI E RUOLI
    admin      chatta, APPROVA gli ordini di lavoro, vede Esperimenti e
               Monitoraggio, promuove modelli. Default: admin / admin.
    operatore  (opzionale, PMC_OPERATOR_USER/PASSWORD) chatta e consulta,
               ma non può approvare né amministrare.
    Le credenziali stanno nel .env. All'avvio, se la password è ancora "admin",
    il log mostra un avviso; con PMC_ENV=prod l'API si rifiuta di partire
    (a meno di PMC_ALLOW_DEFAULT_PASSWORD=true).

DUE MODI DI AUTENTICARSI
    1. Console web: POST /auth/login con utente e password -> l'API imposta un
       COOKIE DI SESSIONE firmato (HttpOnly: JavaScript non può leggerlo, quindi
       un eventuale script malevolo non può rubarlo). Il browser lo rimanda da
       solo a ogni richiesta, anche per le immagini dei grafici.
    2. Script e strumenti (curl, Prometheus): HTTP Basic Auth, cioè l'header
       "Authorization: Basic base64(utente:password)". Sempre e solo su HTTPS.

COME SI FIRMA UN COOKIE
    itsdangerous serializza {utente, ruolo} e aggiunge una firma HMAC calcolata
    con un segreto che solo il server conosce. Se qualcuno modifica il cookie
    (es. cambia "operatore" in "admin"), la firma non torna e il cookie viene
    rifiutato. Il cookie scade dopo PMC_SESSION_HOURS ore.

LIMITI D'USO
    * rate limit: massimo N richieste chat al minuto per utente (finestra mobile)
    * budget giornaliero di token per utente (contati in app.db)
    * login: dopo 5 tentativi falliti dallo stesso indirizzo, blocco di 5 minuti
    I contatori del rate limit sono in memoria: con più istanze dell'API
    servirebbe un archivio condiviso (Redis). Il budget invece è già su app.db.
"""

from __future__ import annotations

import base64
import hmac
import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from fastapi import HTTPException, Request
from fastapi.security.utils import get_authorization_scheme_param
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from pmcopilot.config import Settings

COOKIE_NAME = "pmc_session"


@dataclass(frozen=True)
class User:
    name: str
    role: str  # "admin" | "operatore"

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def _same(a: str, b: str) -> bool:
    """Confronto a tempo costante: non rivela quanti caratteri iniziali sono giusti."""
    return hmac.compare_digest(a.encode(), b.encode())


def verify_credentials(settings: Settings, username: str, password: str) -> User | None:
    if _same(username, settings.admin_user) and _same(password, settings.admin_password.get_secret_value()):
        return User(settings.admin_user, "admin")
    if (settings.operator_user and settings.operator_password
            and _same(username, settings.operator_user)
            and _same(password, settings.operator_password.get_secret_value())):
        return User(settings.operator_user, "operatore")
    return None


def session_secret(settings: Settings) -> str:
    """Segreto per firmare i cookie: dal .env o generato una volta e salvato su disco.

    Salvarlo su file fa sì che le sessioni sopravvivano ai riavvii e che più
    processi uvicorn sulla stessa macchina usino lo stesso segreto.
    """
    if settings.session_secret:
        return settings.session_secret.get_secret_value()
    path = settings.data_dir / "processed" / ".session_secret"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_urlsafe(48))
        path.chmod(0o600)
    return path.read_text().strip()


class Authenticator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.serializer = URLSafeTimedSerializer(session_secret(settings), salt="pmc-session")
        self._failed: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    # ------------------------------------------------------------- sessioni
    def make_cookie(self, user: User) -> str:
        return self.serializer.dumps({"u": user.name, "r": user.role})

    def read_cookie(self, value: str) -> User | None:
        try:
            data = self.serializer.loads(value, max_age=self.settings.session_hours * 3600)
        except (BadSignature, SignatureExpired):
            return None
        return User(data["u"], data["r"])

    def user_from_request(self, request: Request) -> User | None:
        cookie = request.cookies.get(COOKIE_NAME)
        if cookie and (user := self.read_cookie(cookie)):
            return user
        scheme, param = get_authorization_scheme_param(request.headers.get("Authorization"))
        if scheme.lower() == "basic" and param:
            try:
                username, _, password = base64.b64decode(param).decode().partition(":")
            except Exception:
                return None
            return verify_credentials(self.settings, username, password)
        return None

    # -------------------------------------------------- tentativi di login
    def check_login_allowed(self, client: str) -> None:
        with self._lock:
            attempts = self._failed[client]
            while attempts and time.monotonic() - attempts[0] > 300:
                attempts.popleft()
            if len(attempts) >= 5:
                raise HTTPException(429, "Troppi tentativi falliti: riprova tra qualche minuto.")

    def record_failure(self, client: str) -> None:
        with self._lock:
            self._failed[client].append(time.monotonic())


class RateLimiter:
    """Finestra mobile: al massimo `limit` eventi negli ultimi 60 secondi per chiave."""

    def __init__(self, limit_per_minute: int):
        self.limit = limit_per_minute
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        if self.limit <= 0:
            return
        now = time.monotonic()
        with self._lock:
            events = self._events[key]
            while events and now - events[0] > 60:
                events.popleft()
            if len(events) >= self.limit:
                retry = int(60 - (now - events[0])) + 1
                raise HTTPException(429, f"Limite di {self.limit} richieste al minuto raggiunto. "
                                         f"Riprova tra {retry} secondi.", headers={"Retry-After": str(retry)})
            events.append(now)


# ------------------------------------------------------------ dipendenze
def current_user(request: Request) -> User:
    """Dipendenza FastAPI: l'utente autenticato, oppure 401."""
    user = request.app.state.auth.user_from_request(request)
    if user is None:
        raise HTTPException(401, "Autenticazione richiesta.", headers={"WWW-Authenticate": "Basic"})
    request.state.user = user
    return user


def require_admin(request: Request) -> User:
    user = current_user(request)
    if not user.is_admin:
        raise HTTPException(403, "Operazione riservata all'amministratore.")
    return user


def check_limits(request: Request, user: User) -> None:
    """Rate limit per utente e budget giornaliero di token (429 se superati)."""
    request.app.state.rate_limiter.check(user.name)
    budget = request.app.state.settings.daily_token_budget
    if budget and request.app.state.store.tokens_used_today(user.name) >= budget:
        raise HTTPException(429, f"Budget giornaliero di {budget} token esaurito per {user.name}.")
