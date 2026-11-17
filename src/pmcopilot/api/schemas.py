"""
schemas.py — i "contratti" dell'API, scritti con Pydantic.

PERCHÉ PYDANTIC
    FastAPI usa questi modelli per tre cose, automaticamente:
      1. VALIDARE l'input: se rpm è una stringa o la temperatura è 5000 K,
         la richiesta viene rifiutata con un errore 422 che spiega cosa non va
         (anche questo è un guardrail: input assurdi non arrivano al modello);
      2. SERIALIZZARE l'output in JSON;
      3. DOCUMENTARE l'API: apri http://localhost:8000/docs e trovi tutto,
         con esempi provabili dal browser (Swagger UI).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- /predict
class PredictRequest(BaseModel):
    machine_type: Literal["L", "M", "H"] = Field(description="Tipo/qualità della macchina")
    air_temp_k: float = Field(ge=250, le=350, description="Temperatura aria [K]")
    process_temp_k: float = Field(ge=250, le=400, description="Temperatura di processo [K]")
    rpm: float = Field(ge=0, le=5000, description="Velocità di rotazione [rpm]")
    torque_nm: float = Field(ge=0, le=150, description="Coppia [Nm]")
    tool_wear_min: float = Field(ge=0, le=400, description="Usura utensile [min]")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"machine_type": "L", "air_temp_k": 302.0, "process_temp_k": 310.5,
                 "rpm": 1300, "torque_nm": 60.0, "tool_wear_min": 210}
            ]
        }
    }


class PredictResponse(BaseModel):
    failure_proba: float
    predicted_failure: bool
    risk_level: Literal["basso", "medio", "alto"]
    threshold: float
    model_version: str
    risk_factors: list[str]
    derived: dict[str, float]


# ------------------------------------------------------------------- /chat
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$",
                                  description="Ometti per iniziare una nuova conversazione")


class PendingActionOut(BaseModel):
    tool: str
    args: dict
    description: str = ""


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    route: str
    tools_used: list[str]
    sources: list[str]
    charts: list[str] = Field(description="URL relativi dei grafici generati")
    guard_flags: list[str]
    pending_approval: list[PendingActionOut]
    trace: dict
    request_id: str = ""
    langfuse_url: str | None = None
    cost_usd: float = 0.0


class DecisionRequest(BaseModel):
    approve: bool
    message: str | None = Field(default=None, max_length=500,
                                description="Motivazione in caso di rifiuto")


# ----------------------------------------------------------------- /health
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    checks: dict[str, bool]
    model_version: str | None = None
    llm_model: str


# ------------------------------------------------------------------ auth/feedback
class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=200)


class FeedbackRequest(BaseModel):
    request_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    thread_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    rating: Literal[-1, 1]
    comment: str | None = Field(default=None, max_length=1000)


# ------------------------------------------------------------- admin / esperimenti
class EvalRunRequest(BaseModel):
    kind: Literal["guardrails", "retrieval", "agent"]
    label: str | None = Field(default=None, max_length=120)
    mode: Literal["dense", "bm25", "hybrid", "all"] | None = None  # solo retrieval
    with_llm: bool = False  # solo guardrails: valuta anche il classificatore LLM
    limit: int | None = Field(default=None, ge=1, le=500)  # solo agent
    categories: list[str] | None = None  # solo agent


class SimulateRequest(BaseModel):
    hours: float = Field(default=24, gt=0, le=24 * 30)
    scenario: Literal["normal", "seasonal", "sensor_bias", "late_replacement", "concept"] = "normal"
    intensity: float = Field(default=1.0, ge=0, le=3)
    ramp: bool = False


class TrainRequest(BaseModel):
    quick: bool = True
    split: Literal["temporal", "random"] = "temporal"
    promote_policy: Literal["if_better", "always", "never"] = "if_better"
