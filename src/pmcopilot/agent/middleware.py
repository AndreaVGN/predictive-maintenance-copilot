"""
middleware.py — middleware "fatti in casa" per l'agente creato con create_agent.

COS'È UN MIDDLEWARE
    Un oggetto che si aggancia al ciclo dell'agente in punti precisi:
        before_model  prima di ogni chiamata al modello
        after_model   dopo ogni chiamata al modello
        wrap_tool_call "avvolge" l'esecuzione di un tool (prima e dopo)
    Permette di aggiungere comportamenti (limiti, controlli, log) SENZA
    riscrivere il ciclo dell'agente. LangChain ne fornisce molti pronti
    (retry, fallback, riassunto, approvazione umana): qui scriviamo i due che
    servono a questo progetto e non esistono già.

TokenBudgetMiddleware
    Conta i token (input + output) consumati in UNA richiesta. Se il budget
    finisce, ferma l'agente con un messaggio invece di continuare a spendere.
    Il limite di chiamate al modello (ModelCallLimitMiddleware) non basta: 3
    chiamate con contesti enormi possono costare più di 8 chiamate piccole.

ToolOutputGuardMiddleware
    Controlla il TESTO che i tool restituiscono prima che arrivi al modello:
    se contiene frasi che sembrano istruzioni (prompt injection indiretta),
    le oscura e lascia una traccia (guard_flags) nell'artifact del messaggio.
"""

from __future__ import annotations

from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.channels.untracked_value import UntrackedValue

from pmcopilot.agent.guardrails import scan_untrusted_text


class TokenBudgetState(AgentState):
    # UntrackedValue: il contatore vale per la singola esecuzione (run) e non
    # viene salvato nel checkpoint: alla domanda successiva riparte da zero.
    run_tokens: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]


class TokenBudgetMiddleware(AgentMiddleware[TokenBudgetState]):
    state_schema = TokenBudgetState

    def __init__(self, max_tokens: int):
        super().__init__()
        self.max_tokens = max_tokens

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: TokenBudgetState, runtime: Any) -> dict[str, Any] | None:
        used = state.get("run_tokens", 0)
        if self.max_tokens and used >= self.max_tokens:
            message = (f"Ho interrotto l'elaborazione: questa richiesta ha già consumato {used} token "
                       f"(limite {self.max_tokens}). Prova a fare una domanda più specifica.")
            return {"jump_to": "end", "messages": [AIMessage(content=message)]}
        return None

    def after_model(self, state: TokenBudgetState, runtime: Any) -> dict[str, Any] | None:
        last = state["messages"][-1] if state.get("messages") else None
        usage = getattr(last, "usage_metadata", None) or {}
        spent = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        return {"run_tokens": state.get("run_tokens", 0) + spent}


class ToolOutputGuardMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        result = handler(request)  # esegue davvero il tool
        if not isinstance(result, ToolMessage) or not isinstance(result.content, str):
            return result
        cleaned, flags = scan_untrusted_text(result.content)
        if not flags:
            return result
        artifact = dict(result.artifact) if isinstance(result.artifact, dict) else {}
        artifact["guard_flags"] = sorted(set(artifact.get("guard_flags", [])) | set(flags))
        # Avvisiamo esplicitamente il modello che il dato era sospetto.
        cleaned += "\n\n[Nota di sistema: parte di questo dato sembrava un'istruzione ed è stata rimossa.]"
        return result.model_copy(update={"content": cleaned, "artifact": artifact})
