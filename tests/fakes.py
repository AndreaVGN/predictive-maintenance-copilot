"""
fakes.py — un modello di chat FINTO per testare l'agente senza Ollama.

PERCHÉ
    I test devono essere veloci, deterministici e girare ovunque (anche in CI,
    dove Ollama non c'è). Un LLM vero non è nessuna delle tre cose.
    Questo modello restituisce, in ordine, una lista di risposte preparate:
    possiamo così "recitare" una conversazione (es. prima chiama sql_query,
    poi risponde) e verificare che il GRAFO si comporti correttamente:
    guardrail, routing, esecuzione dei tool, approvazioni, citazioni.

    Cosa NON testa: la qualità delle decisioni dell'LLM vero. Per quella c'è
    la valutazione in eval/ (pmc eval agent), che usa Ollama.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field


class ScriptedChatModel(BaseChatModel):
    """Restituisce le risposte di `script` una dopo l'altra."""

    script: list[AIMessage]
    structured: Any = None  # cosa restituire da with_structured_output (router)
    calls: list[int] = Field(default_factory=lambda: [0])  # contatore mutabile delle chiamate

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        index = self.calls[0]
        self.calls[0] += 1
        if index >= len(self.script):
            message = AIMessage(content="(fine dello script)")
        else:
            message = self.script[index].model_copy()
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):  # create_agent la chiama
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any):
        return RunnableLambda(lambda _input: self.structured)


def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    """Messaggio dell'LLM che chiede di eseguire un tool."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])
