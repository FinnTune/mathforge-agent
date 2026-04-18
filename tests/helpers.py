"""Test doubles for LangChain / LangGraph."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedToolModel(BaseChatModel):
    """Minimal chat model with tool-calling support for agent tests."""

    def __init__(self, responses: list[AIMessage]) -> None:
        super().__init__()
        self._responses = responses
        self._i = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-model"

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        response = self._responses[self._i]
        if self._i < len(self._responses) - 1:
            self._i += 1
        return ChatResult(generations=[ChatGeneration(message=response)])

    def bind_tools(self, tools, **kwargs: Any) -> BaseChatModel:
        return self
