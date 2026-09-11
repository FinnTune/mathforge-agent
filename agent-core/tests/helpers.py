"""Reusable test doubles for LangChain / LangGraph.

``create_react_agent`` calls ``model.bind_tools(...)``. Many fake models in
langchain_core do not implement ``bind_tools``; this subclass returns ``self`` so
the graph uses our scripted ``_generate`` responses in order.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool, tool


class ScriptedToolModel(BaseChatModel):
    """Return a fixed sequence of ``AIMessage`` values (tool calls + final text)."""

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
        """No-op bind: tool schemas are ignored; responses are fully scripted."""
        return self


def _split_ai_message_into_chunks(message: AIMessage) -> list[AIMessageChunk]:
    """Splits a scripted ``AIMessage`` into several ``AIMessageChunk``s the way
    real providers stream: a tool call's name arrives in the first chunk,
    then only raw ``args`` JSON fragments in later ones (no name/id repeated).
    Each chunk's own ``.tool_calls`` is a *partial*, often-misleading
    reconstruction — only summing chunks via ``+`` gives the real thing. See
    ``grpc_server.py``'s ``Chat`` for why this distinction is load-bearing.
    """
    if message.tool_calls:
        chunks: list[AIMessageChunk] = []
        for call in message.tool_calls:
            args_json = json.dumps(call.get("args", {}))
            chunks.append(
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": call.get("name", ""),
                            "args": "",
                            "id": call.get("id"),
                            "index": 0,
                        }
                    ],
                )
            )
            mid = max(1, len(args_json) // 2)
            for fragment in (args_json[:mid], args_json[mid:]):
                if fragment:
                    chunks.append(
                        AIMessageChunk(
                            content="",
                            tool_call_chunks=[
                                {"name": None, "args": fragment, "id": None, "index": 0}
                            ],
                        )
                    )
        return chunks
    if message.content:
        text = str(message.content)
        mid = max(1, len(text) // 2)
        return [AIMessageChunk(content=text[:mid]), AIMessageChunk(content=text[mid:])]
    return [AIMessageChunk(content="")]


class ScriptedStreamingToolModel(BaseChatModel):
    """Like ``ScriptedToolModel``, but each response streams as several
    ``AIMessageChunk``s instead of one complete message — for tests that need
    to exercise real chunk-accumulation behavior (e.g. the tool-call
    reconstruction in ``grpc_server.py``'s ``Chat``), not just the single
    complete message ``ScriptedToolModel`` hands back via ``_generate``.
    """

    def __init__(self, responses: list[AIMessage]) -> None:
        super().__init__()
        self._responses = responses
        self._i = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-streaming-tool-model"

    def bind_tools(self, tools, **kwargs: Any) -> BaseChatModel:
        return self

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

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs: Any):
        response = self._responses[self._i]
        if self._i < len(self._responses) - 1:
            self._i += 1
        for chunk in _split_ai_message_into_chunks(response):
            yield ChatGenerationChunk(message=chunk)


def make_fake_execute_python_tool() -> BaseTool:
    """A stand-in for the real MCP sandbox tool, named to match it exactly.

    Lets ``ScriptedToolModel``-driven graph tests (tool-call routing,
    checkpointer persistence) run without building/spawning the Rust sandbox
    server — that server has its own test suite in
    ``mcp-servers/sandbox-rs/tests``.
    """

    @tool
    def execute_python(code: str) -> str:
        """Fake sandbox tool for tests: echoes the code it was asked to run."""
        return f"Execution result:\n(fake sandbox — received code: {code!r})"

    return execute_python


def make_fake_search_math_knowledge_tool() -> BaseTool:
    """A stand-in for the real MCP mathkb tool, named to match it exactly.

    Lets graph tests exercise a ``search_math_knowledge`` tool-call without
    Qdrant/Voyage or the mathkb server running — that server has its own test
    suite in ``mcp-servers/mathkb-py/tests``.
    """

    @tool
    def search_math_knowledge(query: str, top_k: int = 3) -> str:
        """Fake mathkb tool for tests: returns a canned note for any query."""
        return f"[source: fake_notes.md | section: Fake] (fake retrieval — query was {query!r})"

    return search_math_knowledge
