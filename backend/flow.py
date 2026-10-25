"""The flow: command -> brain -> gate -> answer, wired together with LangGraph.

    START -> think -> guard -> END     when the brain suggests a tool
    START -> think -> END              when it only replies, or something fails

Rules this file enforces:
- The ONLY way a suggestion can run is the `guard` node, which calls gate.execute.
- This file never imports tools. It cannot run anything except through the gate.
- No LangGraph checkpointer is used, so nothing is saved to disk.
- LangSmith tracing would send your commands to a third party, so the flow
  refuses to start while any tracing switch is on in the environment.
"""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import replace
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from .brain import Brain, BrainError, BrainReply, Turn, clean_command
from .gate.gate import Gate
from .gate.models import Result

_TRACING_SWITCHES = ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2")
_OFF_VALUES = {"", "0", "false", "no", "off"}


class FlowError(Exception):
    """The flow cannot start safely. The message is safe to show."""


class _State(TypedDict, total=False):
    command: Any
    history: tuple
    reply: BrainReply
    result: Result


def _refuse_tracing() -> None:
    for name in _TRACING_SWITCHES:
        if os.environ.get(name, "").strip().lower() not in _OFF_VALUES:
            raise FlowError(f"{name} is turned on, which would send your commands to LangSmith. Unset it first.")


class Flow:
    def __init__(self, brain: Brain, gate: Gate, max_ai_calls: int = 10, window_seconds: float = 60.0,
                 clock: Callable[[], float] = time.monotonic):
        _refuse_tracing()
        self._brain = brain
        self._gate = gate
        self._audit = gate.audit            # share the gate's log, so there is one unbroken chain
        self._max_calls = max_ai_calls
        self._window = window_seconds
        self._clock = clock
        self._ai_calls: deque[float] = deque()
        self._graph = self._build()

    # ---- the graph ----------------------------------------------------------
    def _build(self):
        graph = StateGraph(_State)
        graph.add_node("think", self._think)
        graph.add_node("guard", self._guard)
        graph.add_edge(START, "think")
        graph.add_conditional_edges("think", self._route, {"guard": "guard", "done": END})
        graph.add_edge("guard", END)
        return graph.compile()              # no checkpointer: nothing is persisted

    def run(self, command: Any, history: tuple[Turn, ...] = ()) -> Result:
        """Take one command through the whole flow and return what to tell the user. `history`
        is the one earlier exchange, used only when you are answering J.A.R.V.I.S's question."""
        return self._graph.invoke({"command": command, "history": tuple(history)})["result"]

    # ---- nodes ----------------------------------------------------------------
    def _ai_rate_limited(self) -> bool:
        now = self._clock()
        while self._ai_calls and now - self._ai_calls[0] > self._window:
            self._ai_calls.popleft()
        if len(self._ai_calls) >= self._max_calls:
            return True
        self._ai_calls.append(now)
        return False

    def _think(self, state: _State) -> dict[str, Any]:
        try:
            if self._ai_rate_limited():
                raise BrainError("Too many requests. Please wait a moment.")
            command = clean_command(state.get("command"))
            self._audit.record("command", text=command)
            history = state.get("history") or ()
            if history:
                reply = self._brain.propose(command, self._gate.tool_specs(), history=history)
            else:
                reply = self._brain.propose(command, self._gate.tool_specs())
            if not isinstance(reply, BrainReply):
                raise BrainError("The assistant gave an unusable answer.")
        except BrainError as exc:
            self._audit.record("brain_error", reason=str(exc))
            return {"result": Result(False, str(exc))}
        except Exception as exc:            # a bug or a surprise: show nothing internal
            self._audit.record("brain_error", error_type=type(exc).__name__)
            return {"result": Result(False, "Something went wrong.")}

        if reply.proposal is None:
            self._audit.record("said", text=reply.text)
            return {"result": Result(True, reply.text, question=reply.text.rstrip().endswith("?"), ai_reply=True)}
        return {"reply": reply}

    @staticmethod
    def _route(state: _State) -> str:
        reply = state.get("reply")
        return "guard" if reply is not None and reply.proposal is not None else "done"

    def _guard(self, state: _State) -> dict[str, Any]:
        proposal = state["reply"].proposal
        result = self._gate.execute(proposal.tool, proposal.args)
        # Remember what the AI proposed (its own words), never what the tool returned.
        action = {"tool": proposal.tool, "args": dict(proposal.args) if isinstance(proposal.args, dict) else {}}
        return {"result": replace(result, action=action)}