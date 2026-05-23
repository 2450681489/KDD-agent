from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from data_agent_baseline.benchmark.schema import AnswerTable


@dataclass(frozen=True, slots=True)
class StepRecord:
    step_index: int
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str
    observation: dict[str, Any]
    ok: bool
    phase: str = "react"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentRuntimeState:
    steps: list[StepRecord] = field(default_factory=list)
    answer: AnswerTable | None = None
    failure_reason: str | None = None
    loaded_data: dict[str, Any] | None = None
    engine_schema: dict[str, Any] | None = None
    catalog: dict[str, Any] | None = None
    schema_linking: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    focused_schema: dict[str, Any] | None = None
    knowledge_chunks: list[dict[str, Any]] = field(default_factory=list)
    retrieved_knowledge: list[dict[str, Any]] = field(default_factory=list)
    sql_knowledge_constraints: str | None = None
    sql_attempts: list[dict[str, Any]] = field(default_factory=list)
    final_sql: str | None = None
    final_sql_result: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    task_id: str
    answer: AnswerTable | None
    steps: list[StepRecord]
    failure_reason: str | None
    final_sql: str | None = None
    status: str = "completed"

    @property
    def succeeded(self) -> bool:
        return self.answer is not None and self.failure_reason is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "answer": self.answer.to_dict() if self.answer is not None else None,
            "steps": [step.to_dict() for step in self.steps],
            "failure_reason": self.failure_reason,
            "final_sql": self.final_sql,
            "status": self.status,
            "succeeded": self.succeeded,
        }


def build_trace_payload(
    *,
    task_id: str,
    state: AgentRuntimeState,
    status: str,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "answer": state.answer.to_dict() if state.answer is not None else None,
        "steps": [step.to_dict() for step in state.steps],
        "failure_reason": state.failure_reason,
        "final_sql": state.final_sql,
        "status": status,
        "succeeded": state.answer is not None and state.failure_reason is None,
    }
