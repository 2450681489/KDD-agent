from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_dataset_root() -> Path:
    return PROJECT_ROOT / "data" / "public" / "input"


def _default_run_output_dir() -> Path:
    return PROJECT_ROOT / "artifacts" / "runs"


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    root_path: Path = field(default_factory=_default_dataset_root)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    model: str = "gpt-4.1-mini"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    max_steps: int = 16
    max_sql_attempts: int = 5
    sql_result_limit: int = 200
    catalog_sample_rows: int = 3
    enable_knowledge_retrieval: bool = True
    knowledge_top_k_plan: int = 4
    knowledge_top_k_sql: int = 3
    knowledge_chunk_max_chars: int = 1200
    markdown_extract_max_workers: int = 4
    model_request_timeout_seconds: float = 120.0
    model_max_retries: int = 2
    model_retry_backoff_seconds: float = 2.0
    temperature: float = 0.0


@dataclass(frozen=True, slots=True)
class RunConfig:
    output_dir: Path = field(default_factory=_default_run_output_dir)
    run_id: str | None = None
    max_workers: int = 4
    task_timeout_seconds: int = 600


@dataclass(frozen=True, slots=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    run: RunConfig = field(default_factory=RunConfig)


def _path_value(raw_value: str | None, default_value: Path) -> Path:
    if not raw_value:
        return default_value
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _bool_value(raw_value: object, default_value: bool) -> bool:
    if raw_value is None:
        return default_value
    if isinstance(raw_value, bool):
        return raw_value
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def load_app_config(config_path: Path) -> AppConfig:
    payload = yaml.safe_load(config_path.read_text()) or {}
    dataset_defaults = DatasetConfig()
    agent_defaults = AgentConfig()
    run_defaults = RunConfig()

    dataset_payload = payload.get("dataset", {})
    agent_payload = payload.get("agent", {})
    run_payload = payload.get("run", {})

    dataset_config = DatasetConfig(
        root_path=_path_value(dataset_payload.get("root_path"), dataset_defaults.root_path),
    )
    agent_config = AgentConfig(
        model=str(agent_payload.get("model", agent_defaults.model)),
        api_base=str(agent_payload.get("api_base", agent_defaults.api_base)),
        api_key=str(agent_payload.get("api_key", agent_defaults.api_key)),
        max_steps=int(agent_payload.get("max_steps", agent_defaults.max_steps)),
        max_sql_attempts=int(agent_payload.get("max_sql_attempts", agent_defaults.max_sql_attempts)),
        sql_result_limit=int(agent_payload.get("sql_result_limit", agent_defaults.sql_result_limit)),
        catalog_sample_rows=int(
            agent_payload.get("catalog_sample_rows", agent_defaults.catalog_sample_rows)
        ),
        enable_knowledge_retrieval=_bool_value(
            agent_payload.get("enable_knowledge_retrieval"),
            agent_defaults.enable_knowledge_retrieval,
        ),
        knowledge_top_k_plan=int(
            agent_payload.get("knowledge_top_k_plan", agent_defaults.knowledge_top_k_plan)
        ),
        knowledge_top_k_sql=int(
            agent_payload.get("knowledge_top_k_sql", agent_defaults.knowledge_top_k_sql)
        ),
        knowledge_chunk_max_chars=int(
            agent_payload.get("knowledge_chunk_max_chars", agent_defaults.knowledge_chunk_max_chars)
        ),
        markdown_extract_max_workers=int(
            agent_payload.get(
                "markdown_extract_max_workers",
                agent_defaults.markdown_extract_max_workers,
            )
        ),
        model_request_timeout_seconds=float(
            agent_payload.get(
                "model_request_timeout_seconds",
                agent_defaults.model_request_timeout_seconds,
            )
        ),
        model_max_retries=int(
            agent_payload.get("model_max_retries", agent_defaults.model_max_retries)
        ),
        model_retry_backoff_seconds=float(
            agent_payload.get(
                "model_retry_backoff_seconds",
                agent_defaults.model_retry_backoff_seconds,
            )
        ),
        temperature=float(agent_payload.get("temperature", agent_defaults.temperature)),
    )
    raw_run_id = run_payload.get("run_id")
    run_id = run_defaults.run_id
    if raw_run_id is not None:
        normalized_run_id = str(raw_run_id).strip()
        run_id = normalized_run_id or None

    run_config = RunConfig(
        output_dir=_path_value(run_payload.get("output_dir"), run_defaults.output_dir),
        run_id=run_id,
        max_workers=int(run_payload.get("max_workers", run_defaults.max_workers)),
        task_timeout_seconds=int(run_payload.get("task_timeout_seconds", run_defaults.task_timeout_seconds)),
    )
    return AppConfig(dataset=dataset_config, agent=agent_config, run=run_config)
