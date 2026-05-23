from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.agents.runtime import AgentRuntimeState
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AgentConfig, AppConfig, DatasetConfig, RunConfig
from data_agent_baseline.run.runner import build_model_adapter, create_run_output_dir, run_single_task
from data_agent_baseline.tools.dataengine import DataEngine


def build_config() -> AppConfig:
    return AppConfig(
        dataset=DatasetConfig(root_path=Path("data/public/input")),
        agent=replace(
            AgentConfig(),
            model=os.environ.get("MODEL_NAME", "deepseek-v4-flash"),
            api_base=os.environ.get("MODEL_API_URL", "https://api.deepseek.com"),
            api_key=os.environ.get("MODEL_API_KEY", ""),
            temperature=float(os.environ.get("AGENT_TEMPERATURE", "0")),
            max_steps=int(os.environ.get("AGENT_MAX_STEPS", "16")),
            max_sql_attempts=int(os.environ.get("AGENT_MAX_SQL_ATTEMPTS", "5")),
            markdown_extract_max_workers=int(os.environ.get("MARKDOWN_EXTRACT_MAX_WORKERS", "4")),
        ),
        run=replace(
            RunConfig(),
            output_dir=Path("artifacts/runs"),
            task_timeout_seconds=int(os.environ.get("AGENT_TASK_TIMEOUT_SECONDS", "600")),
            max_workers=1,
        ),
    )


def export_generated_tables(config: AppConfig, run_dir: Path) -> Path:
    task = DABenchPublicDataset(config.dataset.root_path).get_task("task_418")
    debug_dir = run_dir / "task_418" / "debug_tables"
    debug_dir.mkdir(parents=True, exist_ok=True)

    db_path = debug_dir / "task_418.duckdb"
    if db_path.exists():
        db_path.unlink()

    engine = DataEngine(db_path=str(db_path))
    state = AgentRuntimeState()
    state.loaded_data = engine.register_context_dir(task.context_dir)

    agent = ReActAgent(
        model=build_model_adapter(config),
        config=ReActAgentConfig(
            max_steps=config.agent.max_steps,
            max_sql_attempts=config.agent.max_sql_attempts,
            sql_result_limit=config.agent.sql_result_limit,
            catalog_sample_rows=config.agent.catalog_sample_rows,
            enable_knowledge_retrieval=config.agent.enable_knowledge_retrieval,
            knowledge_top_k_plan=config.agent.knowledge_top_k_plan,
            knowledge_top_k_sql=config.agent.knowledge_top_k_sql,
            knowledge_chunk_max_chars=config.agent.knowledge_chunk_max_chars,
            markdown_extract_max_workers=config.agent.markdown_extract_max_workers,
        ),
    )
    agent._augment_markdown_tables_with_llm(task, state, engine)

    for table in engine.show_tables()["tables"]:
        csv_path = debug_dir / f"{table}.csv"
        engine.conn.execute(
            f"COPY (SELECT * FROM {table}) TO '{csv_path.as_posix()}' "
            "(HEADER, DELIMITER ',')"
        )

    return debug_dir


def main() -> int:
    if not os.environ.get("MODEL_API_KEY"):
        raise RuntimeError("Please set MODEL_API_KEY before running this script.")

    config = build_config()
    _, run_dir = create_run_output_dir(config.run.output_dir)
    artifact = run_single_task(
        task_id="task_418",
        config=config,
        run_output_dir=run_dir,
    )

    print("run_dir:", run_dir)
    print("prediction:", artifact.prediction_csv_path)
    print("trace:", artifact.trace_path)
    print("succeeded:", artifact.succeeded)
    print("failure:", artifact.failure_reason)

    debug_dir = export_generated_tables(config, run_dir)
    print("debug_tables:", debug_dir)
    for path in sorted(debug_dir.glob("*")):
        print("  ", path)

    return 0 if artifact.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
