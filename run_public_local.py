from __future__ import annotations

import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

from data_agent_baseline.config import AgentConfig, AppConfig, DatasetConfig, RunConfig
from data_agent_baseline.run.runner import _run_single_task_with_timeout


INPUT_ROOT = Path(os.environ.get("INPUT_ROOT", "data/public/input"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "tmp_output"))
TRACE_ROOT = Path(os.environ.get("TRACE_ROOT", "tmp_traces"))


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    return int(raw_value)


def _float_env(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    return float(raw_value)


def build_local_config() -> AppConfig:
    agent_defaults = AgentConfig()
    run_defaults = RunConfig()
    return AppConfig(
        dataset=DatasetConfig(root_path=INPUT_ROOT),
        agent=replace(
            agent_defaults,
            model=_required_env("MODEL_NAME"),
            api_base=_required_env("MODEL_API_URL"),
            api_key=_required_env("MODEL_API_KEY"),
            temperature=_float_env("AGENT_TEMPERATURE", agent_defaults.temperature),
            max_steps=_int_env("AGENT_MAX_STEPS", agent_defaults.max_steps),
            max_sql_attempts=_int_env(
                "AGENT_MAX_SQL_ATTEMPTS",
                agent_defaults.max_sql_attempts,
            ),
            sql_result_limit=_int_env("AGENT_SQL_RESULT_LIMIT", agent_defaults.sql_result_limit),
            catalog_sample_rows=_int_env(
                "AGENT_CATALOG_SAMPLE_ROWS",
                agent_defaults.catalog_sample_rows,
            ),
            model_request_timeout_seconds=_float_env(
                "MODEL_REQUEST_TIMEOUT_SECONDS",
                agent_defaults.model_request_timeout_seconds,
            ),
            model_max_retries=_int_env("MODEL_MAX_RETRIES", agent_defaults.model_max_retries),
            model_retry_backoff_seconds=_float_env(
                "MODEL_RETRY_BACKOFF_SECONDS",
                agent_defaults.model_retry_backoff_seconds,
            ),
            markdown_extract_max_workers=max(
                1,
                min(
                    _int_env(
                        "MARKDOWN_EXTRACT_MAX_WORKERS",
                        agent_defaults.markdown_extract_max_workers,
                    ),
                    8,
                ),
            ),
        ),
        run=replace(
            run_defaults,
            output_dir=OUTPUT_ROOT,
            task_timeout_seconds=_int_env(
                "AGENT_TASK_TIMEOUT_SECONDS",
                run_defaults.task_timeout_seconds,
            ),
            max_workers=max(1, min(_int_env("AGENT_MAX_WORKERS", run_defaults.max_workers), 16)),
        ),
    )


def task_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.removeprefix("task_")
    try:
        return int(suffix), path.name
    except ValueError:
        return sys.maxsize, path.name


def iter_task_ids(input_root: Path) -> list[str]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    task_dirs = [
        path
        for path in input_root.iterdir()
        if path.is_dir() and path.name.startswith("task_") and (path / "task.json").is_file()
    ]
    return [path.name for path in sorted(task_dirs, key=task_sort_key)]


def select_task_ids(task_ids: list[str]) -> list[str]:
    raw_value = os.environ.get("TASK_IDS", "").strip()
    if not raw_value:
        return task_ids
    requested = {
        item.strip()
        for item in raw_value.split(",")
        if item.strip()
    }
    return [task_id for task_id in task_ids if task_id in requested]


def write_prediction_csv(task_id: str, answer: dict[str, Any] | None) -> Path:
    out_dir = OUTPUT_ROOT / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = out_dir / "prediction.csv"

    if isinstance(answer, dict):
        columns = [str(column) for column in answer.get("columns", [])]
        raw_rows = answer.get("rows", [])
        rows = raw_rows if isinstance(raw_rows, list) else []
    else:
        columns = ["answer"]
        rows = []
    if not columns:
        columns = ["answer"]

    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row if isinstance(row, list) else [row])
    return prediction_path


def run_task(task_id: str, config: AppConfig) -> dict[str, Any]:
    TRACE_ROOT.mkdir(parents=True, exist_ok=True)
    trace_path = TRACE_ROOT / f"{task_id}.json"
    run_result = _run_single_task_with_timeout(
        task_id=task_id,
        config=config,
        trace_path=trace_path,
    )
    trace_path.write_text(
        json.dumps(run_result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return run_result


def run_and_write_task(index: int, total: int, task_id: str, config: AppConfig) -> bool:
    task_started_at = perf_counter()
    try:
        run_result = run_task(task_id, config)
        answer = run_result.get("answer")
        prediction_path = write_prediction_csv(
            task_id,
            answer if isinstance(answer, dict) else None,
        )
        succeeded = run_result.get("status") == "completed" and isinstance(answer, dict)
        status = "ok" if succeeded else "fail"
        elapsed = perf_counter() - task_started_at
        failure_reason = run_result.get("failure_reason")
        reason_suffix = f", reason={failure_reason}" if failure_reason else ""
        print(
            f"[{index}/{total}] {task_id}: {status}, "
            f"elapsed={elapsed:.2f}s, output={prediction_path}{reason_suffix}",
            flush=True,
        )
        return succeeded
    except Exception as exc:  # noqa: BLE001
        prediction_path = write_prediction_csv(task_id, None)
        elapsed = perf_counter() - task_started_at
        print(
            f"[{index}/{total}] {task_id}: error, "
            f"elapsed={elapsed:.2f}s, output={prediction_path}, reason={exc}",
            flush=True,
        )
        return False


def main() -> int:
    started_at = perf_counter()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TRACE_ROOT.mkdir(parents=True, exist_ok=True)

    config = build_local_config()
    task_ids = select_task_ids(iter_task_ids(INPUT_ROOT))
    print(
        f"Discovered {len(task_ids)} tasks under {INPUT_ROOT}; "
        f"output={OUTPUT_ROOT}; traces={TRACE_ROOT}; "
        f"task_workers={config.run.max_workers}; "
        f"markdown_workers={config.agent.markdown_extract_max_workers}; "
        f"task_timeout_seconds={config.run.task_timeout_seconds}",
        flush=True,
    )

    success_count = 0
    failure_count = 0
    with ThreadPoolExecutor(max_workers=config.run.max_workers) as executor:
        future_to_task = {
            executor.submit(run_and_write_task, index, len(task_ids), task_id, config): task_id
            for index, task_id in enumerate(task_ids, start=1)
        }
        for future in as_completed(future_to_task):
            if future.result():
                success_count += 1
            else:
                failure_count += 1

    total_elapsed = perf_counter() - started_at
    print(
        f"Finished {len(task_ids)} tasks: ok={success_count}, "
        f"fail={failure_count}, elapsed={total_elapsed:.2f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
