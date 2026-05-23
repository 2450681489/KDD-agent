from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.knowledge import (
    KnowledgeChunk,
    build_sql_knowledge_constraints,
    knowledge_chunk_summary,
    knowledge_corpus_summary,
    parse_knowledge_markdown,
    render_retrieved_knowledge,
    retrieve_knowledge_chunks,
)
from data_agent_baseline.agents.prompt import (
    BASE_SYSTEM_PROMPT,
    build_answer_prompt,
    build_catalog_prompt,
    build_nl2sql_prompt,
    build_plan_prompt,
    build_system_prompt,
)
from data_agent_baseline.agents.runtime import (
    AgentRunResult,
    AgentRuntimeState,
    StepRecord,
    build_trace_payload,
)
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.dataengine import DataEngine
from data_agent_baseline.tools.registry import ToolRegistry, create_dataengine_tool_registry

SQL_HISTORY_WINDOW = 2
SQL_PREVIEW_ROWS = 2
SQL_ERROR_CHARS = 240
MARKDOWN_LLM_MAX_SELECTED_CHUNKS = 120
MARKDOWN_LLM_BATCH_CHUNKS = 24
MARKDOWN_LLM_MAX_CHUNK_CHARS = 1100
MARKDOWN_LLM_MAX_CONTEXT_CHARS = 22000
MARKDOWN_LLM_MAX_SELECTED_CHARS = 120000


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16
    max_sql_attempts: int = 5
    sql_result_limit: int = 200
    catalog_sample_rows: int = 3
    enable_knowledge_retrieval: bool = True
    knowledge_top_k_plan: int = 4
    knowledge_top_k_sql: int = 3
    knowledge_chunk_max_chars: int = 1200
    markdown_extract_max_workers: int = 4


def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    return text


def _load_single_json_object(text: str) -> dict[str, Any]:
    payload, end = json.JSONDecoder().raw_decode(text)
    remainder = text[end:].strip()
    if remainder:
        cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
        if cleaned_remainder:
            raise ValueError("Model response must contain only one JSON object.")
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    return payload


def parse_model_payload(raw_response: str) -> dict[str, Any]:
    normalized = _strip_json_fence(raw_response)
    return _load_single_json_object(normalized)


def parse_model_step(raw_response: str) -> ModelStep:
    payload = parse_model_payload(raw_response)

    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")

    return ModelStep(
        thought=thought,
        action=action,
        action_input=action_input,
        raw_response=raw_response,
    )


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def _shorten_text(value: Any, *, max_chars: int) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _extract_result_data(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    payload = result.get("data")
    if isinstance(payload, dict):
        return payload
    return None


def _extract_table_names(value: Any) -> set[str]:
    table_names: set[str] = set()
    if isinstance(value, str):
        for table_name, _ in re.findall(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b", value):
            table_names.add(table_name)
        return table_names
    if isinstance(value, dict):
        for item in value.values():
            table_names.update(_extract_table_names(item))
        return table_names
    if isinstance(value, list):
        for item in value:
            table_names.update(_extract_table_names(item))
    return table_names


def _compact_plan(plan: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "target",
        "source_tables",
        "table_roles",
        "join_steps",
        "filters",
        "select_expressions",
        "aggregations",
        "group_by",
        "order_by",
        "limit",
        "distinct",
        "final_columns",
        "validation_checks",
    ]
    return {key: plan[key] for key in keys if key in plan}


def _compact_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    compact_tables: dict[str, Any] = {}
    tables = catalog.get("tables")
    if isinstance(tables, dict):
        for table_name, table_payload in tables.items():
            if not isinstance(table_name, str) or not isinstance(table_payload, dict):
                continue
            compact_columns: dict[str, Any] = {}
            columns = table_payload.get("columns")
            if isinstance(columns, dict):
                for column_name, column_payload in columns.items():
                    if not isinstance(column_name, str) or not isinstance(column_payload, dict):
                        continue
                    compact_column = {
                        "semantic": str(column_payload.get("semantic", "")),
                        "type_hint": str(column_payload.get("type_hint", "unknown")),
                        "notes": str(column_payload.get("notes", "")),
                    }
                    compact_columns[column_name] = compact_column
            compact_tables[table_name] = {
                "description": str(table_payload.get("description", "")),
                "columns": compact_columns,
            }

    compact_join_hints: list[dict[str, str]] = []
    join_hints = catalog.get("join_hints")
    if isinstance(join_hints, list):
        for hint in join_hints:
            if not isinstance(hint, dict):
                continue
            compact_join_hints.append(
                {
                    "left_table": str(hint.get("left_table", "")),
                    "left_column": str(hint.get("left_column", "")),
                    "right_table": str(hint.get("right_table", "")),
                    "right_column": str(hint.get("right_column", "")),
                    "confidence": str(hint.get("confidence", "")),
                }
            )

    return {
        "tables": compact_tables,
        "join_hints": compact_join_hints,
        "task_relevant_fields": _normalize_string_list(catalog.get("task_relevant_fields")),
        "warnings": _normalize_string_list(catalog.get("warnings")),
    }


def _get_case_insensitive_mapping(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    mapping: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(key, str):
            mapping[key.lower()] = value
    return mapping


def _align_catalog_with_schema(
    catalog: dict[str, Any],
    engine_schema: dict[str, Any],
) -> dict[str, Any]:
    schema_tables = engine_schema.get("tables")
    if not isinstance(schema_tables, dict):
        return catalog

    raw_tables = _get_case_insensitive_mapping(catalog.get("tables"))
    aligned_tables: dict[str, Any] = {}
    valid_columns_by_table: dict[str, set[str]] = {}

    for table_name, schema_payload in schema_tables.items():
        if not isinstance(table_name, str) or not isinstance(schema_payload, dict):
            continue

        raw_table_payload = raw_tables.get(table_name.lower(), {})
        if not isinstance(raw_table_payload, dict):
            raw_table_payload = {}
        raw_columns = _get_case_insensitive_mapping(raw_table_payload.get("columns"))

        aligned_columns: dict[str, Any] = {}
        schema_columns = schema_payload.get("columns")
        if isinstance(schema_columns, list):
            for column_name in schema_columns:
                if not isinstance(column_name, str):
                    continue
                raw_column_payload = raw_columns.get(column_name.lower(), {})
                if not isinstance(raw_column_payload, dict):
                    raw_column_payload = {}
                aligned_columns[column_name] = {
                    "semantic": str(raw_column_payload.get("semantic", "")),
                    "type_hint": str(raw_column_payload.get("type_hint", "unknown")),
                    "notes": str(raw_column_payload.get("notes", "")),
                }

        aligned_tables[table_name] = {
            "description": str(raw_table_payload.get("description", "")),
            "columns": aligned_columns,
        }
        valid_columns_by_table[table_name] = set(aligned_columns)

    aligned_join_hints: list[dict[str, str]] = []
    raw_join_hints = catalog.get("join_hints")
    if isinstance(raw_join_hints, list):
        for hint in raw_join_hints:
            if not isinstance(hint, dict):
                continue
            left_table = str(hint.get("left_table", ""))
            left_column = str(hint.get("left_column", ""))
            right_table = str(hint.get("right_table", ""))
            right_column = str(hint.get("right_column", ""))
            if (
                left_table in valid_columns_by_table
                and right_table in valid_columns_by_table
                and left_column in valid_columns_by_table[left_table]
                and right_column in valid_columns_by_table[right_table]
            ):
                aligned_join_hints.append(
                    {
                        "left_table": left_table,
                        "left_column": left_column,
                        "right_table": right_table,
                        "right_column": right_column,
                        "confidence": str(hint.get("confidence", "")),
                    }
                )

    aligned_task_relevant_fields: list[str] = []
    for field_ref in _normalize_string_list(catalog.get("task_relevant_fields")):
        if "." not in field_ref:
            continue
        table_name, column_name = field_ref.split(".", 1)
        if column_name in valid_columns_by_table.get(table_name, set()):
            aligned_task_relevant_fields.append(field_ref)

    return {
        "tables": aligned_tables,
        "join_hints": aligned_join_hints,
        "task_relevant_fields": aligned_task_relevant_fields,
        "warnings": _normalize_string_list(catalog.get("warnings")),
    }


def _build_focused_schema(
    engine_schema: dict[str, Any],
    plan: dict[str, Any],
    catalog: dict[str, Any],
) -> dict[str, Any] | None:
    tables_payload = engine_schema.get("tables")
    if not isinstance(tables_payload, dict) or len(tables_payload) <= 1:
        return None

    relevant_tables = set(_normalize_string_list(plan.get("source_tables")))
    relevant_tables.update(_extract_table_names(plan))
    relevant_tables.update(_extract_table_names(catalog.get("task_relevant_fields", [])))
    if not relevant_tables:
        return None

    focused_tables = {
        name: value for name, value in tables_payload.items() if name in relevant_tables
    }
    if not focused_tables or len(focused_tables) >= len(tables_payload):
        return None

    return {
        "table_count": len(focused_tables),
        "tables": focused_tables,
    }


def _classify_sql_issue(
    *,
    ok: bool,
    is_final: bool,
    result: dict[str, Any] | None,
    error_message: str | None,
) -> str:
    if ok:
        row_count = result.get("row_count") if isinstance(result, dict) else None
        if row_count == 0:
            return "empty_result"
        return "final_result_ready" if is_final else "partial_success"

    message = (error_message or "").lower()
    if "column" in message:
        return "unknown_column"
    if "table" in message or "catalog" in message:
        return "unknown_table"
    if "group by" in message or "aggregate" in message:
        return "group_or_aggregation_error"
    if "parser" in message or "syntax" in message:
        return "syntax_error"
    if "type" in message or "cast" in message:
        return "type_mismatch"
    return "execution_error"


def _summarize_sql_attempt(
    *,
    sql: str | None,
    ok: bool,
    is_final: bool,
    result: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    data = _extract_result_data(result)
    columns = _normalize_string_list(data.get("columns")) if data else []
    rows = data.get("rows") if data else []
    preview_rows = rows[:SQL_PREVIEW_ROWS] if isinstance(rows, list) else []
    summary = {
        "sql": sql,
        "ok": ok,
        "is_final": is_final,
        "issue_hint": _classify_sql_issue(
            ok=ok,
            is_final=is_final,
            result=result,
            error_message=error_message,
        ),
    }
    if columns:
        summary["returned_columns"] = columns
    if preview_rows:
        summary["sample_rows_preview"] = preview_rows
    if isinstance(result, dict):
        row_count = result.get("row_count")
        if isinstance(row_count, int):
            summary["row_count"] = row_count
        truncated = result.get("truncated")
        if isinstance(truncated, bool):
            summary["truncated"] = truncated
    short_error = _shorten_text(error_message, max_chars=SQL_ERROR_CHARS)
    if short_error is not None:
        summary["error_message_short"] = short_error
    return summary


def _build_answer_from_result(
    result: dict[str, Any],
    plan: dict[str, Any],
) -> AnswerTable | None:
    data = _extract_result_data(result)
    if data is None:
        return None

    raw_columns = data.get("columns")
    raw_rows = data.get("rows")
    if not isinstance(raw_columns, list) or not isinstance(raw_rows, list):
        return None

    rows: list[list[Any]] = []
    for row in raw_rows:
        if not isinstance(row, list):
            return None
        rows.append(list(row))

    result_columns = [str(column) for column in raw_columns]
    planned_columns = _normalize_string_list(plan.get("final_columns"))
    columns = planned_columns if len(planned_columns) == len(result_columns) else result_columns
    return AnswerTable(columns=columns, rows=rows)


def _collect_schema_terms(engine_schema: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    tables = engine_schema.get("tables")
    if not isinstance(tables, dict):
        return terms

    for table_name, table_payload in tables.items():
        if isinstance(table_name, str):
            terms.append(table_name)
        if not isinstance(table_payload, dict):
            continue
        columns = table_payload.get("columns")
        if isinstance(columns, list):
            terms.extend(str(column) for column in columns if isinstance(column, str))
    return terms


def _collect_catalog_terms(catalog: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    terms.extend(_normalize_string_list(catalog.get("task_relevant_fields")))

    tables = catalog.get("tables")
    if not isinstance(tables, dict):
        return terms

    for table_name, table_payload in tables.items():
        if isinstance(table_name, str):
            terms.append(table_name)
        if not isinstance(table_payload, dict):
            continue
        description = table_payload.get("description")
        if isinstance(description, str):
            terms.append(description)
        columns = table_payload.get("columns")
        if not isinstance(columns, dict):
            continue
        for column_name, column_payload in columns.items():
            if isinstance(column_name, str):
                terms.append(column_name)
            if not isinstance(column_payload, dict):
                continue
            for key in ("semantic", "notes", "type_hint"):
                value = column_payload.get(key)
                if isinstance(value, str):
                    terms.append(value)
    return terms


def _tokenize_query_text(text: str) -> set[str]:
    stop_words = {
        "among",
        "whose",
        "them",
        "they",
        "their",
        "there",
        "that",
        "this",
        "what",
        "which",
        "with",
        "without",
        "have",
        "has",
        "how",
        "many",
        "much",
        "patient",
        "patients",
        "level",
        "levels",
        "number",
        "count",
        "aren",
        "isn",
        "yet",
    }
    tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_]+", text)
        if len(token) >= 3 and token.lower() not in stop_words
    }
    expansions = {
        "creatinine": {"cre", "renal", "kidney", "glomerular", "filtration"},
        "cre": {"creatinine", "renal", "kidney"},
        "abnormal": {"elevated", "high", "impaired", "borderline", "normal"},
        "age": {"birthday", "birthdate", "born"},
        "birthday": {"age", "birthdate", "born"},
        "diagnosis": {"diagnosed", "disease"},
        "admission": {"inpatient", "outpatient", "admitted"},
    }
    for token in list(tokens):
        tokens.update(expansions.get(token, set()))
    if re.search(r"\b\d{1,3}\b", text) and re.search(r"\b(?:age|old|younger|older|yet|under|over)\b", text, flags=re.IGNORECASE):
        tokens.update({"age", "birthday", "birthdate", "born"})
    return tokens


def _extract_numeric_ids_from_text(text: str) -> set[str]:
    return set(re.findall(r"\b\d{4,9}\b", text))


def _extract_patient_ids_from_text(text: str) -> set[str]:
    patterns = [
        r"\bpatient(?:\s+(?:assigned|registered|associated|with|number|file|ID))*\s*(?:number|ID)?\s*(\d{4,9})\b",
        r"\bMedical Record Number\s+(\d{4,9})\b",
        r"\bfile number\s+(\d{4,9})\b",
        r"\bfile\s+(\d{4,9})\b",
    ]
    ids: set[str] = set()
    for pattern in patterns:
        ids.update(match.group(1) for match in re.finditer(pattern, text, flags=re.IGNORECASE))
    return ids


class ReActAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry | None = None,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
        trace_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or BASE_SYSTEM_PROMPT
        self.trace_callback = trace_callback

    def _build_messages(self, user_content: str) -> list[ModelMessage]:
        system_content = build_system_prompt(
            "",
            system_prompt=self.system_prompt,
        )
        return [
            ModelMessage(role="system", content=system_content),
            ModelMessage(role="user", content=user_content),
        ]

    def _complete_json(self, user_content: str) -> tuple[str, dict[str, Any]]:
        raw_response = self.model.complete(self._build_messages(user_content))
        return raw_response, parse_model_payload(raw_response)

    def _task_record_payload(self, task: PublicTask) -> dict[str, Any]:
        task_json_path = task.task_dir / "task.json"
        source_path = task_json_path
        if source_path.exists():
            payload = json.loads(source_path.read_text())
            if isinstance(payload, dict):
                return payload
        return {
            "task_id": task.task_id,
            "difficulty": task.difficulty,
            "question": task.question,
        }

    def _read_markdown_sections(self, task: PublicTask, paths: list[Path]) -> tuple[list[str], str]:
        relative_paths = []
        sections = []
        for path in paths:
            if not path.exists() or not path.is_file():
                continue
            content = path.read_text(errors="replace").strip()
            if not content:
                continue
            relative_path = path.relative_to(task.context_dir).as_posix()
            relative_paths.append(relative_path)
            sections.append(f"[{relative_path}]\n{content}")
        return relative_paths, "\n\n".join(sections)

    def _catalog_knowledge_text(self, task: PublicTask) -> str:
        _, text = self._read_markdown_sections(task, [task.context_dir / "knowledge.md"])
        return text

    def _plan_doc_bundle(self, task: PublicTask) -> tuple[list[str], str]:
        doc_dir = task.context_dir / "doc"
        if not doc_dir.exists():
            return [], ""
        markdown_paths = sorted(path for path in doc_dir.rglob("*.md") if path.is_file())
        return self._read_markdown_sections(task, markdown_paths)

    def _markdown_paragraph_chunks(self, task: PublicTask) -> list[dict[str, str]]:
        doc_dir = task.context_dir / "doc"
        if not doc_dir.exists():
            return []
        chunks: list[dict[str, str]] = []
        for path in sorted(doc_dir.rglob("*.md")):
            if not path.is_file():
                continue
            table_name = re.sub(r"\W+", "_", path.stem).strip("_").lower() or "document"
            current_section = ""
            current_lines: list[str] = []
            chunk_index = 1

            def flush() -> None:
                nonlocal current_lines, chunk_index
                text = " ".join(line.strip() for line in current_lines if line.strip()).strip()
                current_lines = []
                if not text or text.startswith("#") or set(text) <= {"-"}:
                    return
                chunks.append(
                    {
                        "chunk_id": f"{table_name}:{chunk_index}",
                        "table": table_name,
                        "section": current_section,
                        "text": text[:MARKDOWN_LLM_MAX_CHUNK_CHARS],
                    }
                )
                chunk_index += 1

            for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw_line.strip()
                heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
                if heading_match is not None:
                    flush()
                    current_section = heading_match.group(2).strip()
                    continue
                if not line:
                    flush()
                    continue
                current_lines.append(line)
            flush()
        return chunks

    def _markdown_doc_table_names(self, task: PublicTask) -> set[str]:
        doc_dir = task.context_dir / "doc"
        if not doc_dir.exists():
            return set()
        return {
            re.sub(r"\W+", "_", path.stem).strip("_").lower() or "document"
            for path in doc_dir.rglob("*.md")
            if path.is_file()
        }

    def _select_markdown_chunks_for_question(
        self,
        task: PublicTask,
        allowed_doc_tables: set[str] | None = None,
    ) -> list[dict[str, str]]:
        chunks = self._markdown_paragraph_chunks(task)
        if allowed_doc_tables is not None:
            chunks = [chunk for chunk in chunks if chunk["table"] in allowed_doc_tables]
        if not chunks:
            return []

        query_terms = _tokenize_query_text(task.question)
        scored_chunks: list[tuple[float, dict[str, str]]] = []
        for chunk in chunks:
            searchable = f"{chunk['table']} {chunk['section']} {chunk['text']}".lower()
            score = 0.0
            for term in query_terms:
                if term in searchable:
                    score += 2.0 if term in chunk["text"].lower() else 1.0
            if re.search(r"\bpatient\s+\d{4,9}\b", searchable):
                score += 0.25
            if chunk["table"] in {"patient", "laboratory"}:
                score += 0.3
            if score > 0:
                scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        selected: list[dict[str, str]] = []
        used_ids: set[str] = set()
        total_chars = 0

        chunks_by_table: dict[str, list[tuple[float, dict[str, str]]]] = {}
        for item in scored_chunks:
            chunks_by_table.setdefault(item[1]["table"], []).append(item)
        table_quota = max(8, MARKDOWN_LLM_MAX_SELECTED_CHUNKS // max(len(chunks_by_table), 1))
        ordered_candidates: list[tuple[float, dict[str, str]]] = []
        for table_name in sorted(chunks_by_table):
            ordered_candidates.extend(chunks_by_table[table_name][:table_quota])
        ordered_candidates.extend(scored_chunks)

        for _, chunk in ordered_candidates:
            if chunk["chunk_id"] in used_ids:
                continue
            next_chars = len(chunk["text"])
            if total_chars + next_chars > MARKDOWN_LLM_MAX_SELECTED_CHARS:
                continue
            selected.append(chunk)
            used_ids.add(chunk["chunk_id"])
            total_chars += next_chars
            reserve_for_id_linked = 4 if chunk["table"] != "patient" else 0
            if len(selected) >= MARKDOWN_LLM_MAX_SELECTED_CHUNKS - reserve_for_id_linked:
                break

        selected_ids: set[str] = set()
        for chunk in selected:
            if chunk["table"] != "patient":
                selected_ids.update(_extract_patient_ids_from_text(chunk["text"]))
        if selected_ids:
            patient_candidates = [
                chunk
                for chunk in chunks
                if chunk["table"] == "patient"
                and chunk["chunk_id"] not in used_ids
                and (_extract_patient_ids_from_text(chunk["text"]) & selected_ids)
            ]
        else:
            patient_candidates = []

        if selected_ids and patient_candidates:
            for candidate in patient_candidates:
                if candidate["chunk_id"] in used_ids:
                    continue
                while len(selected) >= MARKDOWN_LLM_MAX_SELECTED_CHUNKS:
                    removable_index = next(
                        (
                            index
                            for index in range(len(selected) - 1, -1, -1)
                            if selected[index]["table"] == "patient"
                            and not (_extract_patient_ids_from_text(selected[index]["text"]) & selected_ids)
                        ),
                        None,
                    )
                    if removable_index is None:
                        break
                    removed = selected.pop(removable_index)
                    used_ids.discard(removed["chunk_id"])
                    total_chars -= len(removed["text"])
                if len(selected) >= MARKDOWN_LLM_MAX_SELECTED_CHUNKS:
                    break
                next_chars = len(candidate["text"])
                if total_chars + next_chars > MARKDOWN_LLM_MAX_SELECTED_CHARS:
                    continue
                selected.append(candidate)
                used_ids.add(candidate["chunk_id"])
                total_chars += next_chars
        return selected

    def _build_markdown_extraction_prompt(
        self,
        task: PublicTask,
        chunks: list[dict[str, str]],
    ) -> str:
        chunk_payload = [
            {
                "chunk_id": chunk["chunk_id"],
                "table": chunk["table"],
                "section": chunk["section"],
                "text": chunk["text"],
            }
            for chunk in chunks
        ]
        return (
            "Stage: query-focused markdown-to-table extraction.\n\n"
            "You are given selected paragraphs from context/doc/*.md. Each file name is a "
            "candidate table name, and paragraphs often describe one row or a partial row. "
            "Use the question, schema knowledge, file names, section names, and paragraph text "
            "to decide which tables and columns are needed. Extract only structured rows needed "
            "to answer the question; do not reconstruct the whole database.\n\n"
            "Rules:\n"
            "- Return exactly one JSON object.\n"
            "- Use table names exactly from chunk.table.\n"
            "- Choose output tables based on relevant doc file names and knowledge.md schema. "
            "For example, chunks from Patient.md should produce table patient only if patient "
            "fields are needed; chunks from Laboratory.md should produce table laboratory only "
            "if lab fields are needed.\n"
            "- Extract explicit values only; use null when unavailable.\n"
            "- Prefer corrected, confirmed, verified, final, revised, or adjusted values over "
            "initial/preliminary values.\n"
            "- Include key fields such as ID, Date, Birthday, foreign keys, and any "
            "question-relevant fields. Preserve join keys explicitly mentioned in the text, "
            "such as cards_id, member_id, event_id, or link/reference identifiers.\n"
            "- If a doc file represents a missing table needed to join with already-loaded "
            "structured tables, prioritize the columns needed for that join and the task "
            "filter. For legality/ruling documents, extract ID, cards_id, format, and status "
            "when supported by the question, schema knowledge, file name, or paragraph text.\n"
            "- For any field-specific status column, mark abnormal/normal/borderline only when "
            "the same measurement is explicitly described that way, or when the text gives an "
            "explicit reference range/threshold for that measurement. Do not transfer a broader "
            "panel, organ-system, or neighboring-field abnormality onto a specific field.\n"
            "- If a paragraph says a general profile is abnormal, but the target measurement "
            "itself is only mentioned as a value without an explicit abnormal/high/elevated/low "
            "qualifier, extract the value and leave that measurement status null.\n"
            "- For creatinine, use column CRE and CRE_status only when creatinine/CRE itself is "
            "explicitly high/elevated/abnormal, normal, or borderline; do not infer CRE_status "
            "from urea nitrogen, uric acid, renal profile, or glomerular filtration wording alone.\n"
            "- Preserve row granularity. For lab rows, prefer ID + Date. For patient rows, "
            "prefer one row per ID.\n"
            "- Add source_chunk_id to every extracted row.\n\n"
            "Return JSON shape:\n"
            "{\n"
            "  \"thought\": \"brief extraction rationale\",\n"
            "  \"tables\": {\n"
            "    \"table_name\": [\n"
            "      {\"ID\": 123, \"Date\": \"YYYY-MM-DD\", \"field\": \"value\", "
            "\"source_chunk_id\": \"table:1\"}\n"
            "    ]\n"
            "  }\n"
            "}\n\n"
            f"Question:\n{task.question}\n\n"
            f"Schema knowledge from context/knowledge.md:\n{self._catalog_knowledge_text(task)[:5000] or '(missing)'}\n\n"
            f"Selected chunks:\n{json.dumps(chunk_payload, ensure_ascii=False, indent=2)}"
        )

    def _markdown_extraction_batches(
        self,
        chunks: list[dict[str, str]],
    ) -> list[list[dict[str, str]]]:
        batches: list[list[dict[str, str]]] = []
        current: list[dict[str, str]] = []
        current_chars = 0
        for chunk in chunks:
            next_chars = len(chunk["text"])
            if current and (
                len(current) >= MARKDOWN_LLM_BATCH_CHUNKS
                or current_chars + next_chars > MARKDOWN_LLM_MAX_CONTEXT_CHARS
            ):
                batches.append(current)
                current = []
                current_chars = 0
            current.append(chunk)
            current_chars += next_chars
        if current:
            batches.append(current)
        return batches

    def _extract_markdown_batch(
        self,
        task: PublicTask,
        batch_index: int,
        batch_chunks: list[dict[str, str]],
        allowed_tables: set[str],
    ) -> tuple[int, str, dict[str, list[dict[str, Any]]], str | None]:
        try:
            raw_response, payload = self._complete_json(
                self._build_markdown_extraction_prompt(task, batch_chunks)
            )
            tables = payload.get("tables")
            if not isinstance(tables, dict):
                raise ValueError("Markdown extraction response must contain a tables object.")
            rows_by_table: dict[str, list[dict[str, Any]]] = {}
            for table_name, rows in tables.items():
                normalized_table = re.sub(r"\W+", "_", str(table_name)).strip("_").lower()
                if normalized_table not in allowed_tables or not isinstance(rows, list):
                    continue
                cleaned_rows = [row for row in rows if isinstance(row, dict)]
                if cleaned_rows:
                    rows_by_table.setdefault(normalized_table, []).extend(cleaned_rows)
            return batch_index, raw_response, rows_by_table, None
        except Exception as exc:  # noqa: BLE001
            return batch_index, "", {}, str(exc)

    def _augment_markdown_tables_with_llm(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        engine: DataEngine,
        allowed_doc_tables: set[str] | None = None,
    ) -> None:
        chunks = self._select_markdown_chunks_for_question(
            task,
            allowed_doc_tables=allowed_doc_tables,
        )
        if not chunks:
            return

        try:
            batches = self._markdown_extraction_batches(chunks)
            rows_by_table: dict[str, list[dict[str, Any]]] = {}
            raw_responses: list[str] = []
            batch_errors: list[str] = []
            allowed_tables = {chunk["table"] for chunk in chunks}

            max_workers = max(
                1,
                min(self.config.markdown_extract_max_workers, len(batches)),
            )
            batch_results: list[tuple[int, str, dict[str, list[dict[str, Any]]], str | None]] = []
            if max_workers == 1:
                for batch_index, batch_chunks in enumerate(batches, start=1):
                    batch_results.append(
                        self._extract_markdown_batch(
                            task,
                            batch_index,
                            batch_chunks,
                            allowed_tables,
                        )
                    )
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_batch = {
                        executor.submit(
                            self._extract_markdown_batch,
                            task,
                            batch_index,
                            batch_chunks,
                            allowed_tables,
                        ): batch_index
                        for batch_index, batch_chunks in enumerate(batches, start=1)
                    }
                    for future in as_completed(future_to_batch):
                        batch_results.append(future.result())

            for batch_index, raw_response, batch_rows, error in sorted(
                batch_results,
                key=lambda item: item[0],
            ):
                if error is not None:
                    batch_errors.append(f"batch {batch_index}: {error}")
                    continue
                raw_responses.append(f"[batch {batch_index}]\n{raw_response}")
                for table_name, rows in batch_rows.items():
                    rows_by_table.setdefault(table_name, []).extend(rows)

            registered_tables: list[str] = []
            for normalized_table, cleaned_rows in rows_by_table.items():
                deduped_rows: list[dict[str, Any]] = []
                seen_rows: set[str] = set()
                for row in cleaned_rows:
                    row_key = json.dumps(row, sort_keys=True, default=str, ensure_ascii=False)
                    if row_key in seen_rows:
                        continue
                    seen_rows.add(row_key)
                    deduped_rows.append(row)
                if not deduped_rows:
                    continue
                base_columns = set(engine.catalog.get(normalized_table, {}).get("columns", []))
                extracted_columns = {
                    str(column)
                    for row in deduped_rows
                    for column in row
                    if str(column) != "source_chunk_id"
                }
                if base_columns and extracted_columns and extracted_columns <= base_columns:
                    continue
                output_table_name = (
                    f"{normalized_table}_llm" if base_columns else normalized_table
                )
                registered_name = engine.register_rows(
                    output_table_name,
                    deduped_rows,
                    source_type="markdown_llm",
                    source_path=f"{task.task_id}:{normalized_table}",
                )
                if registered_name is not None:
                    registered_tables.append(registered_name)

            if not registered_tables and not engine.tables:
                error_detail = "; ".join(batch_errors) if batch_errors else "no rows returned"
                raise ValueError(f"Markdown extraction produced no queryable rows: {error_detail}")

            self._append_step(
                task.task_id,
                state,
                phase="markdown_extract",
                thought="Extracted query-focused rows from markdown batches.",
                action="extract_markdown_rows",
                action_input={
                    "chunk_count": len(chunks),
                    "batch_count": len(batches),
                    "batch_max_workers": max_workers,
                    "allowed_doc_tables": sorted(allowed_doc_tables) if allowed_doc_tables else None,
                    "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
                },
                raw_response="\n\n".join(raw_responses),
                observation={
                    "ok": True,
                    "content": {
                        "registered_tables": registered_tables,
                        "batch_errors": batch_errors,
                    },
                },
                ok=True,
            )
        except Exception as exc:  # noqa: BLE001
            self._append_step(
                task.task_id,
                state,
                phase="markdown_extract",
                thought="Markdown LLM extraction failed; continuing with existing DataEngine tables.",
                action="extract_markdown_rows",
                action_input={
                    "chunk_count": len(chunks),
                    "allowed_doc_tables": sorted(allowed_doc_tables) if allowed_doc_tables else None,
                    "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
                },
                raw_response="",
                observation={
                    "ok": False,
                    "error": str(exc),
                },
                ok=False,
            )

    def _append_step(
        self,
        task_id: str,
        state: AgentRuntimeState,
        *,
        phase: str,
        thought: str,
        action: str,
        action_input: dict[str, Any],
        raw_response: str,
        observation: dict[str, Any],
        ok: bool,
    ) -> None:
        state.steps.append(
            StepRecord(
                step_index=len(state.steps) + 1,
                thought=thought,
                action=action,
                action_input=action_input,
                raw_response=raw_response,
                observation=observation,
                ok=ok,
                phase=phase,
            )
        )
        self._emit_trace(task_id, state, status="running")

    def _emit_trace(self, task_id: str, state: AgentRuntimeState, *, status: str) -> None:
        if self.trace_callback is None:
            return
        self.trace_callback(
            build_trace_payload(
                task_id=task_id,
                state=state,
                status=status,
            )
        )

    def _load_data(self, task: PublicTask, state: AgentRuntimeState) -> DataEngine:
        engine = DataEngine()
        loaded_data = engine.register_context_dir(task.context_dir)
        loaded_ok = bool(loaded_data.get("success"))
        state.loaded_data = loaded_data
        self._append_step(
            task.task_id,
            state,
            phase="load_data",
            thought="Loaded supported context files into DataEngine.",
            action="load_data",
            action_input={"context_dir": str(task.context_dir)},
            raw_response="",
            observation={
                "ok": loaded_ok,
                "content": loaded_data,
            },
            ok=loaded_ok,
        )
        if not loaded_ok:
            raise RuntimeError("DataEngine failed to load all supported context files.")

        if not engine.tables:
            self._augment_markdown_tables_with_llm(task, state, engine)
        else:
            existing_tables = set(engine.tables)
            missing_doc_tables = self._markdown_doc_table_names(task) - existing_tables
            if missing_doc_tables:
                self._augment_markdown_tables_with_llm(
                    task,
                    state,
                    engine,
                    allowed_doc_tables=missing_doc_tables,
                )
        loaded_data["table_count"] = len(engine.tables)
        if not engine.tables:
            raise RuntimeError("DataEngine failed to load or extract any queryable tables.")
        return engine

    def _generate_catalog(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        engine_schema: dict[str, Any],
    ) -> dict[str, Any]:
        raw_response, payload = self._complete_json(
            build_catalog_prompt(
                task,
                task_record=self._task_record_payload(task),
                schema_knowledge_text=self._catalog_knowledge_text(task),
                engine_schema=engine_schema,
            )
        )
        catalog = payload.get("catalog")
        if not isinstance(catalog, dict):
            raise ValueError("Catalog stage response must contain a catalog object.")
        compact_catalog = _compact_catalog(_align_catalog_with_schema(catalog, engine_schema))
        state.catalog = compact_catalog
        self._append_step(
            task.task_id,
            state,
            phase="catalog",
            thought=str(payload.get("thought", "")),
            action="generate_catalog",
            action_input={},
            raw_response=raw_response,
            observation={
                "ok": True,
                "content": {
                    "catalog": compact_catalog,
                },
            },
            ok=True,
        )
        return compact_catalog

    def _build_knowledge_context(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        *,
        engine_schema: dict[str, Any],
        catalog: dict[str, Any],
    ) -> tuple[str, str, list[KnowledgeChunk]]:
        empty_result = ("(none)", "(none)", [])

        if not self.config.enable_knowledge_retrieval:
            self._append_step(
                task.task_id,
                state,
                phase="knowledge",
                thought="Knowledge retrieval is disabled by config.",
                action="retrieve_knowledge",
                action_input={"enabled": False},
                raw_response="",
                observation={
                    "ok": True,
                    "content": {
                        "enabled": False,
                        "chunk_count": 0,
                        "retrieved_for_plan": [],
                        "sql_knowledge_constraints": "(none)",
                    },
                },
                ok=True,
            )
            return empty_result

        try:
            knowledge_text = self._catalog_knowledge_text(task)
            chunks = parse_knowledge_markdown(
                knowledge_text,
                chunk_max_chars=self.config.knowledge_chunk_max_chars,
            )
            schema_terms = _collect_schema_terms(engine_schema)
            catalog_terms = _collect_catalog_terms(catalog)
            plan_chunks = retrieve_knowledge_chunks(
                chunks,
                query=task.question,
                schema_terms=schema_terms,
                catalog_terms=catalog_terms,
                top_k=self.config.knowledge_top_k_plan,
                mode="plan",
            )
            retrieved_knowledge = render_retrieved_knowledge(plan_chunks)
            sql_constraints = build_sql_knowledge_constraints(
                plan_chunks,
                top_k=self.config.knowledge_top_k_sql,
            )

            state.knowledge_chunks = knowledge_corpus_summary(chunks)
            state.retrieved_knowledge = [
                knowledge_chunk_summary(chunk) for chunk in plan_chunks
            ]
            state.sql_knowledge_constraints = sql_constraints

            self._append_step(
                task.task_id,
                state,
                phase="knowledge",
                thought=(
                    "Retrieved task-relevant knowledge for planning and compact SQL guardrails."
                ),
                action="retrieve_knowledge",
                action_input={
                    "enabled": True,
                    "top_k_plan": self.config.knowledge_top_k_plan,
                    "top_k_sql": self.config.knowledge_top_k_sql,
                },
                raw_response="",
                observation={
                    "ok": True,
                    "content": {
                        "chunk_count": len(chunks),
                        "retrieved_for_plan": state.retrieved_knowledge,
                        "sql_knowledge_constraints": sql_constraints,
                    },
                },
                ok=True,
            )
            return retrieved_knowledge, sql_constraints, plan_chunks
        except Exception as exc:  # noqa: BLE001
            state.knowledge_chunks = []
            state.retrieved_knowledge = []
            state.sql_knowledge_constraints = "(none)"
            self._append_step(
                task.task_id,
                state,
                phase="knowledge",
                thought="Knowledge retrieval failed; continuing without retrieved knowledge.",
                action="retrieve_knowledge",
                action_input={"enabled": True},
                raw_response="",
                observation={
                    "ok": True,
                    "content": {
                        "warning": str(exc),
                        "chunk_count": 0,
                        "retrieved_for_plan": [],
                        "sql_knowledge_constraints": "(none)",
                    },
                },
                ok=True,
            )
            return empty_result

    def _generate_plan(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        catalog: dict[str, Any],
        engine_schema: dict[str, Any],
        retrieved_knowledge: str,
    ) -> dict[str, Any]:
        plan_doc_files, plan_doc_text = self._plan_doc_bundle(task)
        raw_response, payload = self._complete_json(
            build_plan_prompt(
                task,
                catalog=catalog,
                engine_schema=engine_schema,
                plan_doc_text=plan_doc_text,
                retrieved_knowledge=retrieved_knowledge,
            )
        )
        plan = payload.get("plan")
        if not isinstance(plan, dict):
            raise ValueError("Plan stage response must contain a plan object.")
        state.plan = plan
        state.focused_schema = _build_focused_schema(engine_schema, plan, catalog)
        self._append_step(
            task.task_id,
            state,
            phase="plan",
            thought=str(payload.get("thought", "")),
            action="generate_plan",
            action_input={},
            raw_response=raw_response,
            observation={
                "ok": True,
                "content": {
                    "plan_doc_files": plan_doc_files,
                    "retrieved_knowledge": state.retrieved_knowledge,
                    "plan": plan,
                    "focused_schema": state.focused_schema,
                },
            },
            ok=True,
        )
        return plan

    def _run_nl2sql_loop(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        engine: DataEngine,
        catalog: dict[str, Any],
        plan: dict[str, Any],
        engine_schema: dict[str, Any],
        sql_knowledge_constraints: str,
    ) -> None:
        tools = create_dataengine_tool_registry(
            engine,
            generated_catalog=catalog,
            sql_result_limit=self.config.sql_result_limit,
            schema_sample_rows=self.config.catalog_sample_rows,
        )
        max_attempts = (
            self.config.max_sql_attempts
            if self.config.max_sql_attempts > 0
            else self.config.max_steps
        )
        last_successful_sql: str | None = None
        last_successful_result: dict[str, Any] | None = None
        for _ in range(max_attempts):
            recent_attempts = state.sql_attempts[-SQL_HISTORY_WINDOW:]
            raw_response, payload = self._complete_json(
                build_nl2sql_prompt(
                    task,
                    plan=_compact_plan(plan),
                    engine_schema=engine_schema,
                    focused_schema=state.focused_schema,
                    recent_attempts=recent_attempts,
                    tool_descriptions=tools.describe_for_prompt(["execute_dataengine_sql"]),
                    sql_result_limit=self.config.sql_result_limit,
                    sql_knowledge_constraints=sql_knowledge_constraints,
                )
            )
            try:
                model_step = parse_model_step(raw_response)
                if model_step.action != "execute_dataengine_sql":
                    raise ValueError("plan2sql stage must call execute_dataengine_sql.")
                tool_result = tools.execute(task, model_step.action, model_step.action_input)
                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                attempt = _summarize_sql_attempt(
                    sql=str(model_step.action_input.get("sql", "")),
                    ok=tool_result.ok,
                    is_final=bool(payload.get("is_final")),
                    result=tool_result.content,
                )
                state.sql_attempts.append(attempt)
                self._append_step(
                    task.task_id,
                    state,
                    phase="plan2sql",
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=tool_result.ok,
                )
                if tool_result.ok:
                    last_successful_sql = str(model_step.action_input.get("sql", ""))
                    last_successful_result = tool_result.content
                if tool_result.ok and bool(payload.get("is_final")):
                    state.final_sql = last_successful_sql
                    state.final_sql_result = last_successful_result
                    self._emit_trace(task.task_id, state, status="running")
                    return
            except Exception as exc:
                sql_value = None
                if isinstance(payload.get("action_input"), dict):
                    sql_value = payload["action_input"].get("sql")
                observation = {
                    "ok": False,
                    "error": str(exc),
                }
                state.sql_attempts.append(
                    _summarize_sql_attempt(
                        sql=str(sql_value) if sql_value is not None else None,
                        ok=False,
                        is_final=bool(payload.get("is_final")),
                        error_message=str(exc),
                    )
                )
                self._append_step(
                    task.task_id,
                    state,
                    phase="plan2sql",
                    thought=str(payload.get("thought", "")),
                    action="__error__",
                    action_input={},
                    raw_response=raw_response,
                    observation=observation,
                    ok=False,
                )
        if last_successful_sql is not None and last_successful_result is not None:
            state.final_sql = last_successful_sql
            state.final_sql_result = last_successful_result
            self._emit_trace(task.task_id, state, status="running")
            return
        raise RuntimeError("Agent did not produce a successful SQL query within max_sql_attempts.")

    def _generate_answer(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        engine: DataEngine,
        catalog: dict[str, Any],
        plan: dict[str, Any],
    ) -> None:
        if state.final_sql is None or state.final_sql_result is None:
            raise RuntimeError("Answer stage requires a final SQL result.")

        direct_answer = _build_answer_from_result(state.final_sql_result, plan)
        if direct_answer is not None:
            state.answer = direct_answer
            self._append_step(
                task.task_id,
                state,
                phase="answer",
                thought="Directly converted final SQL result into the answer table.",
                action="answer_direct",
                action_input={
                    "columns": direct_answer.columns,
                    "row_count": len(direct_answer.rows),
                },
                raw_response="",
                observation={
                    "ok": True,
                    "content": {
                        "column_count": len(direct_answer.columns),
                        "row_count": len(direct_answer.rows),
                    },
                },
                ok=True,
            )
            return

        tools = create_dataengine_tool_registry(
            engine,
            generated_catalog=catalog,
            sql_result_limit=self.config.sql_result_limit,
            schema_sample_rows=self.config.catalog_sample_rows,
        )
        raw_response = self.model.complete(
            self._build_messages(
                build_answer_prompt(
                    task,
                    plan=_compact_plan(plan),
                    final_sql=state.final_sql,
                    final_sql_result=state.final_sql_result,
                    tool_descriptions=tools.describe_for_prompt(["answer"]),
                )
            )
        )
        model_step = parse_model_step(raw_response)
        if model_step.action != "answer":
            raise ValueError("Answer stage must call answer.")
        tool_result = tools.execute(task, model_step.action, model_step.action_input)
        observation = {
            "ok": tool_result.ok,
            "tool": model_step.action,
            "content": tool_result.content,
        }
        self._append_step(
            task.task_id,
            state,
            phase="answer",
            thought=model_step.thought,
            action=model_step.action,
            action_input=model_step.action_input,
            raw_response=raw_response,
            observation=observation,
            ok=tool_result.ok,
        )
        if tool_result.is_terminal:
            state.answer = tool_result.answer

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        try:
            engine = self._load_data(task, state)
            engine_schema = engine.describe_schema(sample_rows=self.config.catalog_sample_rows)
            state.engine_schema = engine_schema
            catalog = self._generate_catalog(task, state, engine_schema)
            retrieved_knowledge, sql_knowledge_constraints, _ = self._build_knowledge_context(
                task,
                state,
                engine_schema=engine_schema,
                catalog=catalog,
            )
            plan = self._generate_plan(
                task,
                state,
                catalog,
                engine_schema,
                retrieved_knowledge,
            )
            self._run_nl2sql_loop(
                task,
                state,
                engine,
                catalog,
                plan,
                engine_schema,
                sql_knowledge_constraints,
            )
            self._generate_answer(task, state, engine, catalog, plan)
        except Exception as exc:
            state.failure_reason = str(exc)
            self._append_step(
                task.task_id,
                state,
                phase="error",
                thought="",
                action="__error__",
                action_input={},
                raw_response="",
                observation={
                    "ok": False,
                    "error": str(exc),
                },
                ok=False,
            )

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer."
            self._emit_trace(task.task_id, state, status="failed")

        final_status = "failed" if state.failure_reason is not None else "completed"
        self._emit_trace(task.task_id, state, status=final_status)

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
            final_sql=state.final_sql,
            status=final_status,
        )
