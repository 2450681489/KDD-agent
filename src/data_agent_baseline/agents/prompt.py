from __future__ import annotations

import json

from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask


BASE_SYSTEM_PROMPT = """
You are a staged NL2SQL data agent. Solve the task using only the already-loaded DuckDB/DataEngine tables and the provided task documents. Always return exactly one JSON object wrapped in a single ```json fenced block and no extra text.

---

# Data Representation

- The model does NOT see raw CSV files, raw JSON documents, or raw SQLite files at query time.
- All usable data has already been loaded into DuckDB/DataEngine as structured relational tables.
- Treat every table in the schema as a normal relational table with columns and rows.
- Sample rows are only tabular previews of loaded tables, not nested JSON to traverse and not CSV text to parse.
- Never reason as if you need to read JSON paths, parse CSV delimiters, or inspect file formats inside SQL.

# Data Usage Rules

- Only use columns that exist in the observed schema.
- Do NOT create or rename columns.
- DataEngine exposes lowercase table names directly. When writing SQL, use table names exactly as shown in the schema/catalog and do not quote table names.
- Allowed transformations:
  - Original columns
  - Aggregations: SUM, AVG, COUNT, MIN, MAX(original_column)

- Column format for aggregation:
  AGG_FUNCTION(original_column_name)

- Do not perform string operations or column concatenation.

- Do not assume indirect relationships unless explicitly required.
- Always verify whether a direct relationship exists.

---

# Output Rules

- Return only columns required by the question.
- Column names must exactly match schema or valid aggregations.
- Output must contain exactly one final answer via `answer` tool with:
  {columns, rows}
- Evaluation focuses on the set of answer values; however, you must still provide valid output columns.

---

# Schema & Semantic Rules

- All columns must be traceable to observed tables.
- Do not use intermediate/proxy fields if a higher-level semantic field is required.
- When multiple tables are involved, ensure correct join path to target entity.

---

# Validation (before answer)

Check:
- no new columns created
- all columns exist in schema
- correct join path used
- output matches question requirement exactly
""".strip()

REACT_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT

CATALOG_PROMPT = """
Stage: read_schema_doc & generate catalog.

Read the task metadata, task question, DataEngine schema, sample rows, and `context/knowledge.md`. Build a compact semantic catalog that keeps only core fields needed for later planning and plan2sql generation.

Input format:
- `task`: structured task metadata.
- `schema knowledge`: text from `context/knowledge.md`, mainly used to understand table semantics and schema meaning.
- `DataEngine schema`: structured relational table metadata.
- `sample rows`: row previews from already-loaded DuckDB/DataEngine tables.

Important:
- You are reasoning about relational tables, columns, and row values.
- Do not describe the data as raw CSV, raw JSON, nested documents, or file contents.
- Use `context/knowledge.md` to understand what tables and columns mean.
- Use `context/knowledge.md` only as a semantic supplement for tables and columns that already exist in the DataEngine schema.
- Never invent new tables or columns from `knowledge.md`. If the document mentions a field that is not present in the DataEngine schema, do not add it to the catalog.
- Tables whose names end with `_llm` are query-focused markdown extraction supplements.
  If a base table with the same prefix exists and already has the needed fields, prefer the
  base table as the primary source. Use `_llm` tables only for missing fields, cross-checks,
  or explicit fallback; do not automatically union base and `_llm` tables.

Return JSON with this shape:
{
  "thought": "brief reasoning",
  "catalog": {
    "tables": {
      "table_name": {
        "description": "semantic description",
        "columns": {
          "column_name": {
            "semantic": "meaning",
            "type_hint": "identifier/category/measure/date/text/unknown",
            "notes": "query-relevant notes"
          }
        }
      }
    },
    "join_hints": [
      {
        "left_table": "table",
        "left_column": "column",
        "right_table": "table",
        "right_column": "column",
        "confidence": "high/medium/low"
      }
    ],
    "task_relevant_fields": ["table.column"],
    "warnings": []
  }
}

Requirements:
- Keep the catalog concise and practical.
- Keep only core fields shown above.
- Every table name and column name in the catalog must come from the DataEngine schema.
- Use knowledge.md to enrich descriptions, type hints, notes, and join understanding for existing schema fields only.
- Do not include examples, duplicated schema text, long prose, or unused metadata.

Do not write SQL and do not answer the task in this stage.
""".strip()

SCHEMA_LINKING_PROMPT = """
Stage: schema_linking.

Read the question and identify literal or fuzzy values that are likely used as
filters, joins, entity lookups, or value constraints in SQL. These are not column
names; they are values or phrases that may appear inside database rows, possibly
with different spelling/casing/wording.

Return JSON with this shape:
{
  "thought": "brief reasoning",
  "filter_terms": [
    {
      "term": "literal or fuzzy lookup value",
      "kind": "entity/category/date/number/text",
      "reason": "why this value may need database lookup",
      "alternatives": ["optional synonym or rewritten lookup term"]
    }
  ]
}

Example:
{
  "thought": "The event name and shirt size are row values used as filters.",
  "filter_terms": [
    {
      "term": "Women's Soccer",
      "kind": "entity",
      "reason": "Specific event name that should be matched to an event/title column.",
      "alternatives": ["Women Soccer", "Soccer"]
    },
    {
      "term": "Medium",
      "kind": "category",
      "reason": "T-shirt size category used as a member filter.",
      "alternatives": ["M", "medium"]
    }
  ]
}

Rules:
- Include named entities, event names, product/race/school/person names, categories,
  statuses, formats, dates, month/year phrases, and numeric identifiers when they are
  likely used to filter rows.
- Do not include generic analytical words such as count, average, maximum, minimum,
  percentage, total, row, table, column, value, records.
- If a term is fuzzy or could be encoded differently in the database, include 1-3
  alternatives, e.g. "women soccer" for "Women's Soccer", "commander" for
  "format commander", or a YYYYMM rewrite for a month/year phrase.
- Keep the list small and high precision, usually 1-8 terms.
""".strip()

SCHEMA_LINKING_REFINE_PROMPT = """
Stage: schema_linking_refine.

Some lookup terms from the previous schema-linking round had no database hits.
Read the question, schema, no-hit terms, and previous search results. Propose
revised lookup terms that are more likely to match stored database values.

Return JSON with this shape:
{
  "thought": "brief reasoning about why terms were revised",
  "revised_terms": [
    {
      "original_term": "term that had no hits",
      "term": "new lookup term",
      "reason": "why this rewrite may match database rows",
      "alternatives": ["optional additional lookup term"]
    }
  ]
}

Example:
{
  "thought": "The original phrase includes prose words; database values are likely shorter names or encoded dates.",
  "revised_terms": [
    {
      "original_term": "format commander and legal status",
      "term": "commander",
      "reason": "The format value may be stored as just the format name.",
      "alternatives": ["Commander"]
    },
    {
      "original_term": "August 2012",
      "term": "201208",
      "reason": "The month may be stored as a YYYYMM integer/string.",
      "alternatives": ["2012-08", "2012/08"]
    }
  ]
}

Rules:
- Only revise terms that had no useful hits. Do not repeat terms that already had good hits.
- Prefer shorter database-like values over full natural-language phrases.
- Use schema column names and sample rows to infer likely encodings, but output lookup values only.
- Do not write SQL and do not answer the task.
- Keep revised_terms small and high precision, usually 0-8 entries.
""".strip()

PLAN_PROMPT = """
Stage: plan.

Use the task question, observed schema, generated catalog, and `context/doc/*.md` documents to produce a concrete query plan. The plan must capture nearly all query logic needed for the final SQL so that the next stage mainly translates the plan into SQL instead of re-thinking the task.

Input format:
- `question`: natural-language task goal.
- `generated catalog`: semantic summary of relational tables and columns.
- `DataEngine schema`: loaded structured tables with column lists and sample rows.
- `plan documents`: markdown/text files from `context/doc/`, used to find query conditions and business rules.

Important:
- Plan over relational tables, joins, filters, grouping, ordering, and final output columns.
- Do not plan around raw file parsing or nested JSON traversal.
- Read `context/doc/*.md` to find business rules, join conditions, aggregation definitions, aliases, and filtering constraints.
- Use those document-derived conditions explicitly in the plan when they affect joins, filters, aggregations, grouping, ordering, or validation.
- Read markdown/text documents as task instructions or semantic hints, not as queryable tables.
- For current-age wording such as "age", "under N", "over N", "not N yet",
  or "aren't N yet", calculate age from the current year and the patient's
  birth year unless the question explicitly asks for age at an event/test/admission
  date. Do not use lab, examination, admission, or row dates as a proxy for current
  age without explicit wording.
- Tables ending in `_llm` are supplemental rows extracted from selected markdown chunks.
  Prefer the corresponding base table when it contains the needed columns. Use `_llm`
  only when the base table lacks the needed field/value, as a validation aid, or when the
  task specifically needs the extracted subset. Avoid unioning base and `_llm` versions
  unless the plan can justify that they are non-overlapping sources.

Return JSON with this shape:
{
  "thought": "brief reasoning",
  "plan": {
    "target": "what the final table should contain",
    "source_tables": ["table"],
    "table_roles": {
      "table": "fact/dimension/helper"
    },
    "join_steps": [
      {
        "left": "table.column",
        "right": "table.column",
        "join_type": "inner/left"
      }
    ],
    "filters": [
      {
        "field": "table.column",
        "condition": "condition"
      }
    ],
    "select_expressions": [
      {
        "expression": "table.column or valid aggregation",
        "alias": "optional_alias_or_null"
      }
    ],
    "aggregations": [
      {
        "expression": "SUM(column)",
        "group_by": ["table.column"]
      }
    ],
    "group_by": ["table.column"],
    "order_by": [
      {
        "expression": "table.column or valid aggregation",
        "direction": "ASC/DESC"
      }
    ],
    "limit": null,
    "distinct": false,
    "final_columns": ["column_or_valid_aggregation"],
    "validation_checks": []
  }
}

Requirements:
- Plan for the final answer query, not an exploratory query.
- Include enough detail for the next stage to generate the final SQL in one pass whenever possible.
- Do not invent fields or joins that are not supported by the observed schema.
- Use null for unavailable optional fields instead of omitting keys.
- If a `context/doc/*.md` file provides a join path, aggregation meaning, or filter rule, reflect it in `join_steps`, `filters`, `aggregations`, `group_by`, `order_by`, or `validation_checks`.

Do not execute SQL and do not answer the task in this stage.
""".strip()

NL2SQL_PROMPT = """
Stage: plan2sql.

Generate or revise one read-only DuckDB SQL query that faithfully implements the provided plan. Treat the plan as the source of truth for query intent. Use table names exactly as shown in the DataEngine schema/catalog. Use only SELECT or WITH queries. Do not use Python, file-reading functions, DDL, DML, ATTACH, INSTALL, or LOAD.

Input format:
- `plan`: structured query specification.
- `focused schema` / `full schema`: relational tables already loaded in DuckDB/DataEngine.
- `recent compile/execute summaries`: brief feedback from previous SQL attempts.

Important:
- The schema describes loaded relational tables. It is not raw JSON and not raw CSV.
- For current-age filters or expressions, use `CURRENT_TIMESTAMP` with the birth
  date column, for example `EXTRACT(YEAR FROM CURRENT_TIMESTAMP) - EXTRACT(YEAR FROM CAST(Birthday AS DATE))`.
  Use an event date such as lab/exam/admission date only when the plan or question
  explicitly asks for age at that event.
- Write SQL against tables and columns only.
- Do not use JSON extraction syntax, file-reading assumptions, or CSV parsing logic unless such values are already present as ordinary string columns.
- Use lowercase table names exactly as shown in the schema/catalog.
- Do not quote table names unless the SQL dialect absolutely requires it; for DataEngine tables, use unquoted lowercase names.

Return JSON with this shape:
{
  "thought": "brief reasoning",
  "action": "execute_dataengine_sql",
  "action_input": {
    "sql": "SELECT ...",
    "limit": 200
  },
  "is_final": true
}

Requirements:
- Aim to generate the final answer SQL in one pass.
- Do not redesign the query logic if the plan is already sufficient.
- If a previous attempt failed, revise SQL minimally while preserving the plan semantics.
- Do not repeat a SQL query that just failed. Use the error text to make a concrete change.
- If DuckDB reports a conversion error such as "Could not convert string 'X' to DOUBLE"
  for a string comparison, cast the compared column to VARCHAR, e.g.
  `CAST(table.column AS VARCHAR) = 'X'`. This commonly happens when an extracted
  markdown column was inferred as numeric even though the logical field is textual.
- If DuckDB reports a type/cast error around date or year extraction, cast the source
  expression to DATE/TIMESTAMP before calling date functions.
- Prefer focused_schema when it is provided; use full_schema only as a fallback reference.
- Set "is_final" to true only when the query result is intended to be the final result table for the answer stage.
""".strip()

ANSWER_PROMPT = """
Stage: answer.

Use the final SQL result to submit the final table through the answer tool. Do not re-query data, do not invent rows, and do not add columns that are not required by the question. Keep this stage lightweight: format the provided final result into the answer tool payload.

Input format:
- `final SQL result`: an already computed relational result table with `columns` and `rows`.

Important:
- Treat the input as a final tabular result, not raw JSON to transform and not CSV text to rewrite.
- Only package the provided table into the answer payload.

Return JSON with this shape:
{
  "thought": "brief reasoning",
  "action": "answer",
  "action_input": {
    "columns": ["column_name"],
    "rows": [["value"]]
  }
}
""".strip()


def _render_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def build_system_prompt(tool_descriptions: str = "", system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or BASE_SYSTEM_PROMPT
    if not tool_descriptions:
        return base_prompt
    return f"{base_prompt}\n\nAvailable tools:\n{tool_descriptions}"


def build_task_prompt(task: PublicTask) -> str:
    return f"Question: {task.question}"


def build_catalog_prompt(
    task: PublicTask,
    *,
    task_record: dict[str, Any],
    schema_knowledge_text: str,
    engine_schema: dict[str, Any],
) -> str:
    return (
        f"{CATALOG_PROMPT}\n\n"
        f"Task:\n{_render_payload(task_record)}\n\n"
        f"Question:\n{task.question}\n\n"
        f"Schema knowledge from context/knowledge.md:\n{schema_knowledge_text or '(missing)'}\n\n"
        f"DataEngine schema:\n{_render_payload(engine_schema)}"
    )


def build_schema_linking_prompt(
    task: PublicTask,
    *,
    catalog: dict[str, Any],
    engine_schema: dict[str, Any],
    retrieved_knowledge: str = "(none)",
) -> str:
    return (
        f"{SCHEMA_LINKING_PROMPT}\n\n"
        f"Question:\n{task.question}\n\n"
        f"Generated catalog:\n{_render_payload(catalog)}\n\n"
        "Retrieved knowledge from context/knowledge.md:\n"
        f"{retrieved_knowledge or '(none)'}\n\n"
        f"DataEngine schema:\n{_render_payload(engine_schema)}"
    )


def build_schema_linking_refine_prompt(
    task: PublicTask,
    *,
    catalog: dict[str, Any],
    engine_schema: dict[str, Any],
    retrieved_knowledge: str,
    previous_linking: dict[str, Any],
    no_hit_terms: list[dict[str, Any]],
) -> str:
    return (
        f"{SCHEMA_LINKING_REFINE_PROMPT}\n\n"
        f"Question:\n{task.question}\n\n"
        f"Generated catalog:\n{_render_payload(catalog)}\n\n"
        "Retrieved knowledge from context/knowledge.md:\n"
        f"{retrieved_knowledge or '(none)'}\n\n"
        f"No-hit terms from previous round:\n{_render_payload({'terms': no_hit_terms})}\n\n"
        f"Previous search results:\n{_render_payload(previous_linking)}\n\n"
        f"DataEngine schema:\n{_render_payload(engine_schema)}"
    )


def build_plan_prompt(
    task: PublicTask,
    *,
    catalog: dict[str, Any],
    engine_schema: dict[str, Any],
    plan_doc_text: str,
    retrieved_knowledge: str = "(none)",
    schema_linking: dict[str, Any] | None = None,
) -> str:
    return (
        f"{PLAN_PROMPT}\n\n"
        f"Question:\n{task.question}\n\n"
        f"Generated catalog:\n{_render_payload(catalog)}\n\n"
        "Retrieved knowledge from context/knowledge.md:\n"
        f"{retrieved_knowledge or '(none)'}\n\n"
        "Knowledge usage rules:\n"
        "- Use retrieved knowledge for KPI formulas, calculation logic, filters, "
        "date/time formats, unit conversions, ambiguity resolution, and reusable query patterns.\n"
        "- Treat retrieved knowledge as semantic guidance only. Use only tables and columns "
        "that exist in the DataEngine schema.\n"
        "- If retrieved knowledge conflicts with the observed schema, the observed schema wins.\n"
        "- Capture important knowledge-derived constraints in filters, aggregations, group_by, "
        "order_by, final_columns, or validation_checks.\n\n"
        f"Plan documents from context/doc:\n{plan_doc_text or '(missing)'}\n\n"
        "Schema linking evidence from database value search:\n"
        f"{_render_payload(schema_linking or {'terms': []})}\n\n"
        "Schema linking usage rules:\n"
        "- Use search hits to map question values to real table columns and stored values.\n"
        "- If a question phrase has hits in one column, prefer that column for the filter.\n"
        "- If no exact hit exists, reason from alternatives and sample rows before choosing a filter.\n"
        "- Do not use search hits that contradict the observed schema or the question intent.\n\n"
        f"DataEngine schema:\n{_render_payload(engine_schema)}"
    )


def build_nl2sql_prompt(
    task: PublicTask,
    *,
    plan: dict[str, Any],
    engine_schema: dict[str, Any],
    focused_schema: dict[str, Any] | None,
    recent_attempts: list[dict[str, Any]],
    tool_descriptions: str,
    sql_result_limit: int,
    sql_knowledge_constraints: str = "(none)",
) -> str:
    schema_mode = "focused_schema" if focused_schema else "full_schema"
    return (
        f"{NL2SQL_PROMPT}\n\n"
        f"Available tools:\n{tool_descriptions}\n\n"
        f"Use limit={sql_result_limit} unless a smaller limit is sufficient.\n\n"
        f"Question:\n{task.question}\n\n"
        f"Plan:\n{_render_payload(plan)}\n\n"
        "SQL knowledge constraints:\n"
        f"{sql_knowledge_constraints or '(none)'}\n\n"
        "SQL constraint usage rules:\n"
        "- The plan is the source of truth. Use these constraints only as guardrails while "
        "translating it.\n"
        "- Do not redesign query intent from knowledge at this stage.\n"
        "- Do not use any table or column name from knowledge unless it exists in the "
        "focused/full schema.\n\n"
        f"Schema strategy: prefer {schema_mode} and fall back to full_schema only when needed.\n\n"
        f"Focused schema:\n{_render_payload(focused_schema or {'tables': {}})}\n\n"
        f"Full schema:\n{_render_payload(engine_schema)}\n\n"
        f"Recent compile/execute summaries:\n{_render_payload({'attempts': recent_attempts})}"
    )


def build_answer_prompt(
    task: PublicTask,
    *,
    plan: dict[str, Any],
    final_sql: str,
    final_sql_result: dict[str, Any],
    tool_descriptions: str,
) -> str:
    return (
        f"{ANSWER_PROMPT}\n\n"
        f"Available tools:\n{tool_descriptions}\n\n"
        f"Question:\n{task.question}\n\n"
        f"Plan:\n{_render_payload(plan)}\n\n"
        f"Final SQL:\n{final_sql}\n\n"
        f"Final SQL result:\n{_render_payload(final_sql_result)}"
    )


def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2, default=str)
    return f"Observation:\n{rendered}"
