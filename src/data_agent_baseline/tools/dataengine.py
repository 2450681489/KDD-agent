import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp
from datetime import date, datetime, time


DEFAULT_MAX_LOAD_FILE_BYTES = 512 * 1024 * 1024


class DataEngine:

    def __init__(self, db_path=":memory:", max_load_file_bytes=DEFAULT_MAX_LOAD_FILE_BYTES):
        self.conn = duckdb.connect(db_path)
        self.db_path = db_path
        self.max_load_file_bytes = max_load_file_bytes

        self.tables = set()
        self.sqlite_load = False
        self.catalog = {}

    def register(self, file_path):
        try:
            file_path = os.path.abspath(file_path)
            file_size_bytes = self._validate_load_file_size(file_path)
            file_type = self._detect_type(file_path)

            if file_type == "csv":
                table_name = self._reserve_table_name(self._make_table_name(file_path))
                self._register_csv(table_name, file_path)
            elif file_type == "json":
                table_name = self._reserve_table_name(self._make_table_name(file_path))
                self._register_json(table_name, file_path)
            else:
                table_name = self._make_table_name(file_path)
                self._register_sqlite(file_path)
            
            return {
                "success": True,
                "table": table_name,
                "source_path": file_path,
                "source_type": file_type,
                "file_size_bytes": file_size_bytes,
            }

        except FileNotFoundError:
            print(f"[ERROR] File not found: {file_path}")
            return {
                "success": False,
                "source_path": str(file_path),
                "error": "FileNotFoundError",
            }

        except Exception as e:
            print(f"[ERROR] register failed: {e}")
            return {
                "success": False,
                "source_path": str(file_path),
                "error": str(e),
            }

    def register_context_dir(self, context_dir: str | Path) -> dict[str, Any]:
        context_root = Path(context_dir).resolve()
        supported_suffixes = {".csv", ".json", ".db", ".sqlite"}
        metadata_filenames = {"task.json"}
        loaded_files = []
        failed_files = []
        for path in sorted(context_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in supported_suffixes:
                continue
            if path.name.lower() in metadata_filenames:
                continue
            result = self.register(path)
            relative_path = path.relative_to(context_root).as_posix()
            if result.get("success"):
                source_tables = self._tables_for_source_path(path)
                loaded_files.append(
                    {
                        "path": relative_path,
                        "table": result.get("table"),
                        "tables": source_tables,
                        "source_type": result.get("source_type"),
                        "file_size_bytes": result.get("file_size_bytes"),
                    }
                )
            else:
                failed_files.append(
                    {
                        "path": relative_path,
                        "error": result.get("error"),
                    }
                )
        return {
            "success": not failed_files,
            "context_dir": str(context_root),
            "loaded_files": loaded_files,
            "failed_files": failed_files,
            "table_count": len(self.tables),
            "max_load_file_bytes": self.max_load_file_bytes,
        }

    def query(self, sql, limit: int = 200):
        try:
            self._validate_read_only_sql(sql)
            limited_sql = self._apply_limit(sql, limit + 1)
            df = self.conn.execute(limited_sql).fetchdf()
            truncated = len(df.index) > limit
            if truncated:
                df = df.head(limit)
            return {
                "success": True,
                "data": {
                    "columns": df.columns.tolist(),
                    "rows": [
                        [self._json_safe_value(value) for value in row]
                        for row in df.values.tolist()
                    ],
                },
                "row_count": len(df.index),
                "truncated": truncated,
                "sql": sql,
                "rewritten_sql": sql,
                "error": None,
            }
        except Exception as e:
            retry_sql = self._rewrite_string_comparison_casts(sql, str(e))
            if retry_sql is not None and retry_sql != sql:
                try:
                    self._validate_read_only_sql(retry_sql)
                    limited_sql = self._apply_limit(retry_sql, limit + 1)
                    df = self.conn.execute(limited_sql).fetchdf()
                    truncated = len(df.index) > limit
                    if truncated:
                        df = df.head(limit)
                    return {
                        "success": True,
                        "data": {
                            "columns": df.columns.tolist(),
                            "rows": [
                                [self._json_safe_value(value) for value in row]
                                for row in df.values.tolist()
                            ],
                        },
                        "row_count": len(df.index),
                        "truncated": truncated,
                        "sql": sql,
                        "rewritten_sql": retry_sql,
                        "error": None,
                    }
                except Exception:
                    pass
            return {
                "success": False,
                "data": None,
                "row_count": 0,
                "truncated": False,
                "sql": sql,
                "rewritten_sql": None,
                "error": str(e),
            }

    def _rewrite_string_comparison_casts(self, sql: str, error_message: str) -> str | None:
        if "could not convert string" not in error_message.lower():
            return None
        pattern = re.compile(
            r"(?<![\w.])((?:[A-Za-z_][\w]*\.)?[A-Za-z_][\w]*)\s*=\s*'([^']*)'",
            flags=re.IGNORECASE,
        )
        replacements = 0

        def replace(match: re.Match[str]) -> str:
            nonlocal replacements
            column = match.group(1)
            literal = match.group(2)
            if column.upper() in {"CAST", "EXTRACT", "DATE", "TIMESTAMP"}:
                return match.group(0)
            replacements += 1
            return f"CAST({column} AS VARCHAR) = '{literal}'"

        rewritten = pattern.sub(replace, sql)
        if replacements == 0:
            return None
        return rewritten

    def show_tables(self):
        return {
            "count": len(self.tables),
            "tables": sorted(list(self.tables)),
        }

    def describe_schema(self, sample_rows: int = 3) -> dict[str, Any]:
        tables: dict[str, Any] = {}
        for table_name in sorted(self.catalog):
            entry = self.catalog[table_name]
            sample = self._sample_rows(table_name, sample_rows)
            tables[table_name] = {
                "source_type": entry["source_type"],
                "columns": list(entry["columns"]),
                "sample_rows": sample,
                "_meta": dict(entry.get("_meta", {})),
            }
        return {
            "table_count": len(tables),
            "tables": tables,
        }

    def global_search(
        self,
        terms: list[str],
        *,
        max_terms: int = 8,
        max_hits_per_term: int = 12,
        max_value_length: int = 160,
    ) -> dict[str, Any]:
        results: dict[str, Any] = {
            "terms": [],
            "max_hits_per_term": max_hits_per_term,
        }
        clean_terms: list[str] = []
        seen_terms: set[str] = set()
        for term in terms:
            clean_term = str(term).strip()
            if len(clean_term) < 2:
                continue
            key = clean_term.lower()
            if key in seen_terms:
                continue
            seen_terms.add(key)
            clean_terms.append(clean_term)
            if len(clean_terms) >= max_terms:
                break

        for term in clean_terms:
            hits: list[dict[str, Any]] = []
            pattern = f"%{term}%"
            for table_name, entry in sorted(self.catalog.items()):
                for column in entry.get("columns", []):
                    if len(hits) >= max_hits_per_term:
                        break
                    try:
                        table_sql = self._quote_identifier(str(table_name))
                        column_sql = self._quote_identifier(str(column))
                        query = (
                            f"SELECT {column_sql} AS _match_value, * "
                            f"FROM {table_sql} "
                            f"WHERE CAST({column_sql} AS VARCHAR) ILIKE {self._quote_string(pattern)} "
                            "LIMIT 2"
                        )
                        df = self.conn.execute(query).fetchdf()
                    except Exception:
                        continue
                    for row in df.to_dict(orient="records"):
                        if len(hits) >= max_hits_per_term:
                            break
                        raw_value = self._json_safe_value(row.pop("_match_value", None))
                        value = "" if raw_value is None else str(raw_value)
                        if len(value) > max_value_length:
                            value = value[: max_value_length - 3] + "..."
                        sample_row = {
                            key: self._json_safe_value(value)
                            for key, value in list(row.items())[:8]
                        }
                        hits.append(
                            {
                                "table": table_name,
                                "column": str(column),
                                "value": value,
                                "sample_row": sample_row,
                            }
                        )
                if len(hits) >= max_hits_per_term:
                    break
            results["terms"].append(
                {
                    "term": term,
                    "hit_count": len(hits),
                    "hits": hits,
                }
            )
        return results

    def register_rows(
        self,
        table_name: str,
        rows: list[dict[str, Any]],
        *,
        source_type: str = "extracted",
        source_path: str = "memory",
    ) -> str | None:
        if not rows:
            return None
        safe_table_name = self._reserve_table_name(self._sanitize_identifier(table_name))
        normalized_rows = self._normalize_extracted_rows(rows)
        self._register_dataframe(
            safe_table_name,
            normalized_rows,
            source_type=source_type,
            file_path=source_path,
        )
        return safe_table_name

    def _normalize_extracted_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        text_name_fragments = {
            "name",
            "status",
            "format",
            "type",
            "category",
            "label",
            "element",
            "sex",
            "gender",
            "diagnosis",
            "disease",
            "publisher",
            "source",
            "description",
            "title",
            "surname",
            "forename",
        }
        normalized_rows: list[dict[str, Any]] = []
        for row in rows:
            normalized_row: dict[str, Any] = {}
            for key, value in row.items():
                key_text = str(key)
                key_lower = key_text.lower()
                should_text = any(fragment in key_lower for fragment in text_name_fragments)
                if should_text and value is not None:
                    normalized_row[key_text] = str(value)
                else:
                    normalized_row[key_text] = value
            normalized_rows.append(normalized_row)
        return normalized_rows

    def _register_markdown_docs(self, context_root: Path) -> dict[str, Any]:
        doc_root = context_root / "doc"
        if not doc_root.is_dir():
            return {"loaded_files": [], "failed_files": []}

        loaded_files = []
        failed_files = []
        for path in sorted(doc_root.rglob("*.md")):
            try:
                rows = self._extract_markdown_rows(path)
                if not rows:
                    continue
                table_name = self._reserve_table_name(self._make_table_name(str(path)))
                self._register_dataframe(
                    table_name,
                    rows,
                    source_type="markdown",
                    file_path=str(path.resolve()),
                )
                loaded_files.append(
                    {
                        "path": path.relative_to(context_root).as_posix(),
                        "table": table_name,
                        "tables": [table_name],
                        "source_type": "markdown",
                        "file_size_bytes": path.stat().st_size,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                failed_files.append(
                    {
                        "path": path.relative_to(context_root).as_posix(),
                        "error": str(exc),
                    }
                )
        return {"loaded_files": loaded_files, "failed_files": failed_files}

    def _register_dataframe(
        self,
        table_name: str,
        rows: list[dict[str, Any]],
        *,
        source_type: str,
        file_path: str,
    ) -> None:
        df = pd.DataFrame(rows)
        temp_name = f"_tmp_{table_name}_{hashlib.md5(file_path.encode()).hexdigest()[:8]}"
        self.conn.register(temp_name, df)
        try:
            self.conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {temp_name};")
        finally:
            try:
                self.conn.unregister(temp_name)
            except Exception:
                pass
        self.tables.add(table_name)
        self.catalog[table_name] = {
            "source_type": source_type,
            "columns": self._get_columns(table_name),
            "_meta": {
                "file_path": file_path,
            },
        }

    def _extract_markdown_rows(self, path: Path) -> list[dict[str, Any]]:
        text = path.read_text(encoding="utf-8", errors="replace")
        table_hint = self._sanitize_identifier(path.stem)
        if table_hint == "patient":
            return self._extract_patient_markdown(text)
        if table_hint == "laboratory":
            return self._extract_laboratory_markdown(text)
        return self._extract_generic_markdown(text, table_hint)

    def _extract_patient_markdown(self, text: str) -> list[dict[str, Any]]:
        rows_by_id: dict[int, dict[str, Any]] = {}
        for chunk_index, (section, paragraph) in enumerate(self._markdown_paragraphs(text), start=1):
            patient_id = self._extract_patient_id(paragraph)
            if patient_id is None:
                continue
            row = rows_by_id.setdefault(
                patient_id,
                {
                    "ID": patient_id,
                    "SEX": None,
                    "Birthday": None,
                    "Description": None,
                    "First_Date": None,
                    "Admission": None,
                    "Diagnosis": None,
                    "source_chunk_ids": "",
                },
            )
            row["source_chunk_ids"] = self._append_source_id(
                row.get("source_chunk_ids"),
                f"patient:{chunk_index}",
            )

            sex = self._extract_sex(paragraph)
            if sex is not None:
                row["SEX"] = sex

            birthday = self._extract_birthday(paragraph)
            if birthday is not None:
                row["Birthday"] = birthday

            description = self._extract_record_date(paragraph)
            if description is not None:
                row["Description"] = description

            first_date = self._extract_first_visit_date(paragraph)
            if first_date is not None:
                row["First_Date"] = first_date

            admission = self._extract_admission(paragraph)
            if admission is not None:
                row["Admission"] = admission

            diagnosis = self._extract_diagnosis(paragraph)
            if diagnosis is not None:
                row["Diagnosis"] = diagnosis

        return list(rows_by_id.values())

    def _extract_laboratory_markdown(self, text: str) -> list[dict[str, Any]]:
        rows_by_key: dict[tuple[int, str | None], dict[str, Any]] = {}
        for chunk_index, (section, paragraph) in enumerate(self._markdown_paragraphs(text), start=1):
            patient_ids = self._extract_patient_ids(paragraph)
            if not patient_ids:
                continue
            row_date = self._extract_lab_date(paragraph)
            metrics = self._extract_lab_metrics(paragraph)
            if not metrics and not self._has_lab_null_panel(paragraph):
                continue

            for patient_id in patient_ids:
                key = (patient_id, row_date)
                row = rows_by_key.setdefault(
                    key,
                    {
                        "ID": patient_id,
                        "Date": row_date,
                        "section": section,
                        "source_chunk_ids": "",
                    },
                )
                row["source_chunk_ids"] = self._append_source_id(
                    row.get("source_chunk_ids"),
                    f"laboratory:{chunk_index}",
                )
                if row.get("Date") is None and row_date is not None:
                    row["Date"] = row_date
                if row.get("section") in (None, "") and section:
                    row["section"] = section
                row.update(metrics)

        return list(rows_by_key.values())

    def _extract_generic_markdown(self, text: str, table_hint: str) -> list[dict[str, Any]]:
        rows = []
        for chunk_index, (section, paragraph) in enumerate(self._markdown_paragraphs(text), start=1):
            patient_id = self._extract_patient_id(paragraph)
            if patient_id is None:
                continue
            rows.append(
                {
                    "ID": patient_id,
                    "section": section,
                    "text": paragraph,
                    "source_chunk_id": f"{table_hint}:{chunk_index}",
                }
            )
        return rows

    def _markdown_paragraphs(self, text: str) -> list[tuple[str, str]]:
        paragraphs: list[tuple[str, str]] = []
        current_section = ""
        current_lines: list[str] = []

        def flush() -> None:
            nonlocal current_lines
            paragraph = " ".join(line.strip() for line in current_lines if line.strip()).strip()
            current_lines = []
            if not paragraph:
                return
            if paragraph.startswith("#") or set(paragraph) <= {"-"}:
                return
            paragraphs.append((current_section, paragraph))

        for raw_line in text.splitlines():
            line = raw_line.strip()
            heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if heading_match:
                flush()
                current_section = heading_match.group(2).strip()
                continue
            if not line:
                flush()
                continue
            current_lines.append(line)
        flush()
        return paragraphs

    def _append_source_id(self, existing: Any, source_id: str) -> str:
        if not existing:
            return source_id
        parts = str(existing).split("|")
        if source_id in parts:
            return str(existing)
        return f"{existing}|{source_id}"

    def _extract_patient_ids(self, text: str) -> list[int]:
        patterns = [
            r"\bpatient(?:\s+(?:assigned|registered|associated|with|number|file|ID))*\s*(?:number|ID)?\s*(\d{4,9})\b",
            r"\bMedical Record Number\s+(\d{4,9})\b",
            r"\bfile number\s+(\d{4,9})\b",
            r"\bfile\s+(\d{4,9})\b",
        ]
        ids: list[int] = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                patient_id = int(match.group(1))
                if patient_id not in ids:
                    ids.append(patient_id)
        return ids

    def _extract_patient_id(self, text: str) -> int | None:
        ids = self._extract_patient_ids(text)
        return ids[0] if ids else None

    def _extract_sex(self, text: str) -> str | None:
        lowered = text.lower()
        if re.search(r"\b(female|she|her)\b", lowered):
            return "F"
        if re.search(r"\b(male|he|his)\b", lowered):
            return "M"
        return None

    def _extract_birthday(self, text: str) -> str | None:
        patterns = [
            r"(?:born|birthday|birthdate|date of birth)[^.;]*?(?:correct(?:ed)?(?: date)? (?:is|to)|confirmed(?: the correct date)? (?:is|to)|rectified to|to the correct date of)\s+([^.;]+)",
            r"(?:born|birthday|birthdate|date of birth)[^.;]*?\b(?:on|as|is|was)\s+([^.;]+)",
            r"birth year .*? corrected .*? to\s+([^.;]+)",
        ]
        return self._extract_date_by_patterns(text, patterns)

    def _extract_record_date(self, text: str) -> str | None:
        patterns = [
            r"(?:record|chart|file|data)[^.;]*?(?:correct(?:ed)?|confirmed|amended)[^.;]*?(?:as|to|is)\s+([^.;]+)",
            r"(?:record|chart|file)[^.;]*?(?:created|opened|initiated|established|formal creation|formally opened)[^.;]*?\b(?:on|as|is|was)\s+([^.;]+)",
            r"(?:data (?:was )?(?:recorded|entered)|first data recording)[^.;]*?\b(?:on|as|is|was)\s+([^.;]+)",
        ]
        return self._extract_date_by_patterns(text, patterns)

    def _extract_first_visit_date(self, text: str) -> str | None:
        patterns = [
            r"first (?:hospital )?visit[^.;]*?(?:correct(?:ed)?|confirmed|amended)[^.;]*?(?:as|to|is)\s+([^.;]+)",
            r"first (?:hospital )?visit[^.;]*?\b(?:on|occurred on|was on|recorded on)\s+([^.;]+)",
            r"first came to the hospital\s+on\s+([^.;]+)",
            r"first seen\s+on\s+([^.;]+)",
        ]
        return self._extract_date_by_patterns(text, patterns)

    def _extract_admission(self, text: str) -> str | None:
        lowered = text.lower()
        if "outpatient" in lowered or "followed in the outpatient clinic" in lowered:
            return "-"
        if "inpatient" in lowered or "admitted" in lowered or "required an inpatient stay" in lowered:
            return "+"
        return None

    def _extract_diagnosis(self, text: str) -> str | None:
        patterns = [
            r"diagnos(?:is|es) of\s+([A-Za-z0-9+/\- ,]+?)(?:\s+(?:required|necessitated|and|in|with)\b|[.;])",
            r"diagnosed with\s+([A-Za-z0-9+/\- ,]+?)(?:\s+(?:required|necessitated|and|in|with)\b|[.;])",
            r"managed for\s+([A-Za-z0-9+/\- ,]+?)(?:\s+(?:in|on|with)\b|[.;])",
            r"followed .*? for\s+([A-Za-z0-9+/\- ,]+?)(?:\s+(?:in|on|with)\b|[.;])",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                diagnosis = match.group(1).strip(" ,")
                return re.sub(r"\s+", " ", diagnosis)
        return None

    def _extract_lab_date(self, text: str) -> str | None:
        patterns = [
            r"(?:from|on|dated|date of|corresponding to(?: the)? sample from|records from|tested on|assessed on|evaluated on|collected on)\s+([^,;.]+(?:,\s*\d{4})?)",
        ]
        return self._extract_date_by_patterns(text, patterns)

    def _extract_lab_metrics(self, text: str) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        aliases = {
            "GOT": ["GOT", "glutamic oxaloacetic transaminase"],
            "GPT": ["GPT", "glutamic pyruvic transaminase"],
            "LDH": ["LDH", "lactate dehydrogenase"],
            "ALP": ["ALP", "alkaline phosphatase"],
            "T_BIL": ["T-BIL", "total bilirubin"],
            "TP": ["total protein", "TP"],
            "ALB": ["albumin", "ALB"],
            "UA": ["uric acid", "UA"],
            "UN": ["urea nitrogen", "UN"],
            "CRE": ["creatinine", "CRE"],
            "CPK": ["CPK"],
            "WBC": ["WBC", "white blood cell"],
            "RBC": ["RBC", "red blood cell"],
            "HGB": ["HGB", "hemoglobin"],
            "PLT": ["PLT", "platelet"],
            "APTT": ["APTT"],
            "PT": ["PT"],
            "IGG": ["IgG", "IGG"],
            "IGA": ["IgA", "IGA"],
            "IGM": ["IgM", "IGM"],
            "C3": ["C3"],
            "C4": ["C4"],
            "CRP": ["CRP"],
            "RF": ["RF"],
            "RA": ["RA"],
        }
        for column, column_aliases in aliases.items():
            value = self._extract_metric_value(text, column_aliases)
            if value is not None:
                metrics[column] = value
        cre_status = self._extract_cre_status(text, metrics.get("CRE"))
        if cre_status is not None:
            metrics["CRE_status"] = cre_status
        return metrics

    def _extract_metric_value(self, text: str, aliases: list[str]) -> float | str | None:
        sentences = re.split(r"(?<=[.;])\s+", text)
        for sentence in sentences:
            if not any(re.search(rf"\b{re.escape(alias)}\b", sentence, flags=re.IGNORECASE) for alias in aliases):
                continue
            if re.search(r"\b(?:NaN|None|not available|not recorded|unavailable|not measured|not performed)\b", sentence, flags=re.IGNORECASE):
                return None
            if re.search(r"\b(?:data voids?|data gaps?|complete absence|lacked .* data|no .* data)\b", sentence, flags=re.IGNORECASE):
                return None

            lowered = sentence.lower()
            numbers = [float(match.group(1)) for match in re.finditer(r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)", sentence)]
            if not numbers:
                qualitative = self._extract_qualitative_value(sentence, aliases)
                if qualitative is not None:
                    return qualitative
                continue

            if any(word in lowered for word in ["correct", "confirmed", "verified", "final", "revised", "adjusted", "rectified"]):
                return numbers[-1]

            alias_positions = [
                match.end()
                for alias in aliases
                for match in re.finditer(rf"\b{re.escape(alias)}\b", sentence, flags=re.IGNORECASE)
            ]
            if alias_positions:
                start = min(alias_positions)
                tail = sentence[start:]
                stop_positions = [
                    pos
                    for marker in self._metric_markers()
                    for pos in [tail.lower().find(marker)]
                    if pos > 0
                ]
                if stop_positions:
                    tail = tail[: min(stop_positions)]
                tail_numbers = [
                    float(match.group(1))
                    for match in re.finditer(r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)", tail)
                ]
                if tail_numbers:
                    return tail_numbers[0]
                continue
            return numbers[-1]
        return None

    def _metric_markers(self) -> list[str]:
        return [
            " got",
            " gpt",
            " ldh",
            " alp",
            " t-bil",
            " total bilirubin",
            " total protein",
            " albumin",
            " uric acid",
            " urea nitrogen",
            " creatinine",
            " cpk",
            " wbc",
            " rbc",
            " hgb",
            " plt",
            " aptt",
            " igg",
            " iga",
            " igm",
            " crp",
        ]

    def _extract_qualitative_value(self, sentence: str, aliases: list[str]) -> str | None:
        lowered = sentence.lower()
        if any(alias.lower() in lowered for alias in aliases):
            if "negative" in lowered or re.search(r"\b-\b", sentence):
                return "-"
            if "positive" in lowered or re.search(r"\b\+\b", sentence):
                return "+"
        return None

    def _extract_cre_status(self, text: str, cre_value: Any = None) -> str | None:
        try:
            numeric_cre = float(cre_value)
        except (TypeError, ValueError):
            numeric_cre = None
        if numeric_cre is not None:
            if numeric_cre > 1.2:
                return "abnormal"
            if numeric_cre == 1.2:
                return "borderline"
            return "normal"

        if not re.search(r"\b(?:creatinine|CRE|renal|kidney|glomerular)\b", text, flags=re.IGNORECASE):
            return None
        for sentence in re.split(r"(?<=[.;])\s+", text):
            if not re.search(r"\b(?:creatinine|CRE)\b", sentence, flags=re.IGNORECASE):
                continue
            lowered = sentence.lower()
            if "upper limit of the normal range" in lowered or "borderline renal" in lowered:
                return "borderline"
            if any(
                phrase in lowered
                for phrase in [
                    "creatinine was significantly elevated",
                    "creatinine was elevated",
                    "creatinine level was significantly elevated",
                    "creatinine level was elevated",
                    "indicating impaired renal filtration",
                    "confirming a significant reduction in renal clearance",
                ]
            ):
                return "abnormal"
            if "normal" in lowered or "healthy kidney function" in lowered or "within normal" in lowered:
                return "normal"
        return None

    def _has_lab_null_panel(self, text: str) -> bool:
        return bool(
            re.search(r"\b(?:UA|UN|CRE|GOT|GPT|LDH|ALP|T-BIL)\b", text)
            and re.search(r"\b(?:NaN|None|not available|unavailable|not recorded)\b", text, flags=re.IGNORECASE)
        )

    def _extract_date_by_patterns(self, text: str, patterns: list[str]) -> str | None:
        candidates = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                candidates.append(match.group(1))
        for candidate in reversed(candidates):
            parsed = self._parse_fuzzy_date(candidate)
            if parsed is not None:
                return parsed
        return None

    def _parse_fuzzy_date(self, text: str) -> str | None:
        month_map = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }
        cleaned = text.strip(" .;,")
        cleaned = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", cleaned, flags=re.IGNORECASE)
        lowered = cleaned.lower()

        iso_match = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", cleaned)
        if iso_match:
            return self._date_or_none(
                int(iso_match.group(1)),
                int(iso_match.group(2)),
                int(iso_match.group(3)),
            )

        match = re.search(
            r"\b("
            + "|".join(month_map)
            + r")\s+(\d{1,2}),?\s+(\d{4})\b",
            lowered,
        )
        if match:
            return self._date_or_none(int(match.group(3)), month_map[match.group(1)], int(match.group(2)))

        match = re.search(
            r"\b(\d{1,2})\s+(?:day of\s+)?("
            + "|".join(month_map)
            + r")(?:\s+in)?(?:\s+the\s+year\s+of)?\s+(\d{4})\b",
            lowered,
        )
        if match:
            return self._date_or_none(int(match.group(3)), month_map[match.group(2)], int(match.group(1)))

        match = re.search(
            r"\b(?:the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|twenty-first|twenty-second|twenty-third|twenty-fourth|twenty-fifth|twenty-sixth|twenty-seventh|twenty-eighth|twenty-ninth|thirtieth|thirty-first)\s+(?:day\s+)?of\s+("
            + "|".join(month_map)
            + r").*?(\d{4})\b",
            lowered,
        )
        if match:
            day = self._ordinal_word_to_int(match.group(1))
            if day is not None:
                return self._date_or_none(int(match.group(3)), month_map[match.group(2)], day)

        match = re.search(r"\b(late|mid|middle of|end of|near the end of|first week of|final week of)\s+(" + "|".join(month_map) + r").*?(\d{4})\b", lowered)
        if match:
            day = {
                "first week of": 4,
                "mid": 15,
                "middle of": 15,
                "late": 25,
                "end of": 28,
                "near the end of": 25,
                "final week of": 25,
            }.get(match.group(1), 15)
            return self._date_or_none(int(match.group(3)), month_map[match.group(2)], day)

        match = re.search(r"\b(\d{4})\b", lowered)
        if match and any(month in lowered for month in month_map):
            for month_name, month_number in month_map.items():
                if month_name in lowered:
                    return self._date_or_none(int(match.group(1)), month_number, 15)
        return None

    def _date_or_none(self, year: int, month: int, day: int) -> str | None:
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            return None

    def _ordinal_word_to_int(self, value: str) -> int | None:
        mapping = {
            "first": 1,
            "second": 2,
            "third": 3,
            "fourth": 4,
            "fifth": 5,
            "sixth": 6,
            "seventh": 7,
            "eighth": 8,
            "ninth": 9,
            "tenth": 10,
            "eleventh": 11,
            "twelfth": 12,
            "thirteenth": 13,
            "fourteenth": 14,
            "fifteenth": 15,
            "sixteenth": 16,
            "seventeenth": 17,
            "eighteenth": 18,
            "nineteenth": 19,
            "twentieth": 20,
            "twenty-first": 21,
            "twenty-second": 22,
            "twenty-third": 23,
            "twenty-fourth": 24,
            "twenty-fifth": 25,
            "twenty-sixth": 26,
            "twenty-seventh": 27,
            "twenty-eighth": 28,
            "twenty-ninth": 29,
            "thirtieth": 30,
            "thirty-first": 31,
        }
        return mapping.get(value.lower())

    def _detect_type(self, file_path):
        # get extension
        ext = file_path.split(".")[-1].lower()

        match ext:
            case "csv":
                return "csv"
            case "json":
                return "json"
            case "db":
                return "sqlite"
            case "sqlite":
                return "sqlite"
            case _:
                raise ValueError(f"Unsupported file type: {ext}")

    def _validate_load_file_size(self, file_path):
        file_size_bytes = os.path.getsize(file_path)
        if (
            self.max_load_file_bytes is not None
            and file_size_bytes > self.max_load_file_bytes
        ):
            size_mb = file_size_bytes / 1024 / 1024
            limit_mb = self.max_load_file_bytes / 1024 / 1024
            raise ValueError(
                f"File exceeds DataEngine load limit: {size_mb:.2f} MiB > "
                f"{limit_mb:.2f} MiB"
            )
        return file_size_bytes

    def _register_csv(self, table_name, file_path):
        try:
            sql = f"""
            CREATE OR REPLACE VIEW {table_name} AS
            SELECT * FROM read_csv_auto({self._quote_string(file_path)});
            """
            self.conn.execute(sql)
            self.tables.add(table_name)
            columns = self._get_columns(table_name)

            self.catalog[table_name] = {
                "source_type": "csv",
                "columns": columns,
                "_meta": {
                    "file_path": file_path,
                },
            }

        except Exception as e:
            print(f"[CSV REGISTER ERROR] {e}")
            raise

    def _register_json(self, table_name, file_path):
        try:
            wrapper_meta = self._detect_records_wrapper(file_path)
            read_options = self._json_read_options()
            if wrapper_meta is not None:
                sql = f"""
                CREATE OR REPLACE VIEW {table_name} AS
                SELECT r.*
                FROM read_json_auto(
                    {self._quote_string(file_path)}{read_options}
                ) AS src,
                     UNNEST(src.records) AS t(r);
                """
            else:
                sql = f"""
                CREATE OR REPLACE VIEW {table_name} AS
                SELECT * FROM read_json_auto(
                    {self._quote_string(file_path)}{read_options}
                );
                """
            self.conn.execute(sql)
            self.tables.add(table_name)
            columns = self._get_columns(table_name)

            self.catalog[table_name] = {
                "source_type": "json",
                "columns": columns,
                "_meta": {
                    "file_path": file_path,
                    **(wrapper_meta or {}),
                },
            }

        except Exception as e:
            print(f"[JSON REGISTER ERROR] {e}")
            raise

    def _json_read_options(self):
        if self.max_load_file_bytes is None:
            return ""
        return f", maximum_object_size={int(self.max_load_file_bytes)}"

    def _register_sqlite(self, file_path):
        try:
            if not self.sqlite_load:
                try:
                    self.conn.execute("LOAD sqlite;")
                except Exception:
                    self.conn.execute("INSTALL sqlite;")
                    self.conn.execute("LOAD sqlite;")
                self.sqlite_load = True

            alias_seed = hashlib.md5(str(file_path).encode()).hexdigest()[:6]
            alias_base = self._sanitize_identifier(os.path.splitext(os.path.basename(file_path))[0])
            alias = f"db_{alias_base}_{alias_seed}"

            self.conn.execute(
                f"""
                ATTACH IF NOT EXISTS DATABASE {self._quote_string(file_path)} AS {alias} (TYPE SQLITE);
                """
            )

            dbtables = self.conn.execute(
                f"""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_catalog = {self._quote_string(alias)}
                  AND table_schema = 'main'
                ORDER BY table_name;
                """
            ).fetchall()

            for (source_table_name_raw,) in dbtables:
                source_table_name = self._sanitize_identifier(str(source_table_name_raw))
                table_name = self._reserve_table_name(source_table_name)
                sql = f"""
                CREATE OR REPLACE VIEW {table_name} AS
                SELECT * FROM {alias}.main.{self._quote_identifier(str(source_table_name_raw))};
                """
                self.conn.execute(sql)
                self.tables.add(table_name)
                columns = self._get_columns(table_name)
                self.catalog[table_name] = {
                    "source_type": "sqlite",
                    "columns": columns,
                    "_meta": {
                        "file_path": file_path,
                        "source_table_name": str(source_table_name_raw),
                    },
                }

        except Exception as e:
            print(f"[SQLITE REGISTER ERROR] {e}")
            raise

    def _make_table_name(self, file_path):
        base = os.path.splitext(os.path.basename(file_path))[0]
        return self._sanitize_identifier(base)

    def _detect_records_wrapper(self, file_path: str) -> dict[str, Any] | None:
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return None

        if not isinstance(payload, dict):
            return None

        records = payload.get("records")
        if not isinstance(records, list) or not records:
            return None
        if not all(isinstance(item, dict) for item in records):
            return None

        source_table_name = payload.get("table")
        return {
            "json_wrapper": "records",
            "source_table_name": str(source_table_name) if source_table_name is not None else None,
        }

    def _get_columns(self, table_name: str):
        try:
            result = self.conn.execute(f"DESCRIBE {table_name}").fetchall()

            columns = [row[0] for row in result]
            return columns

        except Exception as e:
            print(f"[SCHEMA ERROR] {e}")
            return []

    def _sample_rows(self, table_name: str, sample_rows: int) -> list[dict[str, Any]]:
        if sample_rows <= 0:
            return []
        try:
            df = self.conn.execute(f"SELECT * FROM {table_name} LIMIT {sample_rows}").fetchdf()
            rows = []
            for row in df.to_dict(orient="records"):
                rows.append({key: self._json_safe_value(value) for key, value in row.items()})
            return rows
        except Exception as e:
            print(f"[SAMPLE ERROR] {e}")
            return []

    def _validate_read_only_sql(self, sql: str) -> None:
        tree = sqlglot.parse_one(sql, read="duckdb")
        if not isinstance(tree, (exp.Select, exp.Union, exp.Except, exp.Intersect)):
            raise ValueError("Only read-only SELECT queries are allowed.")

    def _apply_limit(self, sql: str, limit: int) -> str:
        normalized_sql = sql.strip().rstrip(";")
        safe_limit = max(int(limit), 0)
        return f"SELECT * FROM ({normalized_sql}) AS _dataengine_limited LIMIT {safe_limit}"

    def _sanitize_identifier(self, raw_name: str) -> str:
        normalized = re.sub(r"\W+", "_", raw_name).strip("_").lower()
        if not normalized:
            normalized = "table"
        if normalized[0].isdigit():
            normalized = f"t_{normalized}"
        return normalized

    def _reserve_table_name(self, base_name: str) -> str:
        if base_name not in self.catalog:
            return base_name

        suffix = 2
        while f"{base_name}_{suffix}" in self.catalog:
            suffix += 1
        return f"{base_name}_{suffix}"

    def _quote_string(self, value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    def _quote_identifier(self, value: str) -> str:
        return '"' + value.replace('"', '""') + '"'

    def _json_safe_value(self, value: Any) -> Any:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        if isinstance(value, datetime):
            if value.time() == time(0,0):
                return value.date().isoformat()

        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        if isinstance(value, float) and value != value:
            return None
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                pass
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def _tables_for_source_path(self, path: Path) -> list[str]:
        resolved_path = str(path.resolve())
        tables = []
        for table_name, entry in self.catalog.items():
            if entry.get("_meta", {}).get("file_path") == resolved_path:
                tables.append(table_name)
        return sorted(tables)
