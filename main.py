from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import pickle
import re
import tempfile
from typing import Annotated, Any, Literal, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field

from mistralai import agents

DRIVE_CONNECTOR_NAME = "google_drive_mcp"

_DATA_DIR = os.path.join(tempfile.gettempdir(), "data_analyst_datasets")

_DRIVE_URL_ID = re.compile(r"/d/([A-Za-z0-9_-]{10,})|[?&]id=([A-Za-z0-9_-]{10,})")
_PLAIN_ID = re.compile(r"^[A-Za-z0-9_-]{10,}$")

_DATASETS: dict[str, pd.DataFrame] = {}
_TITLES: dict[str, str] = {}

NUMERIC_AGGS = ("count", "nunique", "sum", "mean", "median", "min", "max", "std", "var")

FILTER_OPS = (
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "contains",
    "startswith",
    "endswith",
    "in",
    "not_in",
    "between",
    "is_null",
    "not_null",
)


class Filter(BaseModel):
    column: str = Field(description="Column name to filter on, exactly as returned by describe_columns.")
    op: Literal[
        "eq",
        "ne",
        "gt",
        "gte",
        "lt",
        "lte",
        "contains",
        "startswith",
        "endswith",
        "in",
        "not_in",
        "between",
        "is_null",
        "not_null",
    ] = Field(description="Comparison operator.")
    value: Optional[Union[str, int, float, bool]] = Field(
        default=None,
        description="Value to compare with. For 'in'/'not_in' pass a JSON-encoded list, for 'between' pass 'low;high'.",
    )


class Metric(BaseModel):
    column: str = Field(description="Column to aggregate. Use '*' with agg 'count' to count rows.")
    agg: Literal[
        "count",
        "nunique",
        "sum",
        "mean",
        "median",
        "min",
        "max",
        "std",
        "var",
    ] = Field(description="Aggregation function.")


def _ensure_dir() -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)


def _persist(dataset_id: str, df: pd.DataFrame, title: str) -> None:
    _ensure_dir()
    with open(os.path.join(_DATA_DIR, f"{dataset_id}.pkl"), "wb") as fh:
        pickle.dump({"df": df, "title": title}, fh)


def _load(dataset_id: str) -> tuple[str, pd.DataFrame]:
    dataset_id = dataset_id.strip()
    if not dataset_id and len(_DATASETS) == 1:
        dataset_id = next(iter(_DATASETS))
    if dataset_id in _DATASETS:
        return _TITLES.get(dataset_id, dataset_id), _DATASETS[dataset_id]
    path = os.path.join(_DATA_DIR, f"{dataset_id}.pkl")
    if dataset_id and os.path.exists(path):
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        _DATASETS[dataset_id] = blob["df"]
        _TITLES[dataset_id] = blob["title"]
        return blob["title"], blob["df"]
    if not dataset_id:
        raise agents.ToolError(
            "No dataset is loaded yet. Ask the user for the Google Drive link and call load_csv_from_drive."
        )
    raise agents.ToolError(
        f"Dataset '{dataset_id}' is not loaded in this session. Call load_csv_from_drive again."
    )


def _resolve_dataset(dataset_id: str) -> tuple[str, str, pd.DataFrame]:
    title, df = _load(dataset_id)
    resolved = dataset_id.strip()
    if not resolved:
        resolved = next(iter(_DATASETS))
    return resolved, title, df


def _extract_file_id(source: str) -> str:
    source = source.strip()
    match = _DRIVE_URL_ID.search(source)
    if match:
        return match.group(1) or match.group(2)
    if _PLAIN_ID.match(source):
        return source
    raise agents.ToolError(
        "Could not extract a Google Drive file id from that source. "
        "Ask the user for the file's share link (a URL containing /d/<id> or ?id=<id>)."
    )


def _csv_text_from_connector_result(result: Any) -> str:
    dumped = result.model_dump() if hasattr(result, "model_dump") else result
    blocks = dumped.get("content") or []
    for block in blocks:
        kind = block.get("type")
        if kind == "text":
            text = block.get("text") or ""
            try:
                parsed = json.loads(text)
            except (ValueError, TypeError):
                return text
            if isinstance(parsed, dict):
                payload = parsed.get("content") or parsed.get("text_content")
                if isinstance(payload, str):
                    return _decode(payload)
                return text
            return text
        if kind == "resource":
            resource = block.get("resource") or {}
            blob = resource.get("blob")
            if isinstance(blob, str):
                return _decode(blob)
            if isinstance(resource.get("text"), str):
                return resource["text"]
    raise agents.ToolError(
        "The Google Drive connector did not return the file content. "
        "Check that the file is a CSV (or a Google Sheet) and that the user has access to it."
    )


def _decode(payload: str) -> str:
    try:
        return base64.b64decode(payload, validate=True).decode("utf-8-sig", errors="replace")
    except Exception:
        return payload


def _parse_csv(text: str) -> pd.DataFrame:
    try:
        return pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception as error:
        raise agents.ToolError(f"Could not parse the file as CSV: {error}") from error


def _register(dataset_id: str, title: str, df: pd.DataFrame) -> None:
    _DATASETS[dataset_id] = df
    _TITLES[dataset_id] = title
    _persist(dataset_id, df, title)


def _column_type(dtype: pd.dtype) -> str:
    if pd.api.types.is_numeric_dtype(dtype):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    return "text"


def _coerce(series: pd.Series, value: Any) -> Any:
    if value is None:
        return None
    if pd.api.types.is_numeric_dtype(series):
        try:
            return float(value) if not pd.api.types.is_integer_dtype(series) else int(float(value))
        except (TypeError, ValueError):
            raise agents.ToolError(f"Value '{value}' is not valid for numeric column.")
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(value, errors="coerce")
    return value


def _apply_filters(df: pd.DataFrame, filters: list[Filter]) -> pd.DataFrame:
    for f in filters:
        if f.column not in df.columns:
            raise agents.ToolError(
                f"Unknown column '{f.column}'. Use describe_columns to list the columns."
            )
        series = df[f.column]
        if f.op == "is_null":
            mask = series.isna()
        elif f.op == "not_null":
            mask = series.notna()
        elif f.op == "in" or f.op == "not_in":
            try:
                values = json.loads(f.value) if isinstance(f.value, str) else [f.value]
            except (ValueError, TypeError):
                raise agents.ToolError("For 'in'/'not_in', pass value as a JSON-encoded list, e.g. '[\"a\",\"b\"]'.")
            coerced = [_coerce(series, v) for v in values]
            mask = series.isin(coerced)
            if f.op == "not_in":
                mask = ~mask
        elif f.op == "between":
            if not isinstance(f.value, str) or ";" not in f.value:
                raise agents.ToolError("For 'between', pass value as 'low;high'.")
            low, high = (part.strip() for part in f.value.split(";", 1))
            mask = (series >= _coerce(series, low)) & (series <= _coerce(series, high))
        else:
            value = _coerce(series, f.value)
            if f.op == "eq":
                mask = series == value
            elif f.op == "ne":
                mask = series != value
            elif f.op == "gt":
                mask = series > value
            elif f.op == "gte":
                mask = series >= value
            elif f.op == "lt":
                mask = series < value
            elif f.op == "lte":
                mask = series <= value
            elif f.op == "contains":
                mask = series.astype(str).str.contains(str(f.value), case=False, na=False)
            elif f.op == "startswith":
                mask = series.astype(str).str.startswith(str(f.value), na=False)
            else:
                mask = series.astype(str).str.endswith(str(f.value), na=False)
        df = df[mask]
    return df


def _records(df: pd.DataFrame, columns: Optional[list[str]], limit: int) -> list[dict]:
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise agents.ToolError(f"Unknown columns: {missing}. Use describe_columns.")
        df = df[columns]
    return df.head(max(1, min(limit, 200))).replace({pd.NA: None}).to_dict(orient="records")


@agents.tool(model_access="direct")
async def load_csv_from_drive(
    source: Annotated[str, Field(description="Google Drive share link (URL) or file id of the CSV or Google Sheet.")],
) -> dict:
    """Load a CSV file from the user's Google Drive into this session and return its structure.

    Downloads the file through the user's Google Drive connection, parses it as CSV
    and registers it under a dataset_id. Call describe_columns next for details.
    """
    from mistralai.workflows.client import get_mistral_client
    from mistralai.workflows.core.auth import RunAs

    file_id = _extract_file_id(source)
    client = get_mistral_client(run_as=RunAs.EXECUTOR)
    try:
        result = await client.beta.connectors.call_tool_async(
            connector_id_or_name=DRIVE_CONNECTOR_NAME,
            tool_name="download_file_content",
            arguments={"file_id": file_id, "export_mime_type": "text/csv"},
        )
    except Exception as error:
        raise agents.ToolError(
            f"Google Drive download failed ({type(error).__name__}). Check the link and the Drive connection."
        ) from error

    dumped = result.model_dump() if hasattr(result, "model_dump") else {}
    title = file_id
    for block in dumped.get("content") or []:
        if block.get("type") == "text":
            try:
                parsed = json.loads(block.get("text") or "")
                if isinstance(parsed, dict) and parsed.get("title"):
                    title = parsed["title"]
            except (ValueError, TypeError):
                pass
            break

    text = _csv_text_from_connector_result(result)
    df = _parse_csv(text)
    _register(file_id, title, df)
    return {
        "dataset_id": file_id,
        "title": title,
        "rows": int(len(df)),
        "columns": [
            {
                "name": name,
                "type": _column_type(dtype),
                "non_null": int(df[name].notna().sum()),
                "missing": int(df[name].isna().sum()),
            }
            for name, dtype in df.dtypes.items()
        ],
        "sample_rows": _records(df, None, 5),
    }


@agents.tool(model_access="direct")
def describe_columns(
    dataset_id: Annotated[str, Field(description="Dataset id returned by load_csv_from_drive. Leave empty when only one dataset is loaded.")],
) -> dict:
    """Describe every column of the loaded dataset: type, missing values, uniques, ranges and top values."""
    dataset_id, title, df = _resolve_dataset(dataset_id)
    columns = []
    for name in df.columns:
        series = df[name]
        info: dict[str, Any] = {
            "name": name,
            "type": _column_type(series.dtype),
            "non_null": int(series.notna().sum()),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            info.update(
                {
                    "min": None if series.dropna().empty else float(series.min()),
                    "max": None if series.dropna().empty else float(series.max()),
                    "mean": None if series.dropna().empty else float(series.mean()),
                    "median": None if series.dropna().empty else float(series.median()),
                }
            )
        else:
            top = series.value_counts(dropna=True).head(5)
            info["top_values"] = [{"value": str(v), "count": int(c)} for v, c in top.items()]
        columns.append(info)
    return {"dataset_id": dataset_id, "title": title, "rows": int(len(df)), "columns": columns}


@agents.tool(model_access="direct")
def filter_rows(
    dataset_id: Annotated[str, Field(description="Dataset id. Leave empty when only one dataset is loaded.")],
    filters: Annotated[list[Filter], Field(description="Filters combined with AND.")],
    columns: Annotated[Optional[list[str]], Field(description="Optional subset of columns to return.")] = None,
    limit: Annotated[int, Field(description="Max rows to return (1-200).")] = 20,
) -> dict:
    """Filter rows of the loaded dataset with exact conditions and return the matching rows.

    Use this to look at concrete records. For counts and sums over groups use aggregate instead.
    """
    dataset_id, title, df = _resolve_dataset(dataset_id)
    matched = _apply_filters(df, filters)
    return {
        "dataset_id": dataset_id,
        "title": title,
        "total_rows": int(len(df)),
        "matched_rows": int(len(matched)),
        "returned": min(len(matched), max(1, min(limit, 200))),
        "rows": _records(matched, columns, limit),
    }


@agents.tool(model_access="direct")
def aggregate(
    dataset_id: Annotated[str, Field(description="Dataset id. Leave empty when only one dataset is loaded.")],
    group_by: Annotated[list[str], Field(description="Columns to group by (one or more).")],
    metrics: Annotated[list[Metric], Field(description="Aggregations to compute per group, e.g. [{'column':'revenue','agg':'sum'}].")],
    filters: Annotated[Optional[list[Filter]], Field(description="Optional filters applied before grouping.")] = None,
    order_by: Annotated[Optional[str], Field(description="Metric name to order by, e.g. 'sum_revenue'.")] = None,
    descending: Annotated[bool, Field(description="Order largest first.")] = True,
    limit: Annotated[int, Field(description="Max groups to return (1-200).")] = 50,
) -> dict:
    """Group the loaded dataset and compute aggregates (count, sum, mean, median, min, max, nunique, std, var) per group.

    This is the tool for questions like 'total by category', 'average per month' or 'top 10 by revenue'.
    """
    dataset_id, title, df = _resolve_dataset(dataset_id)
    missing = [c for c in group_by if c not in df.columns]
    if missing:
        raise agents.ToolError(f"Unknown group_by columns: {missing}. Use describe_columns.")
    if not metrics:
        raise agents.ToolError("Pass at least one metric, e.g. [{'column': '*', 'agg': 'count'}].")

    source = _apply_filters(df, filters) if filters else df
    if source.empty:
        return {
            "dataset_id": dataset_id,
            "title": title,
            "rows_before_grouping": 0,
            "groups": [],
            "note": "No rows matched the filters.",
        }

    grouped = source.groupby(list(group_by), dropna=False, observed=True)
    parts: dict[str, pd.Series] = {}
    for metric in metrics:
        label = f"{metric.agg}_{metric.column}".replace("*", "rows")
        if metric.agg == "count" and metric.column == "*":
            parts[label] = grouped.size()
        else:
            if metric.column != "*" and metric.column not in source.columns:
                raise agents.ToolError(f"Unknown metric column '{metric.column}'. Use describe_columns.")
            target = metric.column if metric.column != "*" else None
            column = grouped[target] if target else grouped
            try:
                parts[label] = getattr(column, metric.agg)()
            except Exception as error:
                raise agents.ToolError(f"Cannot compute {metric.agg} on '{metric.column}': {error}") from error

    result = pd.DataFrame(parts).reset_index()
    if order_by:
        if order_by not in result.columns:
            raise agents.ToolError(
                f"Unknown order_by '{order_by}'. Available metric names: {[c for c in result.columns if c not in group_by]}"
            )
        result = result.sort_values(order_by, ascending=not descending)
    result = result.head(max(1, min(limit, 200)))
    return {
        "dataset_id": dataset_id,
        "title": title,
        "rows_before_grouping": int(len(source)),
        "groups_returned": int(len(result)),
        "groups": result.replace({pd.NA: None}).to_dict(orient="records"),
    }


@agents.tool(model_access="direct")
def preview_rows(
    dataset_id: Annotated[str, Field(description="Dataset id. Leave empty when only one dataset is loaded.")],
    start: Annotated[int, Field(description="0-based first row to show.")] = 0,
    limit: Annotated[int, Field(description="Max rows to return (1-200).")] = 10,
    columns: Annotated[Optional[list[str]], Field(description="Optional subset of columns.")] = None,
) -> dict:
    """Show a slice of raw rows of the loaded dataset, from 'start' onwards."""
    dataset_id, title, df = _resolve_dataset(dataset_id)
    window = df.iloc[max(0, start) : max(0, start) + max(1, min(limit, 200))]
    return {
        "dataset_id": dataset_id,
        "title": title,
        "total_rows": int(len(df)),
        "start": max(0, start),
        "rows": _records(window, columns, limit),
    }


@agents.tool(model_access="direct")
def list_loaded_datasets() -> dict:
    """List the datasets already loaded in this session, so the right one is used in follow-up questions."""
    if not _DATASETS:
        return {"datasets": [], "note": "No dataset loaded yet. Use load_csv_from_drive."}
    return {
        "datasets": [
            {"dataset_id": key, "title": _TITLES.get(key, key), "rows": int(len(df)), "columns": list(df.columns)}
            for key, df in _DATASETS.items()
        ]
    }


agent = agents.Agent(
    model="mistral-medium-latest",
    name="data-analyst",
    description="Conversational data analyst: answers questions in plain language about a CSV file from the user's Google Drive.",
    instructions=(
        "You are a conversational data analyst. The user gives you a link to a CSV file in their "
        "Google Drive and asks questions in plain language. You answer with real numbers computed "
        "from their data.\n"
        "\n"
        "How you work:\n"
        "- When the user gives a Google Drive link (or file id), call load_csv_from_drive with it. "
        "The tool extracts the file id itself, downloads the file through the user's Google Drive "
        "connection and registers it as the session's dataset.\n"
        "- If the user names a file instead of giving a link, use the google_drive_mcp connector's "
        "search_files tool to find its id, then load it with load_csv_from_drive.\n"
        "- After loading, briefly present the dataset: rows, columns and their types "
        "(from the load result or describe_columns).\n"
        "- Answer every data question ONLY with the fixed analysis tools: describe_columns, "
        "filter_rows, aggregate, preview_rows, list_loaded_datasets. Never invent, estimate or "
        "extrapolate numbers. If the tools cannot answer a question, say so.\n"
        "- If exactly one dataset is loaded, omit dataset_id in the other tools. If several are "
        "loaded and the user does not say which one, call list_loaded_datasets and ask.\n"
        "- For follow-up questions, reuse the already loaded dataset: do not reload the file. "
        "Remember everything discussed earlier in the session and build on it.\n"
        "- Prefer aggregate over filter_rows for 'how many', 'total', 'average', 'top N' questions. "
        "Use filter_rows or preview_rows to show concrete records.\n"
        "- Round long decimals sensibly (e.g. 2 decimal places) and state units. When a result "
        "would be huge, summarize it instead of dumping every row.\n"
        "- If a tool fails, explain the problem in plain language and suggest a fix (for example "
        "re-sharing the file or checking the Drive connection).\n"
        "- Reply in the same language the user writes in.\n"
        "\n"
        "You have no shell, no file system and no code execution: the fixed tools are the only way "
        "to touch the data. Never claim to run code."
    ),
    tools=[
        load_csv_from_drive,
        describe_columns,
        filter_rows,
        aggregate,
        preview_rows,
        list_loaded_datasets,
    ],
    connectors=[agents.connector("google_drive_mcp")],
)


async def main() -> None:
    state = await agent.run("Olá! Como você funciona?")
    print(state.text)


if __name__ == "__main__":
    asyncio.run(main())
