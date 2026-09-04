import pandas as pd
from typing import Any, Dict, List


def _quote_col(col: str) -> str:
    return f'"{col}"' if " " in col or "-" in col else col


def generate_auto_dashboard(df: pd.DataFrame, dataset_info: Any) -> Dict[str, Any]:
    """
    Heuristically generate dashboard visuals from dataset schema.
    Uses column stats — no LLM required.
    """
    from .cleaning_service import get_column_stats

    col_stats = get_column_stats(df)
    numeric = [s for s in col_stats if s.get("mean") is not None]
    text_cols = [
        s for s in col_stats
        if s.get("mean") is None and 1 < s.get("unique_values", 0) <= 20
    ]

    date_cols: List[str] = []
    for col in df.columns:
        if "date" in col.lower() or pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
        elif df[col].dtype == "object":
            try:
                parsed = pd.to_datetime(df[col].dropna().head(20), errors="coerce")
                if parsed.notna().sum() > 15:
                    date_cols.append(col)
            except Exception:
                pass

    items: List[Dict[str, Any]] = []
    y_offset = 0

    # ── Row 1: KPI cards (top 3 numeric columns by mean) ──────────────────
    top_numeric = sorted(numeric, key=lambda x: x.get("mean", 0), reverse=True)[:3]
    for i, stat in enumerate(top_numeric):
        col = stat["name"]
        q = _quote_col(col)
        items.append({
            "title": f"Total {col}",
            "sql_query": f"SELECT ROUND(SUM({q}), 2) AS total_value FROM dataset",
            "chart_config": {"type": "kpi", "labelCol": "total_value", "valueCol": "total_value"},
            "layout": {"w": 3, "h": 2, "x": i * 3, "y": y_offset},
        })
    y_offset += 2

    # ── Row 2: Time series (if date + numeric columns exist) ──────────────
    if date_cols and numeric:
        date_col = date_cols[0]
        val_col = top_numeric[0]["name"] if top_numeric else numeric[0]["name"]
        dq, vq = _quote_col(date_col), _quote_col(val_col)
        items.append({
            "title": f"{val_col} Over Time",
            "sql_query": (
                f"SELECT {dq} AS labelCol, ROUND(SUM({vq}), 2) AS valueCol "
                f"FROM dataset GROUP BY {dq} ORDER BY {dq} LIMIT 50"
            ),
            "chart_config": {"type": "line", "labelCol": "labelCol", "valueCol": "valueCol"},
            "layout": {"w": 6, "h": 4, "x": 0, "y": y_offset},
        })

    # ── Row 2 (right): Category breakdown ─────────────────────────────────
    if text_cols and numeric:
        cat_col = text_cols[0]["name"]
        val_col = top_numeric[0]["name"] if top_numeric else numeric[0]["name"]
        cq, vq = _quote_col(cat_col), _quote_col(val_col)
        x_pos = 6 if date_cols else 0
        items.append({
            "title": f"{val_col} by {cat_col}",
            "sql_query": (
                f"SELECT {cq} AS labelCol, ROUND(SUM({vq}), 2) AS valueCol "
                f"FROM dataset GROUP BY {cq} ORDER BY valueCol DESC LIMIT 12"
            ),
            "chart_config": {"type": "bar", "labelCol": "labelCol", "valueCol": "valueCol"},
            "layout": {"w": 6, "h": 4, "x": x_pos, "y": y_offset},
        })
        y_offset += 4

    # ── Row 3: Distribution pie (if small-cardinality category) ───────────
    small_cats = [s for s in text_cols if s.get("unique_values", 0) <= 8]
    if small_cats and numeric:
        cat_col = small_cats[0]["name"]
        val_col = top_numeric[0]["name"] if top_numeric else numeric[0]["name"]
        cq, vq = _quote_col(cat_col), _quote_col(val_col)
        items.append({
            "title": f"{cat_col} Share of {val_col}",
            "sql_query": (
                f"SELECT {cq} AS labelCol, ROUND(SUM({vq}), 2) AS valueCol "
                f"FROM dataset GROUP BY {cq} ORDER BY valueCol DESC LIMIT 8"
            ),
            "chart_config": {"type": "doughnut", "labelCol": "labelCol", "valueCol": "valueCol"},
            "layout": {"w": 6, "h": 4, "x": 0, "y": y_offset},
        })

    filename = getattr(dataset_info, "filename", "dataset")
    summary = (
        f"Auto-generated {len(items)} visuals for '{filename}' — "
        f"{len(top_numeric)} KPIs, "
        f"{'time-series + ' if date_cols else ''}"
        f"{'category breakdown' if text_cols else 'summary metrics'}."
    )

    return {"items": items, "summary": summary}
