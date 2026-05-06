import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime, timezone
from .cleaning_service import get_column_stats
from .forecasting_service import forecast_sales


def generate_report(df: pd.DataFrame, dataset_info: Any) -> Dict[str, Any]:
    """
    Generate a comprehensive intelligence report for a dataset.
    Returns a structured dict that the frontend can render.
    """
    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "filename": getattr(dataset_info, "filename", "Unknown"),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "file_type": getattr(dataset_info, "file_type", "csv"),
        },
        "executive_summary": {},
        "column_insights": [],
        "data_quality": {},
        "top_insights": [],
    }

    # ── Column Stats ──────────────────────────────────────────────────────────
    col_stats = get_column_stats(df)
    report["column_insights"] = col_stats

    numeric_stats = [s for s in col_stats if s.get("mean") is not None]
    text_cols = [s for s in col_stats if s.get("mean") is None]

    # ── Data Quality ─────────────────────────────────────────────────────────
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isnull().sum().sum())
    completeness = round((1 - missing_cells / total_cells) * 100, 2) if total_cells else 100

    duplicate_rows = int(df.duplicated().sum())

    report["data_quality"] = {
        "completeness_pct": completeness,
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "quality_score": _quality_score(completeness, duplicate_rows, df.shape[0]),
    }

    # ── Executive Summary ────────────────────────────────────────────────────
    report["executive_summary"] = {
        "total_numeric_columns": len(numeric_stats),
        "total_categorical_columns": len(text_cols),
        "kpis": [
            {
                "name": s["name"],
                "mean": round(s["mean"], 2),
                "median": round(s.get("median", 0), 2),
                "min": round(s.get("min", 0), 2),
                "max": round(s.get("max", 0), 2),
                "std": round(s.get("std", 0), 2),
            }
            for s in numeric_stats[:6]  # Top 6 numeric columns
        ],
    }

    # ── Top Insights ─────────────────────────────────────────────────────────
    insights = []

    # insight 1: highest-value column
    if numeric_stats:
        top_col = max(numeric_stats, key=lambda s: s.get("mean", 0))
        insights.append({
            "type": "kpi",
            "icon": "📊",
            "title": f"Highest Average: {top_col['name']}",
            "body": f"The column '{top_col['name']}' has the highest average value of {top_col['mean']:.2f}, ranging from {top_col.get('min', 0):.2f} to {top_col.get('max', 0):.2f}.",
        })

    # insight 2: most missing data column
    most_missing = max(col_stats, key=lambda s: s.get("missing_values", 0), default=None)
    if most_missing and most_missing.get("missing_values", 0) > 0:
        insights.append({
            "type": "quality",
            "icon": "⚠️",
            "title": f"Data Gap: {most_missing['name']}",
            "body": f"'{most_missing['name']}' has {most_missing.get('missing_values', 0)} missing values. Consider imputing or investigating this column.",
        })

    # insight 3: high cardinality column
    high_card = max(col_stats, key=lambda s: s.get("unique_values", 0), default=None)
    if high_card:
        insights.append({
            "type": "distribution",
            "icon": "🔢",
            "title": f"Most Unique Values: {high_card['name']}",
            "body": f"'{high_card['name']}' has {high_card.get('unique_values', 0)} unique values — useful as a primary identifier or segmentation key.",
        })

    # insight 4: data quality
    insights.append({
        "type": "quality",
        "icon": "✅" if completeness >= 95 else "🔶",
        "title": f"Data Completeness: {completeness}%",
        "body": f"Your dataset is {'highly complete' if completeness >= 95 else 'moderately complete'}. {missing_cells} cells have missing values across {df.shape[1]} columns.",
    })

    # insight 5: skewed distributions warning
    skewed = [s for s in numeric_stats if abs(s.get('skewness', 0)) > 1.5]
    if skewed:
        worst = max(skewed, key=lambda s: abs(s.get('skewness', 0)))
        direction = 'right-skewed (long tail of high values)' if worst.get('skewness', 0) > 0 else 'left-skewed (long tail of low values)'
        insights.append({
            "type": "distribution",
            "icon": "📐",
            "title": f"Skewed Distribution: {worst['name']}",
            "body": f"'{worst['name']}' is heavily {direction} (skewness: {worst.get('skewness', 0):.2f}). The mean ({worst['mean']:.2f}) may be misleading — prefer the median ({worst.get('median', 0):.2f}) for this column.",
        })

    # ── Correlation Matrix ────────────────────────────────────────────────────
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_data = None
    if len(numeric_df.columns) >= 2:
        corr_matrix = numeric_df.corr().round(2)
        correlation_data = {
            "columns": corr_matrix.columns.tolist(),
            "matrix": corr_matrix.replace({np.nan: None}).values.tolist(),
        }
        # insight 6: strongest correlation
        import itertools
        pairs = []
        for c1, c2 in itertools.combinations(corr_matrix.columns, 2):
            val = corr_matrix.loc[c1, c2]
            if pd.notna(val):
                pairs.append((c1, c2, val))
        if pairs:
            strongest = max(pairs, key=lambda p: abs(p[2]))
            if abs(strongest[2]) > 0.5:
                strength = 'strong' if abs(strongest[2]) > 0.75 else 'moderate'
                direction = 'positive' if strongest[2] > 0 else 'negative'
                insights.append({
                    "type": "correlation",
                    "icon": "🔗",
                    "title": f"{strength.title()} Correlation Found",
                    "body": f"'{strongest[0]}' and '{strongest[1]}' have a {strength} {direction} correlation ({strongest[2]:.2f}). Changes in one metric tend to {'move together with' if strongest[2] > 0 else 'inversely affect'} the other.",
                })

    report["correlation"] = correlation_data
    report["top_insights"] = insights
    return report


def _quality_score(completeness: float, duplicates: int, total_rows: int) -> str:
    dup_pct = (duplicates / total_rows * 100) if total_rows else 0
    score = completeness - dup_pct
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Good"
    elif score >= 50:
        return "Fair"
    else:
        return "Poor"
