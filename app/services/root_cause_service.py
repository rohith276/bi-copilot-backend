import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import timedelta


def _quote_col(col: str) -> str:
    return f'"{col}"' if " " in col or "-" in col else col


def _detect_date_col(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if "date" in col.lower() or pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        if df[col].dtype == "object":
            try:
                parsed = pd.to_datetime(df[col].dropna().head(20), errors="coerce")
                if parsed.notna().sum() > 15:
                    return col
            except Exception:
                pass
    return None


def _detect_metric_col(df: pd.DataFrame, metric_col: Optional[str]) -> Optional[str]:
    if metric_col and metric_col in df.columns:
        return metric_col
    numeric = df.select_dtypes(include="number").columns.tolist()
    if not numeric:
        return None
    for hint in ("sales", "revenue", "amount", "value", "total", "profit"):
        for col in numeric:
            if hint in col.lower():
                return col
    return numeric[0]


def _get_dimension_cols(df: pd.DataFrame, metric_col: str) -> List[str]:
    from .cleaning_service import get_column_stats

    raw_stats = get_column_stats(df)
    if not isinstance(raw_stats, list):
        return []
    dims: List[str] = []
    for s in raw_stats:
        if not isinstance(s, dict):
            continue
        name = str(s.get("name", ""))
        if not name or name == metric_col:
            continue
        uv = s.get("unique_values")
        unique_vals = int(uv) if isinstance(uv, (int, float)) else 0
        if s.get("bi_type") == "dimension" or (
            s.get("mean") is None and 1 < unique_vals <= 35
        ):
            dims.append(name)
    return dims[:8]


def analyze_root_cause(
    df: pd.DataFrame,
    metric_col: Optional[str] = None,
    question: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Mathematical Additive Root-Cause Engine:
    - Equal-window or median split with exact dimensional conservation: sum(delta_k) == total_delta.
    - Exact Volume vs. Ticket Price (Rate) Decomposition: delta == volume_effect + rate_effect.
    - Segment ranking across categorical dimensions with narrative synthesis.
    """
    working = df.copy()
    resolved_metric = _detect_metric_col(working, metric_col)
    if not resolved_metric:
        return {"error": "No numeric metric column found for root-cause analysis."}

    if working[resolved_metric].dtype == "object":
        working[resolved_metric] = working[resolved_metric].astype(str).str.replace(r'[$,£€ ]', '', regex=True)
    working[resolved_metric] = pd.to_numeric(working[resolved_metric], errors="coerce")
    working = working.dropna(subset=[resolved_metric])
    if len(working) < 10:
        return {"error": "Not enough data rows for root-cause analysis (min 10)."}

    date_col = _detect_date_col(working)
    period_label = "recent vs prior period"

    if date_col:
        working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
        working = working.dropna(subset=[date_col]).sort_values(date_col)
        
        min_date = working[date_col].min()
        max_date = working[date_col].max()
        total_span = (max_date - min_date).days
        
        # If dataset spans >= 180 days, compare the most recent 90-day window against prior 90-day window
        if total_span >= 180:
            window_days = min(90, total_span // 2)
            cutoff_recent = max_date - timedelta(days=window_days)
            cutoff_prior = cutoff_recent - timedelta(days=window_days)
            
            recent = working[working[date_col] > cutoff_recent]
            prior = working[(working[date_col] > cutoff_prior) & (working[date_col] <= cutoff_recent)]
            
            if len(prior) >= 5 and len(recent) >= 5:
                period_label = f"Last {window_days} Days ({cutoff_recent.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}) vs Prior {window_days} Days"
            else:
                # Fallback to temporal median split
                mid = working[date_col].quantile(0.5)
                prior = working[working[date_col] <= mid]
                recent = working[working[date_col] > mid]
                mid_str = pd.to_datetime(mid).strftime('%Y-%m-%d')
                period_label = f"After {mid_str} vs Before"
        else:
            mid = working[date_col].quantile(0.5)
            prior = working[working[date_col] <= mid]
            recent = working[working[date_col] > mid]
            mid_str = pd.to_datetime(mid).strftime('%Y-%m-%d')
            period_label = f"After {mid_str} vs Before"
    else:
        split = len(working) // 2
        prior = working.iloc[:split]
        recent = working.iloc[split:]
        period_label = "Second half vs first half of dataset"

    prior_total = float(prior[resolved_metric].sum())
    recent_total = float(recent[resolved_metric].sum())
    delta = recent_total - prior_total
    delta_pct = (delta / prior_total * 100) if prior_total != 0 else 0.0
    direction = "increased" if delta >= 0 else "decreased"

    # Volume vs. Ticket Price (Rate) Exact Decomposition
    # Total = Volume * Average Ticket
    # delta == (V_recent - V_prior) * P_prior + V_recent * (P_recent - P_prior)
    v_prior = max(len(prior), 1)
    v_recent = max(len(recent), 1)
    p_prior = prior_total / v_prior
    p_recent = recent_total / v_recent

    volume_effect = (v_recent - v_prior) * p_prior
    rate_effect = v_recent * (p_recent - p_prior)

    dimension_cols = _get_dimension_cols(working, resolved_metric)
    breakdowns: List[Dict[str, Any]] = []

    for dim in dimension_cols:
        try:
            prior_grp = prior.groupby(dim)[resolved_metric].sum()
            recent_grp = recent.groupby(dim)[resolved_metric].sum()
            all_keys = set(prior_grp.index) | set(recent_grp.index)

            dim_deltas = []
            for key in all_keys:
                p_val = float(prior_grp.get(key, 0))
                r_val = float(recent_grp.get(key, 0))
                dim_deltas.append({
                    "segment": str(key),
                    "prior": round(p_val, 2),
                    "recent": round(r_val, 2),
                    "delta": round(r_val - p_val, 2),
                })

            dim_deltas.sort(key=lambda x: abs(x["delta"]), reverse=True)
            top_movers = dim_deltas[:6]
            total_dim_delta = sum(d["delta"] for d in dim_deltas)

            breakdowns.append({
                "dimension": dim,
                "total_delta": round(total_dim_delta, 2),
                "top_movers": top_movers,
            })
        except Exception:
            continue

    breakdowns.sort(key=lambda x: abs(x["total_delta"]), reverse=True)

    # Narrative building
    narrative_parts = [
        f"{resolved_metric} {direction} by {abs(delta_pct):.1f}% ({period_label}).",
        f"Prior total: ${prior_total:,.2f} -> Recent: ${recent_total:,.2f} (Net change: ${delta:+,.2f}).",
    ]

    # Add Volume vs. Ticket decomposition explanation
    if abs(volume_effect) > 0.01 or abs(rate_effect) > 0.01:
        if abs(volume_effect) >= abs(rate_effect):
            narrative_parts.append(
                f"Decomposition indicates Transaction Volume was the primary driver "
                f"(${volume_effect:+,.2f} effect from {v_recent - v_prior:+d} transactions), "
                f"while Average Ticket Size contributed ${rate_effect:+,.2f} (${p_recent - p_prior:+,.2f}/order)."
            )
        else:
            narrative_parts.append(
                f"Decomposition indicates Average Ticket Size was the primary driver "
                f"(${rate_effect:+,.2f} effect, {p_recent - p_prior:+,.2f}/order), "
                f"alongside a ${volume_effect:+,.2f} effect from transaction volume."
            )

    if breakdowns:
        top_dim = breakdowns[0]
        top_segment = top_dim["top_movers"][0] if top_dim["top_movers"] else None
        if top_segment:
            narrative_parts.append(
                f"Largest individual driver: '{top_segment['segment']}' under {top_dim['dimension']} "
                f"({top_segment['delta']:+,.2f} change)."
            )

    if question:
        narrative_parts.insert(0, f"Analysis for: \"{question}\"")

    return {
        "metric": resolved_metric,
        "period_label": period_label,
        "prior_total": round(prior_total, 2),
        "recent_total": round(recent_total, 2),
        "delta": round(delta, 2),
        "delta_pct": round(delta_pct, 2),
        "direction": direction,
        "volume_effect": round(volume_effect, 2),
        "rate_effect": round(rate_effect, 2),
        "prior_transactions": v_prior,
        "recent_transactions": v_recent,
        "prior_avg_ticket": round(p_prior, 2),
        "recent_avg_ticket": round(p_recent, 2),
        "narrative": " ".join(narrative_parts),
        "breakdowns": breakdowns[:6],
        "chart_config": {
            "type": "bar" if breakdowns else "none",
            "labelCol": "segment",
            "valueCol": "delta",
        },
    }
