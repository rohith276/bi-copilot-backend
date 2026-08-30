import pandas as pd
import numpy as np
from typing import Dict, Any
from ..core.logger import get_logger

logger = get_logger(__name__)

def _strip_text(value):
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    return stripped if stripped else pd.NA

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Starting cleaning process for dataframe with {len(df)} rows")

    # 1. Remove completely empty rows and columns
    df = df.dropna(how='all', axis=0)
    df = df.dropna(how='all', axis=1)

    # 2. Standardize Dates (Crucial for SQLite and AI Time Context)
    for col in df.columns:
        # Check if the column name implies a date or if pandas typed it as datetime
        if 'date' in str(col).lower() or pd.api.types.is_datetime64_any_dtype(df[col]):
            try:
                # Attempt to parse mixed formats (like DD/MM/YYYY)
                temp = pd.to_datetime(df[col], errors='coerce')
                # If over 50% parse successfully, treat it as a hard date column
                if temp.notna().sum() > (len(df) * 0.5):
                    df[col] = temp.dt.strftime('%Y-%m-%d')
            except Exception:
                pass

    # 3. Trim whitespace from strings without overwriting missing data
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].map(_strip_text)

    return df

def _sanitize_stats(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_stats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_stats(v) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, np.floating):
        if pd.isna(obj) or np.isinf(obj):
            return None
        return obj
    return obj

def get_column_stats(df: pd.DataFrame):
    # Performance Optimization for Million-Row Datasets
    # If the dataset is massive, sample it for statistical calculations to prevent hangs
    if len(df) > 100000:
        logger.info(f"Massive dataset detected ({len(df)} rows). Sampling 100k rows for statistics.")
        df_stats = df.sample(n=100000, random_state=42)
    else:
        df_stats = df

    stats = []
    for col in df_stats.columns:
        series = df_stats[col]
        col_stats: Dict[str, Any] = {
            "name": col,
            "type": str(series.dtype),
            "missing_values": int(series.isna().sum()),
            "unique_values": int(series.nunique(dropna=True))
        }
        
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_datetime = pd.api.types.is_datetime64_any_dtype(series) or 'date' in str(col).lower()
        
        if is_datetime:
            col_stats["bi_type"] = "datetime"
        elif is_numeric:
            if col_stats["unique_values"] < 15 or str(col).lower().endswith("id"):
                col_stats["bi_type"] = "dimension"
            else:
                col_stats["bi_type"] = "metric"
        else:
            col_stats["bi_type"] = "dimension"
        if pd.api.types.is_numeric_dtype(series):
            numeric_series = pd.to_numeric(series, errors='coerce').dropna()
            if not numeric_series.empty:
                std_value = numeric_series.std()
                skew_value = numeric_series.skew()
                std_float = 0.0 if pd.isna(std_value) else float(std_value)
                skew_float = 0.0 if pd.isna(skew_value) else float(skew_value)
                try:
                    q1 = float(numeric_series.quantile(0.25))
                    q3 = float(numeric_series.quantile(0.75))
                    col_stats.update({
                        "min": round(float(numeric_series.min()), 2),
                        "max": round(float(numeric_series.max()), 2),
                        "mean": round(float(numeric_series.mean()), 2),
                        "median": round(float(numeric_series.median()), 2),
                        "std": round(std_float, 2),
                        "skewness": round(skew_float, 2),
                        "q1": round(q1, 2),
                        "q3": round(q3, 2),
                        "iqr": round(q3 - q1, 2),
                        "zero_count": int((numeric_series == 0).sum()),
                    })
                except Exception:
                    pass
        stats.append(col_stats)
    return _sanitize_stats(stats)
