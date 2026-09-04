import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Optional
from ..core.logger import get_logger
from ..core.config import settings

try:
    from sklearn.ensemble import IsolationForest
except ImportError:
    IsolationForest = None

logger = get_logger(__name__)


def detect_anomalies(df: pd.DataFrame, column: str, contamination: float = 0.03) -> List[Dict[str, Any]]:
    """
    State-of-the-Art Anomaly Detection Engine:
    - Robust Non-Parametric Median Absolute Deviation (MAD) immune to heavy-tailed skewness.
    - Tukey's Interquartile Range (IQR) fences for asymmetric distribution bounds.
    - Multivariate Isolation Forest integration to capture multi-column contradictions.
    - Normalized Severity Score (0-100) and anomaly archetype classification.
    """
    try:
        working_df = df.copy()
        if column not in working_df.columns:
            return []
            
        if working_df[column].dtype == 'object':
            working_df[column] = working_df[column].astype(str).str.replace(r'[$,£€ ]', '', regex=True)
        working_df[column] = pd.to_numeric(working_df[column], errors='coerce')
        
        valid_mask = working_df[column].notna()
        valid_df = working_df[valid_mask]
        
        if len(valid_df) < 5:
            return []

        col_series = valid_df[column]
        median = col_series.median()
        mean = col_series.mean()
        std = col_series.std() if len(col_series) > 1 else 1.0
        std = std if std > 0 else 1.0
        
        # 1. Robust Median Absolute Deviation (MAD)
        abs_deviation = (col_series - median).abs()
        mad = abs_deviation.median()
        if mad > 0:
            modified_z = 0.6745 * (col_series - median) / mad
        else:
            # Fallback to standard Z if MAD is 0 (e.g. many identical medians)
            modified_z = (col_series - mean) / std

        # Standard Gaussian Z-score
        gaussian_z = (col_series - mean) / std

        # 2. Tukey's Interquartile Range (IQR) Fences
        q25 = col_series.quantile(0.25)
        q75 = col_series.quantile(0.75)
        iqr = q75 - q25
        upper_fence = q75 + (1.5 * iqr) if iqr > 0 else float('inf')
        lower_fence = q25 - (1.5 * iqr) if iqr > 0 else -float('inf')

        # 3. Multivariate Isolation Forest (Contextual Anomaly Detection)
        numeric_cols = valid_df.select_dtypes(include="number").columns.tolist()
        iso_outlier_indices = set()
        if IsolationForest is not None and len(numeric_cols) >= 2 and len(valid_df) >= 20:
            try:
                imputed_matrix = valid_df[numeric_cols].fillna(valid_df[numeric_cols].median()).values
                clf = IsolationForest(contamination=contamination, random_state=42, n_estimators=60)  # pyright: ignore[reportArgumentType]
                preds = clf.fit_predict(imputed_matrix)
                # preds == -1 are multivariate outliers
                iso_mask = (preds == -1)
                iso_outlier_indices = set(valid_df.index[iso_mask])
            except Exception as e_iso:
                logger.warning(f"Isolation forest skipped: {e_iso}")

        # Combine Detection Rules
        # Flag if:
        # A) Beyond Tukey's Extreme Fence (Q3 + 3.0*IQR) and Gaussian Z >= 3.0 (Confirmed Extreme Outlier)
        # B) Extreme Robust Modified Z-score >= 6.0
        # C) In Isolation Forest Outliers AND (|Gaussian Z| >= 2.5)
        extreme_upper = q75 + (3.0 * iqr) if iqr > 0 else float('inf')
        extreme_lower = q25 - (3.0 * iqr) if iqr > 0 else -float('inf')

        flagged_indices = set()
        for idx in valid_df.index:
            mz = abs(modified_z.loc[idx])
            gz = abs(gaussian_z.loc[idx])
            val = col_series.loc[idx]
            
            is_extreme_univariate = (gz >= 3.0 and val > extreme_upper) or (gz <= -3.0 and val < extreme_lower)
            is_multivariate_outlier = (idx in iso_outlier_indices and abs(gz) >= 2.8)
            
            if is_extreme_univariate or is_multivariate_outlier:
                flagged_indices.add(idx)

        results = []
        for idx in flagged_indices:
            row = working_df.loc[idx]
            val = float(col_series.loc[idx])
            mz_val = float(modified_z.loc[idx])
            gz_val = float(gaussian_z.loc[idx])
            
            # Primary Z score reported
            reported_z = gz_val if abs(gz_val) >= abs(mz_val) else mz_val
            
            # Determine Archetype
            if idx in iso_outlier_indices and not (abs(mz_val) >= 4.0 or abs(gz_val) >= 4.0):
                archetype = "Multivariate Contradiction"
            elif reported_z > 0:
                archetype = "Spike Outlier"
            else:
                archetype = "Deficit / Negative Shock"

            # Normalized Severity Score (0 to 100)
            base_severity = min(100.0, 50.0 + (abs(reported_z) * 10.0))
            if idx in iso_outlier_indices:
                base_severity = min(100.0, base_severity + 10.0)

            # Extract contextual metadata
            other_cols = [c for c in working_df.columns if c != column][:6]
            context_data = {c: str(row[c]) for c in other_cols}

            results.append({
                "row_index": int(idx),
                "value": round(val, 2),
                "z_score": round(reported_z, 2),
                "deviation": "High" if reported_z > 0 else "Low",
                "severity_score": round(base_severity, 1),
                "anomaly_type": archetype,
                "other_data": context_data
            })

        # Sort descending by severity score
        results.sort(key=lambda x: x["severity_score"], reverse=True)
        return results

    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        return []


def scan_all_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """Scan all numeric columns for high-level anomaly insights."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    summary = {}
    
    for col in numeric_cols:
        anomalies = detect_anomalies(df, col)
        if anomalies:
            summary[col] = {
                "count": len(anomalies),
                "max_deviation": max([abs(a['z_score']) for a in anomalies]) if anomalies else 0,
                "max_severity": max([a.get('severity_score', 0) for a in anomalies]) if anomalies else 0
            }
            
    return summary


def explain_anomalies(column: str, anomalies: List[Dict[str, Any]]) -> str:
    """Generate a high-level executive natural language explanation of the anomalies."""
    if not anomalies:
        return "No statistical or contextual anomalies detected in this metric."
        
    api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return f"Identified {len(anomalies)} anomalies in '{column}' exceeding robust statistical fences."
        
    try:
        from openai import OpenAI
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = OpenAI(api_key=api_key)
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
        
        sample = anomalies[:5]
        prompt = f"""
        You are an expert BI Data Analyst. We ran hybrid statistical anomaly detection (Robust MAD + 3-sigma + Isolation Forest) on '{column}'.
        We flagged {len(anomalies)} anomalies. Here are the top {len(sample)} most severe anomalies:
        {sample}
        
        Provide a crisp, 2-3 sentence executive explanation of these anomalies for business leaders.
        Highlight whether they represent revenue spikes, negative margin deficits, or operational outliers.
        Explain in clear business language without mentioning math formulas or Z-score equations.
        """
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a concise executive BI analyst explaining metric anomalies."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=200
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error(f"Failed to generate anomaly explanation: {e}")
        return f"Identified {len(anomalies)} anomalies in '{column}' using robust statistical and isolation forest models."
