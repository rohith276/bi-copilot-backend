import pandas as pd
import numpy as np
from typing import List, Dict, Any
from ..core.logger import get_logger

logger = get_logger(__name__)

def detect_anomalies(df: pd.DataFrame, column: str, contamination: float = 0.05) -> List[Dict[str, Any]]:
    """
    Detect statistical outliers in a numeric column.
    Provides row indices and values for anomalies.
    """
    try:
        # Work on a copy to avoid mutating the original
        working_df = df.copy()
        working_df[column] = pd.to_numeric(working_df[column], errors='coerce')
        valid_data = working_df.dropna(subset=[column])
        
        if valid_data.empty:
            return []

        # Simple Z-score implementation for reliability (no heavy sklearn dep needed for basic)
        mean = valid_data[column].mean()
        std = valid_data[column].std()
        
        if std == 0:
            return []

        # Calculate Z-score
        z_scores = (valid_data[column] - mean) / std
        
        # Threshold for anomaly (standard 3 sigma)
        threshold = 3.0
        anomalies_idx = valid_data.index[abs(z_scores) > threshold].tolist()
        
        results = []
        for idx in anomalies_idx:
            row = working_df.loc[idx]
            results.append({
                "row_index": int(idx),
                "value": round(float(row[column]), 2),
                "z_score": round(float(z_scores.loc[idx]), 2),
                "deviation": "High" if z_scores.loc[idx] > 0 else "Low",
                "other_data": {
                    col: str(row[col]) for col in working_df.columns[:5] # Provide some context
                }
            })
            
        return sorted(results, key=lambda x: abs(x['z_score']), reverse=True)
        
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        return []

def scan_all_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Scan all numeric columns for high-level anomaly insights.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = {}
    
    for col in numeric_cols:
        anomalies = detect_anomalies(df, col)
        if anomalies:
            summary[col] = {
                "count": len(anomalies),
                "max_deviation": max([abs(a['z_score']) for a in anomalies]) if anomalies else 0
            }
            
    return summary
