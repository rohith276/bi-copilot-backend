import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

def generate_recommendations(df: pd.DataFrame, product_col: str, sales_col: str, inventory_col: str) -> List[Dict[str, Any]]:
    """
    Suggest business actions based on sales and inventory analysis.
    """
    recommendations = []
    
    # Work on a copy to avoid mutating the original
    df = df.copy()
    
    # Ensure numeric types
    df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce')
    df[inventory_col] = pd.to_numeric(df[inventory_col], errors='coerce')
    
    # Group by product to get aggregate stats
    summary = df.groupby(product_col).agg({
        sales_col: ['sum', 'mean'],
        inventory_col: 'mean'
    }).reset_index()
    summary.columns = [product_col, 'total_sales', 'avg_sales', 'avg_inventory']

    # 1. Critical Restock: Sales > Inventory * 5 (assuming high velocity)
    critical = summary[summary['total_sales'] > summary['avg_inventory'] * 2]
    for _, row in critical.iterrows():
        recommendations.append({
            "product": str(row[product_col]),
            "action": f"Critical Restock: {row[product_col]}",
            "reason": f"Sales velocity ({row['total_sales']:.0f}) is outpacing inventory levels ({row['avg_inventory']:.0f}). High risk of stockout.",
            "priority": "High"
        })
        
    # 2. Promotional Opportunity: High inventory, low sales
    stale = summary[(summary['avg_inventory'] > summary['total_sales'] * 3) & (summary['total_sales'] > 0)]
    for _, row in stale.iterrows():
        recommendations.append({
            "product": str(row[product_col]),
            "action": f"Launch Promotion for {row[product_col]}",
            "reason": "High inventory holding costs with slow sales movement. Recommend 10-15% discount.",
            "priority": "Medium"
        })
        
    # 3. Top Performer: High sales, stable inventory
    top = summary.nlargest(3, 'total_sales')
    for _, row in top.iterrows():
        if not any(r['product'] == str(row[product_col]) for r in recommendations):
            recommendations.append({
                "product": str(row[product_col]),
                "action": f"Maintain Premium Placement for {row[product_col]}",
                "reason": "Generating significant revenue. Ensure prime visibility in marketing channels.",
                "priority": "Low"
            })
        
    return recommendations

def analyze_business_health(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform a comprehensive health check of the business data.
    """
    try:
        if df.empty:
            return {"status": "Analysis Unavailable", "summary": "Dataset is empty."}

        # Data quality metrics
        missing_data_pct = df.isnull().mean().mean() * 100
        duplicate_pct = (df.duplicated().sum() / len(df)) * 100 if len(df) > 0 else 0
        
        # Find numeric columns and compute real stats
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        health_details = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(df.columns) - len(numeric_cols),
            "missing_data_pct": round(missing_data_pct, 1),
            "duplicate_pct": round(duplicate_pct, 1),
        }
        
        # Identify the highest-value numeric column as the primary metric
        if len(numeric_cols) > 0:
            col_sums = {col: round(float(df[col].sum()), 2) for col in numeric_cols}
            primary_col = max(col_sums, key=col_sums.get)
            health_details["primary_metric"] = primary_col
            health_details["primary_metric_total"] = col_sums[primary_col]

        # Composite health score
        quality_score = max(0, 100 - missing_data_pct - duplicate_pct)
        
        if quality_score >= 90:
            status = "Healthy"
            summary = f"Data quality is excellent ({quality_score:.0f}/100). {len(numeric_cols)} numeric columns available for analysis."
        elif quality_score >= 70:
            status = "Moderate"
            summary = f"Data quality is moderate ({quality_score:.0f}/100). {round(missing_data_pct, 1)}% missing data detected — consider imputing gaps."
        else:
            status = "Needs Attention"
            summary = f"Data quality needs improvement ({quality_score:.0f}/100). {round(missing_data_pct, 1)}% missing data and {round(duplicate_pct, 1)}% duplicates detected."
        
        return {
            "status": status,
            "data_quality_score": round(quality_score, 1),
            "details": health_details,
            "summary": summary,
        }
    except Exception:
        return {"status": "Analysis Unavailable", "summary": "Insufficient data for health check."}
