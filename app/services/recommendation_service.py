import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import math


def generate_recommendations(df: pd.DataFrame, product_col: str, sales_col: str, inventory_col: str) -> List[Dict[str, Any]]:
    """
    Supply-Chain Operations Research & Inventory Action Engine:
    - ABC Pareto Classification (Class A: 80% revenue, Class B: 15%, Class C: 5% long tail).
    - Statistical Reorder Point (ROP) and safety stock buffer estimation.
    - Liquidation & capital liberation recommendations for dormant/slow-moving inventory.
    """
    recommendations = []
    
    try:
        working_df = df.copy()
        
        if product_col not in working_df.columns:
            return []
            
        # Clean numeric fields
        for col in [sales_col, inventory_col]:
            if col in working_df.columns:
                if working_df[col].dtype == 'object':
                    working_df[col] = working_df[col].astype(str).str.replace(r'[$,£€ ]', '', regex=True)
                working_df[col] = pd.to_numeric(working_df[col], errors='coerce')
        
        working_df = working_df.dropna(subset=[product_col, sales_col, inventory_col])
        if len(working_df) < 5:
            return []

        # Find date column if present to calculate true daily velocity
        date_cols = [c for c in working_df.columns if 'date' in c.lower()]
        days_span = 90  # Default 90-day window
        if date_cols:
            try:
                d_series = pd.to_datetime(working_df[date_cols[0]], errors='coerce').dropna()
                if len(d_series) > 1:
                    span = (d_series.max() - d_series.min()).days
                    if span > 10:
                        days_span = span
            except Exception:
                pass

        # Aggregate product statistics
        summary = working_df.groupby(product_col).agg(
            total_sales=(sales_col, 'sum'),
            total_qty=(inventory_col, 'sum'),
            avg_qty=(inventory_col, 'mean'),
            std_qty=(inventory_col, 'std'),
            order_count=(sales_col, 'count'),
            avg_ticket=(sales_col, 'mean')
        ).reset_index()

        summary['std_qty'] = summary['std_qty'].fillna(1.0)
        total_portfolio_sales = summary['total_sales'].sum()
        if total_portfolio_sales <= 0:
            return []

        # 1. ABC Pareto Segmentation
        summary = summary.sort_values(by='total_sales', ascending=False)
        summary['cum_sales'] = summary['total_sales'].cumsum()
        summary['cum_pct'] = summary['cum_sales'] / total_portfolio_sales

        def assign_abc(row):
            if row['cum_pct'] <= 0.80 or row.name in summary.index[:5]:
                return 'Class A'
            elif row['cum_pct'] <= 0.95:
                return 'Class B'
            return 'Class C'

        summary['abc_class'] = summary.apply(assign_abc, axis=1)

        # 2. Generate Action Specs
        # Class A: Critical Replenishment & Stockout Prevention (Top 80% of revenue)
        class_a = summary[summary['abc_class'] == 'Class A'].head(10)
        for _, row in class_a.iterrows():
            daily_run_rate = row['total_qty'] / max(days_span, 1)
            # Standard 14-day lead time with 95% service level buffer (Z=1.65)
            lead_time_days = 14
            safety_stock = math.ceil(1.65 * row['std_qty'] * math.sqrt(lead_time_days / 7))
            reorder_units = math.ceil(daily_run_rate * 30 + safety_stock)
            
            recommendations.append({
                "product": str(row[product_col]),
                "action": f"Critical Restock: {reorder_units} units (Class A)",
                "reason": f"Top revenue driver (${row['total_sales']:,.2f}, {row['order_count']} orders). Velocity: {daily_run_rate:.1f} units/day. Protect safety stock ({safety_stock} units).",
                "priority": "High",
                "category": "Class A (Revenue Driver)",
                "velocity": round(float(daily_run_rate), 2),
                "recommended_units": float(reorder_units)
            })

        # Class C: Slow Movers & Capital Liberation (Bottom 5% of portfolio)
        class_c = summary[summary['abc_class'] == 'Class C'].tail(8)
        for _, row in class_c.iterrows():
            recommendations.append({
                "product": str(row[product_col]),
                "action": f"Liquidation / Discount for {row[product_col]} (Class C)",
                "reason": f"Low volume velocity (${row['total_sales']:,.2f} total across {row['order_count']} orders). Recommend 15-20% promotional discount to liberate tied-up capital.",
                "priority": "Medium",
                "category": "Class C (Slow Mover)",
                "velocity": round(float(row['total_qty'] / max(days_span, 1)), 2),
                "recommended_units": 0.0
            })

        # Top Margin / Strategic Anchors
        top_ticket = summary.nlargest(3, 'avg_ticket')
        for _, row in top_ticket.iterrows():
            if not any(r['product'] == str(row[product_col]) for r in recommendations):
                recommendations.append({
                    "product": str(row[product_col]),
                    "action": f"Premium Marketing Placement for {row[product_col]}",
                    "reason": f"Highest transaction ticket size (${row['avg_ticket']:,.2f}/order). High gross margin potential — prioritize in marketing campaigns.",
                    "priority": "Low",
                    "category": "Strategic Anchor",
                    "velocity": round(float(row['total_qty'] / max(days_span, 1)), 2),
                    "recommended_units": float(math.ceil(row['total_qty'] * 0.2))
                })

        return recommendations[:20]

    except Exception as e:
        return []


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
        numeric_cols = df.select_dtypes(include="number").columns
        
        health_details: Dict[str, Any] = {
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
            primary_col = max(col_sums.keys(), key=lambda k: col_sums[k])
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
