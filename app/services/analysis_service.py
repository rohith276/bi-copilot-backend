import pandas as pd
import numpy as np
from typing import List, Dict, Any
from ..schemas.query import QueryRequest, Filter, GroupBy

def process_query(df: pd.DataFrame, request: QueryRequest) -> Dict[str, Any]:
    processed_df: pd.DataFrame = df.copy()

    # 1. Apply Filters
    if request.filters:
        for f in request.filters:
            if f.column not in processed_df.columns:
                continue
            
            if f.operator == "eq":
                processed_df = processed_df[processed_df[f.column] == f.value]
            elif f.operator == "ne":
                processed_df = processed_df[processed_df[f.column] != f.value]
            elif f.operator == "gt":
                processed_df = processed_df[processed_df[f.column] > f.value]
            elif f.operator == "lt":
                processed_df = processed_df[processed_df[f.column] < f.value]
            elif f.operator == "gte":
                processed_df = processed_df[processed_df[f.column] >= f.value]
            elif f.operator == "lte":
                processed_df = processed_df[processed_df[f.column] <= f.value]
            elif f.operator == "contains":
                processed_df = processed_df[processed_df[f.column].astype(str).str.contains(str(f.value), case=False)]

    # 2. Apply Group By
    if request.group_by:
        # Ensure aggregation columns are numeric where possible
        valid_agg_funcs = {}
        for col, func in request.group_by.agg_funcs.items():
            if col in processed_df.columns:
                normalized_func = func.lower()
                if normalized_func != "count":
                    processed_df.loc[:, col] = pd.to_numeric(processed_df[col], errors='coerce')
                valid_agg_funcs[col] = normalized_func
        
        valid_gb_cols = [c for c in request.group_by.columns if c in processed_df.columns]
        
        if valid_gb_cols and valid_agg_funcs:
            # Rename agg funcs keys to avoid collision with groupby cols
            # e.g. {"price": "sum"} -> {"price": ("price", "sum")} in newer pandas or just handle collision
            try:
                # Safely perform aggregation
                processed_df = processed_df.groupby(valid_gb_cols).agg(valid_agg_funcs).reset_index()
            except ValueError as e:
                if "already exists" in str(e):
                    # Handle renaming if collision detected
                    agg_results = processed_df.groupby(valid_gb_cols).agg(valid_agg_funcs)
                    # Rename columns to {col}_{func}
                    agg_results.columns = [f"{col}_{func}" for col, func in valid_agg_funcs.items()]
                    processed_df = agg_results.reset_index()
                else:
                    raise e

    # 3. Apply Sorting
    if request.sort_by and request.sort_by in processed_df.columns:
        processed_df = processed_df.sort_values(by=request.sort_by, ascending=not request.sort_desc)

    total_rows = len(processed_df)
    
    # 4. Apply Limit
    result_df = processed_df.head(request.limit)

    return {
        "columns": result_df.columns.tolist(),
        "data": result_df.replace({np.nan: None}).to_dict(orient="records"),
        "total_rows": total_rows
    }

def calculate_kpis(df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Any]:
    kpis = {}
    for col in numeric_columns:
        if col in df.columns:
            kpis[col] = {
                "sum": df[col].sum(),
                "mean": df[col].mean(),
                "max": df[col].max(),
                "min": df[col].min(),
                "count": df[col].count()
            }
    return kpis
