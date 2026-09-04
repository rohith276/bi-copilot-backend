from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from ..db.session import get_db
from ..services import dataset_service, forecasting_service, recommendation_service, report_service, anomaly_service, analysis_service
from ..schemas.query import QueryRequest, Filter, GroupBy
from ..schemas.analytics import ForecastRequest, ForecastResult, PredictionRequest, PredictionResult, RecommendationResults, AnomalyRequest, AnomalyResult, VisualQueryRequest, VisualQueryResult


router = APIRouter(prefix="/analytics", tags=["analytics"])

def _get_dataset_or_404(dataset_id: int, db: Session):
    """Internal helper — looks up a dataset or raises 404."""
    db_dataset = db.query(dataset_service.DatasetModel).filter(dataset_service.DatasetModel.id == dataset_id).first()
    if db_dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return db_dataset

@router.post("/{dataset_id}/forecast", response_model=ForecastResult)
def get_dataset_forecast(dataset_id: int, request: ForecastRequest, db: Session = Depends(get_db)):
    db_dataset = _get_dataset_or_404(dataset_id, db)
    # Use sampling for forecasting on massive datasets
    nrows = 200000 if db_dataset.row_count > 200000 else None
    df = dataset_service.get_dataset_df(str(db_dataset.file_path), nrows=nrows)
    result = forecasting_service.forecast_sales(df, request.date_col, request.value_col, request.periods)
    if "error" in result:
       raise HTTPException(status_code=400, detail=result["error"])
    return result

@router.post("/{dataset_id}/predict", response_model=PredictionResult)
def get_dataset_prediction(dataset_id: int, request: PredictionRequest, db: Session = Depends(get_db)):
    db_dataset = _get_dataset_or_404(dataset_id, db)
    # Use sampling for prediction on massive datasets
    nrows = 200000 if db_dataset.row_count > 200000 else None
    df = dataset_service.get_dataset_df(str(db_dataset.file_path), nrows=nrows)
    result = forecasting_service.predict_trend(df, request.target_col, request.feature_cols)
    if "error" in result:
       raise HTTPException(status_code=400, detail=result["error"])
    return result

@router.get("/{dataset_id}/recommendations", response_model=List[RecommendationResults])
def get_recommendations(dataset_id: int, product_col: str, sales_col: str, inventory_col: str, db: Session = Depends(get_db)):
    db_dataset = _get_dataset_or_404(dataset_id, db)
    # Sampling for recommendations
    nrows = 100000 if db_dataset.row_count > 100000 else None
    df = dataset_service.get_dataset_df(str(db_dataset.file_path), nrows=nrows)
    return recommendation_service.generate_recommendations(df, product_col, sales_col, inventory_col)

@router.get("/{dataset_id}/report")
def get_dataset_report(dataset_id: int, db: Session = Depends(get_db)):
    """Generate a comprehensive intelligence report for the dataset."""
    db_dataset = _get_dataset_or_404(dataset_id, db)
    # Use sampling for report generation
    nrows = 100000 if db_dataset.row_count > 100000 else None
    df = dataset_service.get_dataset_df(str(db_dataset.file_path), nrows=nrows)
    try:
        return report_service.generate_report(df, db_dataset)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@router.post("/{dataset_id}/anomalies", response_model=List[AnomalyResult])
def get_dataset_anomalies(dataset_id: int, request: AnomalyRequest, db: Session = Depends(get_db)):
    """Detect statistical anomalies in a specific column."""
    db_dataset = _get_dataset_or_404(dataset_id, db)
    # Sampling for anomaly detection
    nrows = 100000 if db_dataset.row_count > 100000 else None
    df = dataset_service.get_dataset_df(str(db_dataset.file_path), nrows=nrows)
    return anomaly_service.detect_anomalies(df, request.column)

@router.post("/{dataset_id}/visual-query", response_model=VisualQueryResult)
def process_visual_query(dataset_id: int, request: VisualQueryRequest, db: Session = Depends(get_db)):
    import pandas as pd
    import numpy as np

    db_dataset = _get_dataset_or_404(dataset_id, db)
    df = dataset_service.get_dataset_df(str(db_dataset.file_path))

    # ── 1. Detect if x-axis is a date column ─────────────────────────
    x_col = request.x_axis
    x_is_date = False
    try:
        test_col = pd.to_datetime(df[x_col], errors='coerce')
        non_null_ratio = test_col.notna().sum() / max(len(test_col), 1)
        if non_null_ratio > 0.5:
            df[x_col] = test_col
            x_is_date = True
    except Exception:
        pass

    # ── 2. Apply drill-down filters ───────────────────────────────────
    q_filters = []
    if request.filters:
        for f in request.filters:
            q_filters.append(Filter(column=f["column"], operator="eq", value=f["value"]))

    # ── 3. Apply range filters on x-axis ──────────────────────────────
    if request.range_min is not None and request.range_min != "":
        if x_is_date:
            try:
                range_min_dt = pd.to_datetime(request.range_min)
                df = df[df[x_col] >= range_min_dt]
            except Exception:
                pass
        else:
            try:
                range_min_num = float(request.range_min)
                df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
                df = df[df[x_col] >= range_min_num]
            except (ValueError, TypeError):
                pass

    if request.range_max is not None and request.range_max != "":
        if x_is_date:
            try:
                range_max_dt = pd.to_datetime(request.range_max)
                df = df[df[x_col] <= range_max_dt]
            except Exception:
                pass
        else:
            try:
                range_max_num = float(request.range_max)
                df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
                df = df[df[x_col] <= range_max_num]
            except (ValueError, TypeError):
                pass

    # ── 4. Apply date granularity bucketing ────────────────────────────
    group_col = x_col
    granularity = request.date_granularity

    if x_is_date:
        # Auto-detect granularity if not specified
        if not granularity:
            n_unique = df[x_col].dt.date.nunique()
            if n_unique > 90:
                granularity = "month"
            elif n_unique > 30:
                granularity = "week"
            else:
                granularity = "day"

        bucket_col = f"__{x_col}_bucket"
        if granularity == "year":
            df[bucket_col] = df[x_col].dt.to_period("Y").astype(str)
        elif granularity == "quarter":
            df[bucket_col] = df[x_col].dt.to_period("Q").astype(str)
        elif granularity == "month":
            df[bucket_col] = df[x_col].dt.to_period("M").astype(str)
        elif granularity == "week":
            df[bucket_col] = df[x_col].dt.to_period("W").apply(lambda p: str(p.start_time.date()))
        else:  # day
            df[bucket_col] = df[x_col].dt.date.astype(str)

        group_col = bucket_col

    # ── 5. Perform aggregation ────────────────────────────────────────
    agg_func = request.aggregate.lower()
    y_col = request.y_axis

    if agg_func != "count":
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')

    # Apply drill-down filters before aggregation
    for f in q_filters:
        if f.column in df.columns:
            df = df[df[f.column] == f.value]

    try:
        if agg_func == "count":
            agg_df = df.groupby(group_col)[y_col].count().reset_index()
        elif agg_func == "avg":
            agg_df = df.groupby(group_col)[y_col].mean().reset_index()
        elif agg_func == "min":
            agg_df = df.groupby(group_col)[y_col].min().reset_index()
        elif agg_func == "max":
            agg_df = df.groupby(group_col)[y_col].max().reset_index()
        else:  # sum
            agg_df = df.groupby(group_col)[y_col].sum().reset_index()
    except Exception:
        agg_df = pd.DataFrame(columns=[group_col, y_col])

    agg_df = agg_df.rename(columns={y_col: "agg_value"})

    # ── 6. Sort results ───────────────────────────────────────────────
    sort_order = request.sort_order
    if x_is_date and not sort_order:
        # Date columns: always sort chronologically by default
        agg_df = agg_df.sort_values(by=group_col, ascending=True)
    elif sort_order == "asc":
        agg_df = agg_df.sort_values(by="agg_value", ascending=True)
    elif sort_order == "desc":
        agg_df = agg_df.sort_values(by="agg_value", ascending=False)

    # ── 7. Smart limits & "Other" bucket ──────────────────────────────
    total_unique = len(agg_df)
    applied_limit = request.limit
    chart_type = request.chart_type or "bar"

    # Auto-apply smart limits if the user hasn't set one
    radial_charts = {"pie", "doughnut", "polarArea", "radar"}
    if applied_limit is None:
        if chart_type in radial_charts and total_unique > 12:
            applied_limit = 12
        elif chart_type in {"bar", "horizontalBar"} and total_unique > 50:
            applied_limit = 50

    if applied_limit is not None and total_unique > applied_limit:
        # For radial charts, bucket overflow into "Other"
        if chart_type in radial_charts:
            # Sort by value desc for top-N selection
            sorted_df = agg_df.sort_values(by="agg_value", ascending=False)
            top_rows = sorted_df.head(applied_limit)
            other_rows = sorted_df.iloc[applied_limit:]
            other_sum = other_rows["agg_value"].sum()

            other_row = pd.DataFrame([{group_col: "Other", "agg_value": other_sum}])
            agg_df = pd.concat([top_rows, other_row], ignore_index=True)
        else:
            # For non-radial: just take top/bottom N
            if sort_order == "asc":
                agg_df = agg_df.head(applied_limit)
            else:
                agg_df = agg_df.sort_values(by="agg_value", ascending=False).head(applied_limit)
                if x_is_date:
                    # Re-sort chronologically after limiting
                    agg_df = agg_df.sort_values(by=group_col, ascending=True)

    # ── 8. Rename the bucket column back to x_axis name ───────────────
    if group_col != x_col:
        agg_df = agg_df.rename(columns={group_col: x_col})

    # ── 9. Build response ─────────────────────────────────────────────
    agg_df = agg_df.replace({np.nan: None})
    result_data = agg_df.to_dict(orient="records")

    # Compute available date/numeric range for the frontend controls
    x_col_type = "date" if x_is_date else "string"
    available_range = {}
    if x_is_date:
        try:
            raw_df = dataset_service.get_dataset_df(str(db_dataset.file_path))
            raw_dates = pd.to_datetime(raw_df[x_col], errors='coerce').dropna()
            if len(raw_dates) > 0:
                available_range = {
                    "min": str(raw_dates.min().date()),
                    "max": str(raw_dates.max().date()),
                }
        except Exception:
            pass

    # Generate SQL representation
    gran_sql = f"DATE_TRUNC('{granularity}', {x_col})" if x_is_date and granularity else x_col
    sql_query = f"SELECT {gran_sql}, {request.aggregate.upper()}({request.y_axis}) AS agg_value FROM dataset"
    if request.range_min or request.range_max:
        clauses = []
        if request.range_min:
            clauses.append(f"{x_col} >= '{request.range_min}'")
        if request.range_max:
            clauses.append(f"{x_col} <= '{request.range_max}'")
        sql_query += " WHERE " + " AND ".join(clauses)
    sql_query += f" GROUP BY {gran_sql}"
    if sort_order:
        sql_query += f" ORDER BY agg_value {'ASC' if sort_order == 'asc' else 'DESC'}"
    if applied_limit:
        sql_query += f" LIMIT {applied_limit}"

    return {
        "sql_query": sql_query,
        "result": {
            "columns": [x_col, "agg_value"],
            "data": result_data,
            "total_rows": total_unique,
        },
        "x_column_type": x_col_type,
        "available_range": available_range,
        "applied_granularity": granularity if x_is_date else None,
    }

