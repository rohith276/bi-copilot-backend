from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from ..db.session import get_db
from ..services import dataset_service, forecasting_service, recommendation_service, report_service, anomaly_service, analysis_service, auto_dashboard_service, dashboard_service, root_cause_service
from ..schemas.dashboard import AutoDashboardResult
from ..schemas.semantic import RootCauseRequest, RootCauseResult
from ..schemas.query import QueryRequest, Filter, GroupBy, JoinSuggestionRequest, JoinSuggestionResponse
from ..schemas.analytics import ForecastRequest, ForecastResult, PredictionRequest, PredictionResult, RecommendationResults, AnomalyRequest, AnomalyResult, AnomalyResponse, VisualQueryRequest, VisualQueryResult


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

    multiplier = request.scenario_multiplier or 1.0
    if multiplier != 1.0:
        base_values = result["values"]
        result["base_values"] = base_values
        result["scenario_values"] = [round(v * multiplier, 2) for v in base_values]
        result["values"] = result["scenario_values"]
        if "lower_bounds" in result and result["lower_bounds"]:
            result["lower_bounds"] = [round(v * multiplier, 2) for v in result["lower_bounds"]]
        if "upper_bounds" in result and result["upper_bounds"]:
            result["upper_bounds"] = [round(v * multiplier, 2) for v in result["upper_bounds"]]

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

@router.post("/{dataset_id}/anomalies", response_model=AnomalyResponse)
def get_dataset_anomalies(dataset_id: int, request: AnomalyRequest, db: Session = Depends(get_db)):
    """Detect statistical anomalies in a specific column and explain them."""
    db_dataset = _get_dataset_or_404(dataset_id, db)
    # Sampling for anomaly detection
    nrows = 100000 if db_dataset.row_count > 100000 else None
    df = dataset_service.get_dataset_df(str(db_dataset.file_path), nrows=nrows)
    
    anomalies = anomaly_service.detect_anomalies(df, request.column)
    explanation = anomaly_service.explain_anomalies(request.column, anomalies)
    
    return AnomalyResponse(
        anomalies=anomalies,
        explanation=explanation
    )

@router.get("/{dataset_id}/suggest-chart")
def suggest_chart_type(dataset_id: int, x_col: str, y_col: str, agg: str = "sum", db: Session = Depends(get_db)):
    """AI chart type suggestion based on data shape heuristics."""
    import pandas as pd
    db_dataset = _get_dataset_or_404(dataset_id, db)
    df = dataset_service.get_dataset_df(str(db_dataset.file_path), nrows=5000)
    
    if x_col not in df.columns or y_col not in df.columns:
        return {"chart_type": "bar", "reason": "Default chart type for unknown columns."}
    
    x_dtype = str(df[x_col].dtype)
    x_nunique = df[x_col].nunique()
    is_date = 'date' in x_col.lower() or pd.api.types.is_datetime64_any_dtype(df[x_col])
    
    # Try to detect date-like object columns
    if not is_date and x_dtype == 'object':
        try:
            parsed = pd.to_datetime(df[x_col].dropna().head(20), errors='coerce')
            if parsed.notna().sum() > 15:
                is_date = True
        except Exception:
            pass
    
    # Decision tree
    if is_date:
        return {"chart_type": "line", "reason": f"'{x_col}' is a date/time column — line charts best show temporal trends."}
    
    if x_nunique <= 5 and agg in ("sum", "count"):
        return {"chart_type": "doughnut", "reason": f"'{x_col}' has only {x_nunique} categories — a donut chart shows composition clearly."}
    
    if x_nunique <= 12:
        return {"chart_type": "bar", "reason": f"'{x_col}' has {x_nunique} categories — bar charts are ideal for categorical comparisons."}
    
    if x_nunique <= 30:
        return {"chart_type": "horizontalBar", "reason": f"'{x_col}' has {x_nunique} categories — horizontal bars handle many labels without overlap."}
    
    if x_nunique > 30 and pd.api.types.is_numeric_dtype(df[x_col]):
        return {"chart_type": "scatter", "reason": f"Both axes are numeric with high cardinality — scatter plots reveal correlations."}
    
    return {"chart_type": "bar", "reason": f"Bar chart is the most versatile default for {x_nunique} categories."}


@router.post("/{dataset_id}/root-cause", response_model=RootCauseResult)
def perform_root_cause_analysis(dataset_id: int, request: RootCauseRequest, db: Session = Depends(get_db)):
    db_dataset = _get_dataset_or_404(dataset_id, db)
    
    try:
        df = dataset_service.get_dataset_df(str(db_dataset.file_path))
        return root_cause_service.analyze_root_cause(df, request.metric_col, request.question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform root cause analysis: {str(e)}")

@router.post("/suggest-joins", response_model=JoinSuggestionResponse)
def suggest_joins(request: JoinSuggestionRequest, db: Session = Depends(get_db)):
    """Auto Data Modeling: Suggest join keys between two datasets based on column names and types."""
    left_ds = _get_dataset_or_404(request.left_dataset_id, db)
    right_ds = _get_dataset_or_404(request.right_dataset_id, db)
    
    left_df = dataset_service.get_dataset_df(str(left_ds.file_path))
    right_df = dataset_service.get_dataset_df(str(right_ds.file_path))
    
    suggestions = []
    
    left_cols = {c.lower(): c for c in left_df.columns}
    right_cols = {c.lower(): c for c in right_df.columns}
    
    # Simple heuristics: Exact match on standard ID columns
    for l_lower, l_orig in left_cols.items():
        if l_lower in right_cols:
            r_orig = right_cols[l_lower]
            # If it's an ID column, high confidence
            if 'id' in l_lower or l_lower.endswith('key'):
                suggestions.append({
                    "left_col": l_orig,
                    "right_col": r_orig,
                    "confidence": 0.95,
                    "reason": "Exact match on identifier column."
                })
            else:
                # Same name, not explicitly an ID, still possible
                suggestions.append({
                    "left_col": l_orig,
                    "right_col": r_orig,
                    "confidence": 0.60,
                    "reason": "Exact match on column name."
                })
                
    # If no exact match, look for fuzzy matches (e.g. user_id -> id)
    if not suggestions:
        for l_lower, l_orig in left_cols.items():
            if l_lower.endswith('_id'):
                entity = l_lower.replace('_id', '')
                if 'id' in right_cols:
                    suggestions.append({
                        "left_col": l_orig,
                        "right_col": right_cols['id'],
                        "confidence": 0.85,
                        "reason": f"Inferred foreign key {l_orig} to primary key 'id'."
                    })
    
    # Sort by confidence
    suggestions = sorted(suggestions, key=lambda x: x['confidence'], reverse=True)
    return JoinSuggestionResponse(suggestions=suggestions)


@router.post("/{dataset_id}/auto-dashboard", response_model=AutoDashboardResult)
def generate_auto_dashboard(dataset_id: int, db: Session = Depends(get_db)):
    """AI auto-dashboard: analyzes schema, generates KPIs + charts, and pins them."""
    db_dataset = _get_dataset_or_404(dataset_id, db)
    nrows = 100000 if db_dataset.row_count > 100000 else None
    df = dataset_service.get_dataset_df(str(db_dataset.file_path), nrows=nrows)

    generated = auto_dashboard_service.generate_auto_dashboard(df, db_dataset)
    pinned_items = []
    for item in generated["items"]:
        pinned = dashboard_service.pin_item(
            dataset_id,
            item["title"],
            item["sql_query"],
            item["chart_config"],
            item["layout"],
            db,
        )
        pinned_items.append(pinned)

    return {
        "summary": generated["summary"],
        "items": pinned_items,
        "pinned_count": len(pinned_items),
    }


def _get_dimension_values(db_dataset, exclude_col: str) -> dict:
    """Return unique values (top 50) for each string/categorical column, for filter UI."""
    import pandas as pd
    try:
        df = dataset_service.get_dataset_df(str(db_dataset.file_path), nrows=10000)
        dim_values = {}
        for col in df.columns:
            if col == exclude_col:
                continue
            if df[col].dtype == "object" or str(df[col].dtype) == "string":
                uniques = df[col].dropna().astype(str).unique()
                if 1 < len(uniques) <= 50:  # Only show filterable columns (not IDs with 10K uniques)
                    dim_values[col] = sorted(uniques.tolist())
        return dim_values
    except Exception:
        return {}

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

    # ── 2b. Apply dimension multi-select filters ──────────────────────
    if request.dimension_filters:
        for dim_col, selected_values in request.dimension_filters.items():
            if dim_col in df.columns and selected_values:
                df = df[df[dim_col].astype(str).isin(selected_values)]

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
    color_by = request.color_by

    if agg_func != "count":
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')

    # Apply drill-down filters before aggregation
    for f in q_filters:
        if f.column in df.columns:
            df = df[df[f.column] == f.value]

    # Determine groupby columns — include color_by if specified
    group_cols = [group_col]
    if color_by and color_by in df.columns:
        group_cols = [group_col, color_by]

    try:
        if agg_func == "count":
            agg_df = df.groupby(group_cols)[y_col].count().reset_index()
        elif agg_func == "avg":
            agg_df = df.groupby(group_cols)[y_col].mean().reset_index()
        elif agg_func == "min":
            agg_df = df.groupby(group_cols)[y_col].min().reset_index()
        elif agg_func == "max":
            agg_df = df.groupby(group_cols)[y_col].max().reset_index()
        else:  # sum
            agg_df = df.groupby(group_cols)[y_col].sum().reset_index()
    except Exception:
        agg_df = pd.DataFrame(columns=group_cols + [y_col])

    agg_df = agg_df.rename(columns={y_col: "agg_value"})

    # ── 6. Sort results ───────────────────────────────────────────────
    sort_order = request.sort_order
    if x_is_date and not sort_order:
        agg_df = agg_df.sort_values(by=group_col, ascending=True)
    elif sort_order == "asc":
        agg_df = agg_df.sort_values(by="agg_value", ascending=True)
    elif sort_order == "desc":
        agg_df = agg_df.sort_values(by="agg_value", ascending=False)

    # ── 7. Smart limits & "Other" bucket ──────────────────────────────
    # For multi-series, count unique x-axis values (not total rows)
    if color_by and color_by in agg_df.columns:
        total_unique = agg_df[group_col].nunique()
    else:
        total_unique = len(agg_df)
    applied_limit = request.limit
    chart_type = request.chart_type or "bar"

    radial_charts = {"pie", "doughnut", "polarArea", "radar"}
    if applied_limit is None:
        if chart_type in radial_charts and total_unique > 12:
            applied_limit = 12
        elif chart_type in {"bar", "horizontalBar"} and total_unique > 50:
            applied_limit = 50

    if applied_limit is not None and total_unique > applied_limit and not (color_by and color_by in agg_df.columns):
        if chart_type in radial_charts:
            sorted_df = agg_df.sort_values(by="agg_value", ascending=False)
            top_rows = sorted_df.head(applied_limit)
            other_rows = sorted_df.iloc[applied_limit:]
            other_sum = other_rows["agg_value"].sum()
            other_row = pd.DataFrame([{group_col: "Other", "agg_value": other_sum}])
            agg_df = pd.concat([top_rows, other_row], ignore_index=True)
        else:
            if sort_order == "asc":
                agg_df = agg_df.head(applied_limit)
            else:
                agg_df = agg_df.sort_values(by="agg_value", ascending=False).head(applied_limit)
                if x_is_date:
                    agg_df = agg_df.sort_values(by=group_col, ascending=True)

    # ── 8. Build multi-series data if color_by is set ─────────────────
    series_data = None
    color_by_values = None

    if color_by and color_by in agg_df.columns:
        # Get unique series names and x-axis labels
        color_by_values = sorted(agg_df[color_by].dropna().astype(str).unique().tolist())
        all_x_labels = sorted(agg_df[group_col].unique().tolist())

        # Apply limit to x-axis labels only (keep all series)
        if applied_limit and len(all_x_labels) > applied_limit:
            # For date axes, take the latest N
            if x_is_date:
                all_x_labels = all_x_labels[-applied_limit:]
            else:
                # For categorical, take top N by total aggregated value across all series
                x_totals = agg_df.groupby(group_col)["agg_value"].sum().sort_values(ascending=False)
                all_x_labels = x_totals.head(applied_limit).index.tolist()
            agg_df = agg_df[agg_df[group_col].isin(all_x_labels)]

        series_data = {}
        for series_name in color_by_values:
            series_df = agg_df[agg_df[color_by].astype(str) == series_name]
            series_dict = dict(zip(series_df[group_col].astype(str), series_df["agg_value"]))
            # Ensure all x labels are present (fill missing with 0)
            series_data[series_name] = [
                {group_col if group_col == x_col else x_col: str(x_label), "agg_value": series_dict.get(x_label, 0)}
                for x_label in all_x_labels
            ]

        # Also build flat result for backwards compatibility
        # Aggregate across all series for the flat view
        flat_df = agg_df.groupby(group_col)["agg_value"].sum().reset_index()
        if group_col != x_col:
            flat_df = flat_df.rename(columns={group_col: x_col})
        flat_df = flat_df.replace({np.nan: None})
        result_data = flat_df.to_dict(orient="records")
    else:
        # ── Single-series path (original) ─────────────────────────────
        if group_col != x_col:
            agg_df = agg_df.rename(columns={group_col: x_col})
        agg_df = agg_df.replace({np.nan: None})
        result_data = agg_df.to_dict(orient="records")

    # ── 9. Build response ─────────────────────────────────────────────
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
    color_sql = f", {color_by}" if color_by else ""
    gran_sql = f"DATE_TRUNC('{granularity}', {x_col})" if x_is_date and granularity else x_col
    sql_query = f"SELECT {gran_sql}{color_sql}, {request.aggregate.upper()}({request.y_axis}) AS agg_value FROM dataset"
    if request.range_min or request.range_max:
        clauses = []
        if request.range_min:
            clauses.append(f"{x_col} >= '{request.range_min}'")
        if request.range_max:
            clauses.append(f"{x_col} <= '{request.range_max}'")
        sql_query += " WHERE " + " AND ".join(clauses)
    sql_query += f" GROUP BY {gran_sql}{color_sql}"
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
        "dimension_values": _get_dimension_values(db_dataset, x_col),
        "series_data": series_data,
        "color_by_values": color_by_values,
    }

