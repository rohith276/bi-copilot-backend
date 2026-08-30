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
    db_dataset = _get_dataset_or_404(dataset_id, db)
    df = dataset_service.get_dataset_df(str(db_dataset.file_path))
    
    q_filters = []
    if request.filters:
        for f in request.filters:
            q_filters.append(Filter(column=f["column"], operator="eq", value=f["value"]))
            
    q_req = QueryRequest(
        filters=q_filters if q_filters else None,
        group_by=GroupBy(columns=[request.x_axis], agg_funcs={request.y_axis: request.aggregate}),
        limit=1000
    )
    
    result = analysis_service.process_query(df, q_req)
    
    # Rename the aggregated column to 'agg_value' for the frontend
    original_agg_col = f"{request.y_axis}_{request.aggregate.lower()}"
    if original_agg_col not in result["columns"] and request.y_axis in result["columns"]:
        original_agg_col = request.y_axis
        
    new_columns = [col if col != original_agg_col else "agg_value" for col in result["columns"]]
    new_data = []
    for row in result["data"]:
        new_row = {}
        for k, v in row.items():
            if k == original_agg_col:
                new_row["agg_value"] = v
            else:
                new_row[k] = v
        new_data.append(new_row)
        
    # Generate fake SQL query for dashboard pinning
    sql_query = f"SELECT {request.x_axis}, {request.aggregate.upper()}({request.y_axis}) AS agg_value FROM dataset GROUP BY {request.x_axis}"
    
    return {
        "sql_query": sql_query,
        "result": {
            "columns": new_columns,
            "data": new_data,
            "total_rows": result["total_rows"]
        }
    }

