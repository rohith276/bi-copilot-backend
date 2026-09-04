from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from .query import QueryResult

class ForecastRequest(BaseModel):
    date_col: str
    value_col: str
    periods: int = 30

class ForecastResult(BaseModel):
    dates: List[str]
    values: List[float]
    trend: str
    model_engine: Optional[str] = None
    r2_score: Optional[float] = None

class PredictionRequest(BaseModel):
    target_col: str
    feature_cols: List[str]

class PredictionResult(BaseModel):
    coefficients: Dict[str, float]
    intercept: float
    r2_score: float
    model_engine: Optional[str] = None

class RecommendationResults(BaseModel):
    product: str
    action: str
    reason: str
    priority: str

class AnomalyRequest(BaseModel):
    column: str

class AnomalyResult(BaseModel):
    row_index: int
    value: float
    z_score: float
    deviation: str
    other_data: Dict[str, str]

class VisualQueryRequest(BaseModel):
    x_axis: str
    y_axis: str
    aggregate: str
    filters: Optional[List[Dict[str, Any]]] = None
    date_granularity: Optional[str] = None      # "day" | "week" | "month" | "quarter" | "year"
    sort_order: Optional[str] = None            # "asc" | "desc" (by agg value)
    limit: Optional[int] = None                 # Top N / Bottom N
    range_min: Optional[str] = None             # Min value or date string for X-axis range
    range_max: Optional[str] = None             # Max value or date string for X-axis range
    chart_type: Optional[str] = None            # Hint for smart auto-limits

class VisualQueryResult(BaseModel):
    sql_query: str
    result: QueryResult
    x_column_type: Optional[str] = None         # "date" | "string" | "numeric"
    available_range: Optional[Dict[str, str]] = None  # {"min": "...", "max": "..."}
    applied_granularity: Optional[str] = None   # The granularity that was actually applied

