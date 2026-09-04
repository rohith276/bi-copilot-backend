from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from .query import QueryResult

class ForecastRequest(BaseModel):
    date_col: str
    value_col: str
    periods: int = 30
    scenario_multiplier: Optional[float] = 1.0

class ForecastResult(BaseModel):
    dates: List[str]
    values: List[float]
    trend: str
    model_engine: Optional[str] = None
    r2_score: Optional[float] = None
    confidence: Optional[float] = None
    frequency: Optional[str] = None
    lower_bounds: Optional[List[float]] = None
    upper_bounds: Optional[List[float]] = None
    test_mape: Optional[float] = None
    base_values: Optional[List[float]] = None
    scenario_values: Optional[List[float]] = None

class PredictionRequest(BaseModel):
    target_col: str
    feature_cols: List[str]

class PredictionResult(BaseModel):
    coefficients: Dict[str, float]
    intercept: float
    r2_score: float
    adjusted_r2: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    model_engine: Optional[str] = None

class RecommendationResults(BaseModel):
    product: str
    action: str
    reason: str
    priority: str
    category: Optional[str] = None
    velocity: Optional[float] = None
    recommended_units: Optional[float] = None

class AnomalyRequest(BaseModel):
    column: str

class AnomalyResult(BaseModel):
    row_index: int
    value: float
    z_score: float
    deviation: str
    severity_score: Optional[float] = None
    anomaly_type: Optional[str] = None
    other_data: Dict[str, str]

class AnomalyResponse(BaseModel):
    anomalies: List[AnomalyResult]
    explanation: Optional[str] = None

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
    dimension_filters: Optional[Dict[str, List[str]]] = None  # {"Region": ["North", "East"]}
    color_by: Optional[str] = None               # Legend dimension for stacked/grouped series
    stack_mode: Optional[str] = None             # "stacked" | "grouped" (default: stacked)

class VisualQueryResult(BaseModel):
    sql_query: str
    result: QueryResult
    x_column_type: Optional[str] = None         # "date" | "string" | "numeric"
    available_range: Optional[Dict[str, str]] = None  # {"min": "...", "max": "..."}
    applied_granularity: Optional[str] = None   # The granularity that was actually applied
    dimension_values: Optional[Dict[str, List[str]]] = None  # Unique values per dimension for filter UI
    series_data: Optional[Dict[str, list]] = None  # Multi-series: {"Bars": [...], "Bites": [...]}
    color_by_values: Optional[List[str]] = None    # Unique values of color_by column

