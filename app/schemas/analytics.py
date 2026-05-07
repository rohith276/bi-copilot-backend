from pydantic import BaseModel
from typing import List, Dict, Any, Optional

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

