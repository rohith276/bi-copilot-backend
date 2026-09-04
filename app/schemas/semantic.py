from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class CalculatedFieldCreate(BaseModel):
    name: str
    expression: str
    description: Optional[str] = None


class CalculatedFieldOut(BaseModel):
    id: int
    dataset_id: int
    name: str
    expression: str
    description: Optional[str] = None

    class Config:
        from_attributes = True


class GenerateFormulaRequest(BaseModel):
    prompt: str


class GenerateFormulaResponse(BaseModel):
    formula: str


class RootCauseRequest(BaseModel):
    metric_col: Optional[str] = None
    question: Optional[str] = None


class RootCauseBreakdown(BaseModel):
    dimension: str
    total_delta: float
    top_movers: List[Dict[str, Any]]


class RootCauseResult(BaseModel):
    metric: str
    period_label: str
    prior_total: float
    recent_total: float
    delta: float
    delta_pct: float
    direction: str
    volume_effect: Optional[float] = None
    rate_effect: Optional[float] = None
    prior_transactions: Optional[int] = None
    recent_transactions: Optional[int] = None
    prior_avg_ticket: Optional[float] = None
    recent_avg_ticket: Optional[float] = None
    narrative: str
    breakdowns: List[RootCauseBreakdown]
    chart_config: Optional[Dict[str, Any]] = None
