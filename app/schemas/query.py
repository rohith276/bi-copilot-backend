from pydantic import BaseModel
from typing import List, Optional, Any, Dict

class ConversationTurn(BaseModel):
    """A single Q&A turn in the conversation history."""
    question: str
    sql: str
    insight: str

class Filter(BaseModel):
    column: str
    operator: str  # eq, ne, gt, lt, contains
    value: Any

class GroupBy(BaseModel):
    columns: List[str]
    agg_funcs: Dict[str, str]  # column_name: func (sum, mean, count)

class QueryRequest(BaseModel):
    filters: Optional[List[Filter]] = None
    group_by: Optional[GroupBy] = None
    sort_by: Optional[str] = None
    sort_desc: bool = False
    limit: int = 100
    sql_query: Optional[str] = None

class QueryResult(BaseModel):
    columns: List[str]
    data: List[Dict[str, Any]]
    total_rows: int

class NLQueryRequest(BaseModel):
    query: str
    conversation_history: Optional[List[ConversationTurn]] = None

class ChartConfig(BaseModel):
    type: str # 'bar', 'line', 'pie', 'none'
    labelCol: str
    valueCol: str

class NLQueryResult(BaseModel):
    sql_query: str
    insights: str
    chart_config: Optional[ChartConfig] = None
    result: QueryResult

class JoinSuggestionRequest(BaseModel):
    left_dataset_id: int
    right_dataset_id: int

class JoinSuggestion(BaseModel):
    left_col: str
    right_col: str
    confidence: float
    reason: str

class JoinSuggestionResponse(BaseModel):
    suggestions: List[JoinSuggestion]
