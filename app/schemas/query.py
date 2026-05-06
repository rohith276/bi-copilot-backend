from pydantic import BaseModel
from typing import List, Optional, Any, Dict

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

class QueryResult(BaseModel):
    columns: List[str]
    data: List[Dict[str, Any]]
    total_rows: int

class NLQueryRequest(BaseModel):
    query: str

class ChartConfig(BaseModel):
    type: str # 'bar', 'line', 'pie', 'none'
    labelCol: str
    valueCol: str

class NLQueryResult(BaseModel):
    sql_query: str
    insights: str
    chart_config: Optional[ChartConfig] = None
    result: QueryResult
