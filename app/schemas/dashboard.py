from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class DashboardItemCreate(BaseModel):
    title: str
    sql_query: str
    chart_config: Dict[str, Any]
    layout: Dict[str, Any]


class DashboardItemOut(BaseModel):
    id: int
    title: str
    sql_query: str
    chart_config: Dict[str, Any]
    layout: Dict[str, Any]

    class Config:
        from_attributes = True


class AutoDashboardResult(BaseModel):
    summary: str
    items: List[DashboardItemOut]
    pinned_count: int
