from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.sql import func
from ..db.session import Base


class DashboardItem(Base):
    __tablename__ = "dashboard_items"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), index=True, nullable=False)
    title = Column(String, nullable=False)
    sql_query = Column(Text, nullable=False)
    chart_config = Column(Text, nullable=False)  # JSON string
    layout = Column(Text, nullable=False)  # JSON string
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class SharedDashboard(Base):
    __tablename__ = "shared_dashboards"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String, unique=True, index=True, nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
