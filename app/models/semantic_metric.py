from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime
from sqlalchemy.sql import func
from ..db.session import Base


class SemanticMetric(Base):
    __tablename__ = "semantic_metrics"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), index=True, nullable=False)
    name = Column(String, nullable=False)
    expression = Column(String, nullable=False)
    description = Column(String, nullable=True)
    synonyms = Column(Text, nullable=True)  # JSON array of alternate names
    created_at = Column(DateTime(timezone=True), server_default=func.now())
