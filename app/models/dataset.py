from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from ..db.session import Base

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    file_path = Column(String)
    file_type = Column(String)  # csv, excel
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    row_count = Column(Integer, nullable=True)
    column_count = Column(Integer, nullable=True)
    
    # DB Connector fields
    source_type = Column(String, default="file") # 'file' or 'database'
    connection_string = Column(String, nullable=True)
    db_query = Column(String, nullable=True)
