from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class DatasetBase(BaseModel):
    filename: str
    file_type: str

class DatasetCreate(DatasetBase):
    file_path: str
    row_count: Optional[int] = None
    column_count: Optional[int] = None

class Dataset(DatasetBase):
    id: int
    created_at: datetime
    row_count: Optional[int]
    column_count: Optional[int]

    class Config:
        from_attributes = True
