from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import or_
from typing import List, Dict, Any
from ..db.session import get_db
from ..models.dataset import Dataset
from ..models.dashboard import DashboardItem
from pydantic import BaseModel

router = APIRouter(prefix="/search", tags=["search"])

class SearchResult(BaseModel):
    id: int
    title: str
    type: str  # 'dataset' or 'dashboard'
    dataset_id: int # To navigate to the correct URL
    description: str

@router.get("/", response_model=List[SearchResult])
def global_search(q: str, db: Session = Depends(get_db)):
    """Search across datasets and dashboard items."""
    results = []
    
    if not q or len(q) < 2:
        return results

    search_pattern = f"%{q}%"

    # Search datasets
    datasets = db.query(Dataset).filter(
        Dataset.filename.ilike(search_pattern)
    ).limit(10).all()
    
    for ds in datasets:
        results.append(SearchResult(
            id=int(ds.id), # type: ignore
            title=str(ds.filename),
            type="dataset",
            dataset_id=int(ds.id), # type: ignore
            description=f"Dataset with {ds.row_count} rows and {ds.column_count} columns"
        ))

    # Search dashboard items
    dashboard_items = db.query(DashboardItem).filter(
        DashboardItem.title.ilike(search_pattern)
    ).limit(10).all()
    
    for item in dashboard_items:
        # Get the parent dataset name for context
        ds = db.query(Dataset).filter(Dataset.id == item.dataset_id).first()
        ds_name = ds.filename if ds else "Unknown Dataset"
        
        results.append(SearchResult(
            id=int(item.id), # type: ignore
            title=str(item.title),
            type="dashboard",
            dataset_id=int(item.dataset_id), # type: ignore
            description=f"Dashboard pinned item in {ds_name}"
        ))

    return results
