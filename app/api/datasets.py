from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from ..db.session import get_db
from ..schemas.dataset import Dataset
from ..schemas.query import QueryRequest, QueryResult, NLQueryRequest, NLQueryResult
from ..services import dataset_service, analysis_service, cleaning_service, nlp_service, sample_data_service

router = APIRouter(prefix="/datasets", tags=["datasets"])

@router.post("/seed", response_model=Dataset)
def seed_dataset(db: Session = Depends(get_db)):
    """Seed a sample dataset for testing."""
    try:
        file_path = sample_data_service.generate_sample_sales_data(row_count=500)
        
        # Read file to get stats
        df = dataset_service.get_dataset_df(file_path)
        
        # Create DB record using the correct model class from dataset_service
        db_dataset = dataset_service.DatasetModel(
            filename="Sample Sales Data.csv",
            file_path=file_path,
            file_type="csv",
            row_count=len(df),
            column_count=len(df.columns)
        )
        db.add(db_dataset)
        db.commit()
        db.refresh(db_dataset)
        return db_dataset

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload", response_model=Dataset)
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        return await dataset_service.save_upload_file(file, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not upload file: {str(e)}")

@router.get("/", response_model=List[Dataset])
def read_datasets(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return dataset_service.get_datasets(db, skip=skip, limit=limit)

def _get_dataset_or_404(dataset_id: int, db: Session):
    """Internal helper — not an endpoint. Looks up a dataset or raises 404."""
    db_dataset = db.query(dataset_service.DatasetModel).filter(dataset_service.DatasetModel.id == dataset_id).first()
    if db_dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return db_dataset

@router.get("/{dataset_id}", response_model=Dataset)
def read_dataset(dataset_id: int, db: Session = Depends(get_db)):
    return _get_dataset_or_404(dataset_id, db)

@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    success = dataset_service.delete_dataset(dataset_id, db)
    if not success:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"message": "Dataset successfully deleted"}

@router.get("/{dataset_id}/preview", response_model=QueryResult)
def get_dataset_preview(dataset_id: int, db: Session = Depends(get_db)):
    db_dataset = _get_dataset_or_404(dataset_id, db)
    # Optimization: Only load first 100 rows for preview
    df = dataset_service.get_dataset_df(db_dataset.file_path, nrows=100)
    return analysis_service.process_query(df, QueryRequest(limit=10))

@router.get("/{dataset_id}/stats")
def get_dataset_stats(dataset_id: int, db: Session = Depends(get_db)):
    db_dataset = _get_dataset_or_404(dataset_id, db)
    
    # Optimization: Use sampling for massive datasets to prevent timeouts
    nrows = 100000 if db_dataset.row_count > 100000 else None
    df = dataset_service.get_dataset_df(db_dataset.file_path, nrows=nrows)
    
    return cleaning_service.get_column_stats(df)

@router.post("/{dataset_id}/query", response_model=QueryResult)
def query_dataset(dataset_id: int, request: QueryRequest, db: Session = Depends(get_db)):
    db_dataset = _get_dataset_or_404(dataset_id, db)
    df = dataset_service.get_dataset_df(db_dataset.file_path)
    return analysis_service.process_query(df, request)

@router.post("/{dataset_id}/nl-query", response_model=NLQueryResult)
def nl_query_dataset(dataset_id: int, request: NLQueryRequest, db: Session = Depends(get_db)):
    db_dataset = _get_dataset_or_404(dataset_id, db)
    df = dataset_service.get_dataset_df(db_dataset.file_path)
    try:
        return nlp_service.process_nl_query(df, request.query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process NL query: {str(e)}")

@router.get("/{dataset_id}/suggest-queries", response_model=List[str])
def suggest_dataset_queries(dataset_id: int, db: Session = Depends(get_db)):
    db_dataset = _get_dataset_or_404(dataset_id, db)
    df = dataset_service.get_dataset_df(db_dataset.file_path)
    return nlp_service.suggest_queries(df)
