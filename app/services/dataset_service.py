from pathlib import Path
import pandas as pd
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session
from ..models.dataset import Dataset as DatasetModel
from .ai_cleaning_service import generate_cleaning_recipe, apply_ai_cleaning
from ..core.config import settings
from ..core.logger import get_logger
import uuid

logger = get_logger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BACKEND_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def _resolve_dataset_path(file_path: str | Path) -> Path:
    resolved = Path(file_path)
    if not resolved.is_absolute():
        resolved = BACKEND_DIR / resolved
    return resolved

def load_csv_with_fallback(file_path: str, nrows: int = None) -> pd.DataFrame:
    encodings = ['utf-8', 'iso-8859-1', 'cp1252', 'latin1']
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc, sep=None, engine='python', nrows=nrows)
        except UnicodeDecodeError:
            continue
    # If all fail, let pandas try its best with unicode_escape
    return pd.read_csv(file_path, encoding='unicode_escape', sep=None, engine='python', nrows=nrows)

async def save_upload_file(upload_file: UploadFile, db: Session):
    if not upload_file.filename:
        raise ValueError("Uploaded file is missing a filename.")

    # Generate unique filename
    file_extension = Path(upload_file.filename).suffix.lower()
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = UPLOAD_DIR / unique_filename

    # Check file size
    content = await upload_file.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {settings.MAX_UPLOAD_SIZE_MB}MB."
        )

    # Save file
    with file_path.open("wb") as buffer:
        buffer.write(content)

    # Load into pandas for basic validation and stats
    try:
        if file_extension.lower() == ".csv":
            df = load_csv_with_fallback(str(file_path))
        elif file_extension.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        else:
            # Clean up and raise error
            file_path.unlink(missing_ok=True)
            raise ValueError("Unsupported file format")

        # Automatically clean the dataset using AI
        logger.info(f"Triggering AI Cleaning for {upload_file.filename}")
        recipe = generate_cleaning_recipe(df)
        df_clean = apply_ai_cleaning(df, recipe)
        
        # Save the permanently cleaned version as CSV regardless of input format
        cleaned_file_path = file_path.with_suffix(".csv")
        df_clean.to_csv(cleaned_file_path, index=False)
        
        # If the original was not exactly the new CSV path, delete the raw file
        if str(file_path) != str(cleaned_file_path):
             file_path.unlink(missing_ok=True)

        row_count, col_count = df_clean.shape
        
        # Create DB record
        db_dataset = DatasetModel(
            filename=upload_file.filename,
            file_path=str(cleaned_file_path),
            file_type="csv", # Internally everything is now a clean CSV
            row_count=row_count,
            column_count=col_count
        )
        db.add(db_dataset)
        db.commit()
        db.refresh(db_dataset)
        
        return db_dataset
    except Exception as e:
        file_path.unlink(missing_ok=True)
        try:
             file_path.with_suffix(".csv").unlink(missing_ok=True)
        except Exception:
             pass
        raise e

def get_datasets(db: Session, skip: int = 0, limit: int = 100):
    return db.query(DatasetModel).offset(skip).limit(limit).all()

def delete_dataset(dataset_id: int, db: Session):
    db_dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not db_dataset:
        return False
    
    # Remove physical file if it exists
    dataset_path = _resolve_dataset_path(db_dataset.file_path)
    if dataset_path.exists():
        try:
            dataset_path.unlink()
        except Exception as e:
            logger.error(f"Error removing physical dataset: {e}")
            
    # Remove DB record
    db.delete(db_dataset)
    db.commit()
    return True

def get_dataset_df(file_path: str, nrows: int = None):
    resolved_path = _resolve_dataset_path(file_path)
    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset file missing from server storage: {resolved_path.name}")

    # Files are now strictly stored as pristine CSVs
    if resolved_path.suffix.lower() == ".csv":
        df = load_csv_with_fallback(str(resolved_path), nrows=nrows)
    else:
        df = pd.read_excel(resolved_path)
    return df # Return immediately: no more slow read-time cleaning necessary
