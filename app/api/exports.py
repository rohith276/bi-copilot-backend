from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse, HTMLResponse
from sqlalchemy.orm import Session

from ..db.session import get_db
from ..services import dataset_service, export_service

router = APIRouter(prefix="/exports", tags=["exports"])


def _get_dataset_or_404(dataset_id: int, db: Session):
    db_dataset = (
        db.query(dataset_service.DatasetModel)
        .filter(dataset_service.DatasetModel.id == dataset_id)
        .first()
    )
    if db_dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return db_dataset


@router.get("/{dataset_id}/markdown")
def export_dashboard_markdown(dataset_id: int, db: Session = Depends(get_db)):
    _get_dataset_or_404(dataset_id, db)
    try:
        content = export_service.generate_markdown_export(dataset_id, db)
        return PlainTextResponse(
            content,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="dashboard_{dataset_id}.md"'},
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/{dataset_id}/slides")
def export_dashboard_slides(dataset_id: int, db: Session = Depends(get_db)):
    _get_dataset_or_404(dataset_id, db)
    try:
        html = export_service.generate_html_slides(dataset_id, db)
        return HTMLResponse(
            content=html,
            headers={"Content-Disposition": f'attachment; filename="dashboard_{dataset_id}_slides.html"'},
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
