from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from ..db.session import get_db
from ..schemas.dashboard import DashboardItemCreate, DashboardItemOut, AutoDashboardResult
from ..services import dataset_service, dashboard_service, auto_dashboard_service, analysis_service

router = APIRouter(prefix="/dashboards", tags=["dashboards"])
shared_router = APIRouter(prefix="/shared", tags=["shared"])


def _get_dataset_or_404(dataset_id: int, db: Session):
    db_dataset = (
        db.query(dataset_service.DatasetModel)
        .filter(dataset_service.DatasetModel.id == dataset_id)
        .first()
    )
    if db_dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return db_dataset


@router.get("/{dataset_id}", response_model=List[DashboardItemOut])
def get_dashboard(dataset_id: int, db: Session = Depends(get_db)):
    _get_dataset_or_404(dataset_id, db)
    return dashboard_service.get_dashboard_items(dataset_id, db)


@router.post("/{dataset_id}/pin", response_model=DashboardItemOut)
def pin_dashboard_item(
    dataset_id: int,
    item: DashboardItemCreate,
    db: Session = Depends(get_db),
):
    _get_dataset_or_404(dataset_id, db)
    return dashboard_service.pin_item(
        dataset_id,
        item.title,
        item.sql_query,
        item.chart_config,
        item.layout,
        db,
    )


@router.delete("/{dataset_id}/items/{item_id}")
def delete_dashboard_item(dataset_id: int, item_id: int, db: Session = Depends(get_db)):
    if not dashboard_service.delete_item(dataset_id, item_id, db):
        raise HTTPException(status_code=404, detail="Dashboard item not found")
    return {"message": "Item removed"}


@router.post("/{dataset_id}/share")
def share_dashboard(dataset_id: int, db: Session = Depends(get_db)):
    _get_dataset_or_404(dataset_id, db)
    share_uuid = dashboard_service.create_share_link(dataset_id, db)
    return {"url": f"/shared/{share_uuid}"}


@shared_router.get("/{share_uuid}")
def get_shared_dashboard(share_uuid: str, db: Session = Depends(get_db)):
    result = dashboard_service.get_shared_dashboard(
        share_uuid,
        db,
        execute_sql_fn=analysis_service.execute_sql_query,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Shared dashboard not found")
    return result
