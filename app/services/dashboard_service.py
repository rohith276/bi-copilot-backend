import json
import uuid as uuid_lib
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from ..models.dashboard import DashboardItem, SharedDashboard


def _serialize_item(item: DashboardItem) -> Dict[str, Any]:
    return {
        "id": item.id,
        "title": item.title,
        "sql_query": item.sql_query,
        "chart_config": json.loads(item.chart_config),
        "layout": json.loads(item.layout),
    }


def get_dashboard_items(dataset_id: int, db: Session) -> List[Dict[str, Any]]:
    items = (
        db.query(DashboardItem)
        .filter(DashboardItem.dataset_id == dataset_id)
        .order_by(DashboardItem.id.asc())
        .all()
    )
    return [_serialize_item(item) for item in items]


def pin_item(
    dataset_id: int,
    title: str,
    sql_query: str,
    chart_config: Dict[str, Any],
    layout: Dict[str, Any],
    db: Session,
) -> Dict[str, Any]:
    item = DashboardItem(
        dataset_id=dataset_id,
        title=title,
        sql_query=sql_query,
        chart_config=json.dumps(chart_config),
        layout=json.dumps(layout),
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return _serialize_item(item)


def delete_item(dataset_id: int, item_id: int, db: Session) -> bool:
    item = (
        db.query(DashboardItem)
        .filter(DashboardItem.id == item_id, DashboardItem.dataset_id == dataset_id)
        .first()
    )
    if not item:
        return False
    db.delete(item)
    db.commit()
    return True


def create_share_link(dataset_id: int, db: Session) -> str:
    share_uuid = str(uuid_lib.uuid4())
    shared = SharedDashboard(uuid=share_uuid, dataset_id=dataset_id)
    db.add(shared)
    db.commit()
    return share_uuid


def get_shared_dashboard(share_uuid: str, db: Session, execute_sql_fn) -> Optional[Dict[str, Any]]:
    """Return shared dashboard with pre-executed chart data."""
    from ..services import dataset_service

    shared = db.query(SharedDashboard).filter(SharedDashboard.uuid == share_uuid).first()
    if not shared:
        return None

    db_dataset = (
        db.query(dataset_service.DatasetModel)
        .filter(dataset_service.DatasetModel.id == shared.dataset_id)
        .first()
    )
    if not db_dataset:
        return None

    df = dataset_service.get_dataset_df(str(db_dataset.file_path))
    items = get_dashboard_items(shared.dataset_id, db)

    enriched_items = []
    for item in items:
        try:
            result = execute_sql_fn(df, item["sql_query"], limit=500)
            item_with_data = {**item, "data": result["data"]}
        except Exception:
            item_with_data = {**item, "data": []}
        enriched_items.append(item_with_data)

    return {
        "dataset": {
            "id": db_dataset.id,
            "filename": db_dataset.filename,
        },
        "items": enriched_items,
    }
