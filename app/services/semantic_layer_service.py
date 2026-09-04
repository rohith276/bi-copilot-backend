import json
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

import os
from openai import OpenAI
from ..models.semantic_metric import SemanticMetric
from ..models.dataset import Dataset
from ..core.config import settings
from . import dataset_service


def get_metrics(dataset_id: int, db: Session) -> List[Dict[str, Any]]:
    items = (
        db.query(SemanticMetric)
        .filter(SemanticMetric.dataset_id == dataset_id)
        .order_by(SemanticMetric.id.asc())
        .all()
    )
    return [_serialize(m) for m in items]


def create_metric(
    dataset_id: int,
    name: str,
    expression: str,
    description: Optional[str],
    db: Session,
) -> Dict[str, Any]:
    metric = SemanticMetric(
        dataset_id=dataset_id,
        name=name.strip(),
        expression=expression.strip(),
        description=description,
    )
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return _serialize(metric)


def delete_metric(dataset_id: int, metric_id: int, db: Session) -> bool:
    metric = (
        db.query(SemanticMetric)
        .filter(SemanticMetric.id == metric_id, SemanticMetric.dataset_id == dataset_id)
        .first()
    )
    if not metric:
        return False
    db.delete(metric)
    db.commit()
    return True


def format_for_prompt(dataset_id: int, db: Session) -> str:
    """Format business metric definitions for LLM prompt injection."""
    metrics = get_metrics(dataset_id, db)
    if not metrics:
        return ""

    lines = ["Business Metric Definitions (use these when the user references these terms):"]
    for m in metrics:
        line = f'- "{m["name"]}" = {m["expression"]}'
        if m.get("description"):
            line += f"  ({m['description']})"
        lines.append(line)
    lines.append(
        "When a user asks about a defined metric, translate it to the expression above in your SQL."
    )
    return "\n".join(lines)


def _serialize(metric: SemanticMetric) -> Dict[str, Any]:
    return {
        "id": metric.id,
        "dataset_id": metric.dataset_id,
        "name": metric.name,
        "expression": metric.expression,
        "description": metric.description,
    }

def generate_measure_formula(dataset_id: int, prompt: str, db: Session) -> str:
    """Generate a SQL/Pandas metric formula using AI based on the dataset schema."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise ValueError("Dataset not found")
        
    df = dataset_service.get_dataset_df(str(dataset.file_path))
    schema = df.dtypes.to_dict()
    schema_str = "\n".join([f"- {col}: {dtype}" for col, dtype in schema.items()])
    
    api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
        
    client_args = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_args["base_url"] = base_url
        
    client = OpenAI(**client_args) # type: ignore
    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    
    system_prompt = """
    You are an expert BI data architect. The user wants to create a calculated measure.
    You will be provided with the schema of the table.
    Return ONLY the raw mathematical or SQL expression. Do NOT include markdown formatting or explanations.
    Examples:
    User: Profit Margin
    You: (Revenue - Cost) / Revenue
    
    User: Total Sales
    You: Quantity * UnitPrice
    """
    
    user_prompt = f"Schema:\n{schema_str}\n\nUser request: {prompt}"
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    formula = (response.choices[0].message.content or "").strip()
    # Clean markdown if present
    if formula.startswith("```"):
        lines = formula.split("\n")
        if len(lines) > 2:
            formula = "\n".join(lines[1:-1])
        else:
            formula = formula.replace("```", "").replace("sql", "")
            
    return formula.strip()
