from datetime import datetime, timezone
from html import escape
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from . import dashboard_service, analysis_service, report_service, dataset_service


def generate_markdown_export(dataset_id: int, db: Session) -> str:
    db_dataset = (
        db.query(dataset_service.DatasetModel)
        .filter(dataset_service.DatasetModel.id == dataset_id)
        .first()
    )
    if not db_dataset:
        raise ValueError("Dataset not found")

    nrows = 100000 if db_dataset.row_count > 100000 else None
    df = dataset_service.get_dataset_df(str(db_dataset.file_path), nrows=nrows)
    report = report_service.generate_report(df, db_dataset)
    items = dashboard_service.get_dashboard_items(dataset_id, db)

    lines = [
        f"# Executive Dashboard — {db_dataset.filename}",
        f"",
        f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        f"",
        f"## Data Quality",
        f"- Completeness: {report['data_quality']['completeness_pct']}%",
        f"- Quality Score: {report['data_quality']['quality_score']}",
        f"",
        f"## Top Insights",
    ]

    for insight in report.get("top_insights", [])[:5]:
        lines.append(f"- **{insight.get('title', 'Insight')}**: {insight.get('body', '')}")

    lines.extend(["", "## Pinned Dashboard Charts", ""])

    for i, item in enumerate(items, 1):
        lines.append(f"### {i}. {item['title']}")
        lines.append(f"- Chart type: `{item['chart_config'].get('type', 'bar')}`")
        lines.append(f"- SQL: `{item['sql_query']}`")
        try:
            result = analysis_service.execute_sql_query(df, item["sql_query"], limit=10)
            if result["data"]:
                cols = result["columns"]
                lines.append("")
                lines.append("| " + " | ".join(cols) + " |")
                lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
                for row in result["data"][:5]:
                    lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
        except Exception:
            lines.append("*Could not execute query*")
        lines.append("")

    return "\n".join(lines)


def generate_html_slides(dataset_id: int, db: Session) -> str:
    db_dataset = (
        db.query(dataset_service.DatasetModel)
        .filter(dataset_service.DatasetModel.id == dataset_id)
        .first()
    )
    if not db_dataset:
        raise ValueError("Dataset not found")

    nrows = 100000 if db_dataset.row_count > 100000 else None
    df = dataset_service.get_dataset_df(str(db_dataset.file_path), nrows=nrows)
    report = report_service.generate_report(df, db_dataset)
    items = dashboard_service.get_dashboard_items(dataset_id, db)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    slides: List[str] = []

    # Title slide
    slides.append(_slide_html(
        f"<h1>{escape(db_dataset.filename)}</h1>"
        f"<p class='subtitle'>Executive Dashboard Presentation</p>"
        f"<p class='meta'>{escape(generated_at)}</p>",
        "title-slide",
    ))

    # Quality slide
    dq = report["data_quality"]
    quality_body = (
        f"<ul>"
        f"<li>Completeness: <strong>{dq['completeness_pct']}%</strong></li>"
        f"<li>Quality Score: <strong>{dq['quality_score']}</strong></li>"
        f"<li>Rows: <strong>{report['dataset']['rows']:,}</strong></li>"
        f"</ul>"
    )
    slides.append(_slide_html("<h2>Data Quality Overview</h2>" + quality_body))

    # Insights slide
    insight_items = "".join(
        f"<li><strong>{escape(i.get('title', ''))}</strong> — {escape(i.get('body', ''))}</li>"
        for i in report.get("top_insights", [])[:4]
    )
    slides.append(_slide_html(f"<h2>Key Insights</h2><ul>{insight_items}</ul>"))

    # One slide per dashboard chart
    for item in items:
        chart_type = item["chart_config"].get("type", "bar")
        table_html = _render_data_table(df, item)
        slides.append(_slide_html(
            f"<h2>{escape(item['title'])}</h2>"
            f"<p class='meta'>Chart: {escape(chart_type.upper())}</p>"
            f"{table_html}",
        ))

    # Closing slide
    slides.append(_slide_html(
        "<h2>Summary</h2>"
        f"<p>{len(items)} pinned visuals · {len(report.get('top_insights', []))} auto-insights</p>"
        "<p class='subtitle'>Powered by BI Copilot</p>",
        "title-slide",
    ))

    return _html_presentation_wrapper(db_dataset.filename, slides)


def _slide_html(body: str, extra_class: str = "") -> str:
    cls = f"slide {extra_class}".strip()
    return f'<section class="{cls}"><div class="slide-content">{body}</div></section>'


def _render_data_table(df, item: Dict[str, Any]) -> str:
    try:
        result = analysis_service.execute_sql_query(df, item["sql_query"], limit=8)
        if not result["data"]:
            return "<p><em>No data returned</em></p>"
        cols = result["columns"]
        header = "".join(f"<th>{escape(str(c))}</th>" for c in cols)
        rows = ""
        for row in result["data"]:
            cells = "".join(f"<td>{escape(str(row.get(c, '')))}</td>" for c in cols)
            rows += f"<tr>{cells}</tr>"
        return f"<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>"
    except Exception:
        return "<p><em>Could not load chart data</em></p>"


def _html_presentation_wrapper(title: str, slides: List[str]) -> str:
    slide_markup = "\n".join(slides)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{escape(title)} — BI Copilot Presentation</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #0b0f17; color: #f3f4f6; }}
  .deck {{ max-width: 960px; margin: 0 auto; padding: 2rem; }}
  .slide {{
    background: #111827; border: 1px solid #1f2937; border-radius: 8px;
    padding: 3rem; margin-bottom: 2rem; min-height: 400px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    page-break-after: always;
  }}
  .title-slide {{ text-align: center; display: flex; align-items: center; justify-content: center; }}
  .slide-content {{ width: 100%; }}
  h1 {{ font-size: 2rem; font-weight: 800; margin-bottom: 0.5rem; }}
  h2 {{ font-size: 1.25rem; font-weight: 700; margin-bottom: 1.5rem; color: #93c5fd; text-transform: uppercase; letter-spacing: 0.05em; }}
  .subtitle {{ color: #9ca3af; font-size: 1rem; margin-top: 0.5rem; }}
  .meta {{ color: #6b7280; font-size: 0.875rem; margin-top: 1rem; font-family: ui-monospace, monospace; }}
  ul {{ list-style: none; }}
  ul li {{ padding: 0.5rem 0; border-bottom: 1px solid #1f2937; font-size: 0.9rem; line-height: 1.5; }}
  table {{ width: 100%; border-collapse: collapse; font-family: ui-monospace, monospace; font-size: 0.8rem; margin-top: 1rem; }}
  th, td {{ border: 1px solid #1f2937; padding: 0.5rem 0.75rem; text-align: left; }}
  th {{ background: #1f2937; color: #93c5fd; font-weight: 700; }}
  tr:nth-child(even) {{ background: #0b0f17; }}
  @media print {{
    body {{ background: white; color: black; }}
    .slide {{ border: 1px solid #ccc; box-shadow: none; }}
  }}
</style>
</head>
<body>
<div class="deck">
{slide_markup}
</div>
</body>
</html>"""
