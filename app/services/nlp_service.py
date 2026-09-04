import os
import json
import sqlite3
import re
import pandas as pd
import numpy as np
from openai import OpenAI
from typing import List, Dict, Any, Optional
from ..schemas.query import NLQueryResult, QueryResult, ConversationTurn
from ..core.config import settings

DISALLOWED_SQL_PATTERN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|truncate|vacuum)\b",
    re.IGNORECASE,
)

def _clean_generated_sql(sql_query: str) -> str:
    cleaned_sql = sql_query.strip()
    if cleaned_sql.startswith("```sql"):
        cleaned_sql = cleaned_sql.replace("```sql", "").replace("```", "").strip()
    elif cleaned_sql.startswith("```"):
        cleaned_sql = cleaned_sql.replace("```", "").strip()

    cleaned_sql = cleaned_sql.rstrip(";").strip()
    if not cleaned_sql:
        raise ValueError("The AI did not generate a valid SQL query.")

    lowered = cleaned_sql.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise ValueError("The AI generated a non-read-only SQL statement.")
    if ";" in cleaned_sql or DISALLOWED_SQL_PATTERN.search(cleaned_sql):
        raise ValueError("The AI generated a SQL statement that is not allowed.")

    return cleaned_sql


def process_nl_query(df: pd.DataFrame, nl_query: str, conversation_history: Optional[List[ConversationTurn]] = None, semantic_context: Optional[str] = None) -> NLQueryResult:
    api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it to use the AI features.")
        
    client_args = {"api_key": api_key}
    model_name = "gpt-4o-mini"
    if api_key.startswith("sk-or-v1-"):
        client_args["base_url"] = "https://openrouter.ai/api/v1"
        model_name = "openai/gpt-4o-mini"
        
    client = OpenAI(**client_args)

    # 1. Prepare schema description
    schema_info = []
    date_hints = []
    
    for col, dtype in df.dtypes.items():
        schema_info.append(f"'{col}' ({dtype})")
        
        # Try to identify date columns and extract their temporal boundaries
        if 'date' in str(col).lower() or pd.api.types.is_datetime64_any_dtype(df[col]):
            try:
                temp_dates = pd.to_datetime(df[col], errors='coerce')
                min_date = temp_dates.min()
                max_date = temp_dates.max()
                if not pd.isna(min_date) and not pd.isna(max_date):
                    # Format as standard YYYY-MM-DD strings
                    date_hints.append(f"Column '{col}' spans from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
            except Exception:
                pass

    schema_str = ", ".join(schema_info)

    semantic_block = ""
    if semantic_context:
        semantic_block = f"\n\n{semantic_context}\n"

    
    date_context_str = ""
    if date_hints:
        date_context_str = "Temporal Dataset Context:\n" + "\n".join(date_hints)
        date_context_str += "\nCRITICAL RULE: This is a historical dataset. If the user asks for 'recent', 'last week/month/year', or 'current' trends, DO NOT use present-day functions like `date('now')` or `CURRENT_DATE`. You MUST filter relative to the historical maximum date in the dataset (e.g. `WHERE \"Order Date\" >= date((SELECT MAX(\"Order Date\") FROM dataset), '-6 weeks')`)."

    # 2. Get SQL query from LLM
    sql_prompt = f"""
    Given a SQLite table named 'dataset' with the following schema:
    {schema_str}
    {semantic_block}
    {date_context_str}

    Write a SQLite SQL query to answer this question: "{nl_query}"
    
    CRITICAL SQL RULES for SQLite compatibility:
    1. Standard Deviation: SQLite DOES NOT have `STDDEV`. To find anomalies (Z-scores), use this variance formula: `(SUM(x*x) - SUM(x)*SUM(x)/COUNT(x)) / (COUNT(x) - 1)`.
    2. To detect high-value anomalies, you can filter for values greater than `(AVG(x) + 2 * SQRT(variance))`.
    3. Column names with spaces MUST be wrapped in double quotes (e.g. "Order Date").
    4. Strings must use single quotes (e.g. 'Technology').
    5. Ensure all numeric outputs are rounded to 2 decimal places using `ROUND(val, 2)`.
    
    Return ONLY the SQL query, no markdown, no explanation.
    """
    
    # Build conversation context for follow-up queries
    messages = [
        {"role": "system", "content": "You are a data analysis assistant that converts natural language to SQLite SQL query. When the user references prior questions (e.g. 'break that down', 'filter that', 'now show...'), use the conversation history to understand context and modify or build upon previous SQL queries."}
    ]
    
    # Inject conversation history as prior turns
    if conversation_history:
        for turn in conversation_history[-5:]:  # Keep last 5 turns to stay within token limits
            messages.append({"role": "user", "content": f"Question: {turn.question}"})
            messages.append({"role": "assistant", "content": turn.sql})
    
    messages.append({"role": "user", "content": sql_prompt})
    
    response = client.chat.completions.create( # type: ignore
        model=model_name,
        messages=messages,
        temperature=0
    )
    sql_query = _clean_generated_sql(response.choices[0].message.content or "")

    # 3. Execute SQL against dataframe via SQLite memory db
    conn = sqlite3.connect(":memory:")
    df.to_sql("dataset", conn, index=False)
    
    try:
        result_df = pd.read_sql(sql_query, conn)
    except Exception as e:
        # AI Query Debugger (#7) - Attempt to fix failed SQL once
        error_msg = str(e)
        try:
            fix_prompt = f"""
            The following SQL query failed to execute against the SQLite database:
            
            {sql_query}
            
            The error message was:
            {error_msg}
            
            Please fix the SQL query so that it executes correctly. 
            Remember SQLite limitations (e.g. no STDDEV, use double quotes for column names with spaces).
            Return ONLY the fixed SQL query, no markdown, no explanation.
            """
            
            fix_resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a data analysis assistant that fixes SQL queries."},
                    {"role": "user", "content": fix_prompt}
                ],
                temperature=0
            )
            sql_query = _clean_generated_sql(fix_resp.choices[0].message.content or "")
            result_df = pd.read_sql(sql_query, conn)
        except Exception as retry_e:
            raise ValueError(f"Failed to execute the generated SQL even after AI debugging: {str(retry_e)}") from retry_e
    finally:
        if 'conn' in locals():
            try:
                conn.close()
            except:
                pass
        
    # 4. Generate Insights and Chart Configuration
    # Just grab first few rows to send back for context
    preview_data = result_df.head(5).to_dict(orient="records")
    insight_prompt = f"""
    The user asked: "{nl_query}"
    The SQL query executed was: {sql_query}
    The returned data result is: {preview_data}
    
    You are a Senior Business Intelligence Analyst.
    You must return a raw JSON object (without markdown code blocks, just the JSON string) exactly matching this structure:
    {{
        "insight": "Your highly accurate, analytical, and professional business insight. Do not just restate numbers. Analyze magnitude, provide actionable perspective, or explain business implications. Max 3 sentences. No SQL mentioned.",
        "chart_config": {{
            "type": "none", 
            "labelCol": "",
            "valueCol": ""
        }}
    }}
    
    Rules for chart_config:
    - If analyzing a trend over time, type is "line".
    - If analyzing categorical comparisons, type is "bar".
    - If analyzing percentage or shares of a whole, type is "pie".
    - If data is a single number or cannot be charted, type is "none".
    - labelCol must be the exact name of the X-axis column from the result set.
    - valueCol must be the exact name of the numeric Y-axis column from the result set.
    """
    insight_resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You output strict JSON configurations for a BI app."},
            {"role": "user", "content": insight_prompt}
        ],
        temperature=0.1
    )
    msg_content = insight_resp.choices[0].message.content
    raw_insight = msg_content.strip() if msg_content else ""
    
    # Clean up JSON if LLM added markdown
    if raw_insight.startswith("```json"):
         raw_insight = raw_insight.replace("```json", "").replace("```", "").strip()
    elif raw_insight.startswith("```"):
         raw_insight = raw_insight.replace("```", "").strip()
         
    try:
        parsed_insight = json.loads(raw_insight)
        insight_text = parsed_insight.get("insight", "Data generated successfully.")
        chart_config = parsed_insight.get("chart_config", None)
    except json.JSONDecodeError:
        insight_text = raw_insight
        chart_config = None

    # 5. Format result
    total_rows = len(result_df)
    
    # Handle NaN values and round floats for JSON serialization
    for col in result_df.select_dtypes(include='number').columns:
        result_df[col] = result_df[col].apply(lambda x: round(float(x), 2) if pd.notna(x) else np.nan)
    result_df = result_df.replace({np.nan: None})
    
    res = QueryResult(
        columns=result_df.columns.tolist(),
        data=result_df.to_dict(orient="records"), # type: ignore
        total_rows=total_rows
    )
    
    return NLQueryResult(
        sql_query=sql_query,
        insights=insight_text,
        chart_config=chart_config,
        result=res
    )

def suggest_queries(df: pd.DataFrame) -> List[str]:
    """
    Recommend 3 meaningful business questions based on the dataset schema.
    """
    api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ["What is the total revenue?", "How are sales trending over time?", "What are the top 5 products?"]
        
    client_args = {"api_key": api_key}
    model_name = "gpt-4o-mini"
    if api_key.startswith("sk-or-v1-"):
        client_args["base_url"] = "https://openrouter.ai/api/v1"
        model_name = "openai/gpt-4o-mini"
        
    client = OpenAI(**client_args)
    
    schema_info = [f"'{col}' ({dtype})" for col, dtype in df.dtypes.items()]
    schema_str = ", ".join(schema_info)
    
    prompt = f"""
    Given a dataset schema: {schema_str}
    Suggest 3 high-value, professional business questions a user might ask an AI to get insights.
    Return ONLY the 3 questions as a valid JSON list of strings [\"q1\", \"q2\", \"q3\"].
    No markdown, no numbers, just the JSON list.
    """
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a BI expert helping users explore data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        msg_content = response.choices[0].message.content
        content = msg_content.strip() if msg_content else ""
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception:
        return ["What is the total value by category?", "Show me the trend over time", "Who are the top performers?"]

