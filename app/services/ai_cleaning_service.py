import os
import json
import pandas as pd
import numpy as np
from openai import OpenAI
from ..core.config import settings
from ..core.logger import get_logger

logger = get_logger(__name__)

def _get_llm_client():
    api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, None
        
    client_args = {"api_key": api_key}
    model_name = "gpt-4o-mini"
    if api_key.startswith("sk-or-v1-"):
        client_args["base_url"] = "https://openrouter.ai/api/v1"
        model_name = "openai/gpt-4o-mini"
        
    return OpenAI(**client_args), model_name


def generate_cleaning_recipe(df_sample: pd.DataFrame) -> dict:
    """
    Analyzes a sample of the dataset and uses AI to generate a definitive cleaning plan.
    """
    client, model_name = _get_llm_client()
    if not client:
        return {} # Fallback to no cleaning if no API key
        
    # Serialize the first 25 rows for the LLM
    preview_data = df_sample.head(25).to_dict(orient="records")
    
    # Send column names and dtypes for context
    schema_info = [f"'{col}' ({dtype})" for col, dtype in df_sample.dtypes.items()]
    
    # Problem 5 Fix: Prevent token limit explosion on massive horizontal schemas
    if len(schema_info) > 60:
        schema_info = schema_info[:60]
        schema_info.append("... [TRUNCATED DUE TO SIZE - FOCUS ON THESE 60 COLUMNS]")
        preview_data = [{k: v for i, (k, v) in enumerate(row.items()) if i < 60} for row in preview_data]
    
    schema_str = ", ".join(schema_info)
    
    prompt = f"""
    You are an expert Data Engineer. I am providing you with a schema and a sample of a raw dataset.
    Your job is to identify explicitly how to sanitize this data for machine learning and BI analysis.
    
    Schema: {schema_str}
    
    Sample Data:
    {json.dumps(preview_data, default=str)}
    
    Return a strict JSON object (NO markdown formatting, just the raw JSON) with the following structure:
    {{
        "date_columns": ["col1", "col2"], // Columns that are clearly dates/timestamps
        "numeric_columns": [
            {{"name": "col3", "strip_currency": true}}, // Use true if the column contains $, €, £, or commas in numbers
            {{"name": "col4", "strip_currency": false}}
        ],
        "categorical_columns": ["col5", "col6"], // Columns that are text categories
        "columns_to_drop": ["Unnamed: 0", "Garbage"] // Columns with nothing but NaNs, unnamed garbage, or useless index duplicates
    }}
    
    Be extremely accurate. If a column is already a proper float/int, strip_currency is false. If it's a string like '$1,000.50', it is a numeric column and strip_currency is true. If it says 'null', it doesn't need dropping unless the entire column is null.
    """
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You output only valid JSON. No markdown blocks like ```json."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=800
        )
        
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
            
        recipe = json.loads(content)
        return recipe
    except Exception as e:
        logger.error(f"AI Cleaning Recipe Generation Failed: {e}")
        return {}


def apply_ai_cleaning(df: pd.DataFrame, recipe: dict) -> pd.DataFrame:
    """
    Applies the AI-generated cleaning recipe to the full Pandas DataFrame.
    """
    df_clean = df.copy()

    # Problem 4 Fix: Deterministic fallback if API fails
    if not recipe:
        logger.warning("AI Sandbox offline/failed. Falling back to local heuristic cleaner.")
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object' or df_clean[col].dtype.name == 'string':
                sample = df_clean[col].dropna().astype(str).head(50)
                if any(sample.str.contains(r'[$£€]', regex=True)):
                    df_clean[col] = df_clean[col].astype(str).str.replace(r'[$,£€]', '', regex=True)
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
                elif 'date' in str(col).lower() or pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                    try:
                        temp = pd.to_datetime(df_clean[col], errors='coerce')
                        if temp.notna().sum() > (len(df_clean) * 0.5):
                            df_clean[col] = temp
                    except Exception:
                        pass
        df_clean.columns = [str(c).strip() for c in df_clean.columns]
        return df_clean
        
    # 1. Drop garbage columns
    cols_to_drop = recipe.get("columns_to_drop", [])
    if cols_to_drop:
        existing_drops = [c for c in cols_to_drop if c in df_clean.columns]
        df_clean = df_clean.drop(columns=existing_drops)
        
    # 2. Clean numeric columns (currencies/commas)
    numeric_cols = recipe.get("numeric_columns", [])
    for ncol in numeric_cols:
        col_name = ncol.get("name")
        if col_name and col_name in df_clean.columns:
            if ncol.get("strip_currency", False):
                if df_clean[col_name].dtype == 'object' or df_clean[col_name].dtype.name == 'string':
                    df_clean[col_name] = df_clean[col_name].astype(str).str.replace(r'[$,£€]', '', regex=True)
            df_clean[col_name] = pd.to_numeric(df_clean[col_name], errors='coerce')
            
    # 3. Clean date columns
    date_cols = recipe.get("date_columns", [])
    for dcol in date_cols:
        if dcol in df_clean.columns:
            df_clean[dcol] = pd.to_datetime(df_clean[dcol], errors='coerce')
            
    # 4. Clean categorical columns (standardize capitalization usually, or strip whitespace)
    cat_cols = recipe.get("categorical_columns", [])
    for ccol in cat_cols:
        if ccol in df_clean.columns and (df_clean[ccol].dtype == 'object' or df_clean[ccol].dtype.name == 'string'):
            df_clean[ccol] = df_clean[ccol].astype(str).str.strip()
            
    # Standardize column names slightly (strip trailing spaces which cause bugs)
    df_clean.columns = [str(c).strip() for c in df_clean.columns]
    
    return df_clean
