import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
try:
    from sklearn.linear_model import LinearRegression
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except ImportError:
    LinearRegression = None
    ExponentialSmoothing = None

def forecast_sales(df: pd.DataFrame, date_col: str, value_col: str, periods: int = 30) -> Dict[str, Any]:
    """
    Perform a simple sales forecast using Exponential Smoothing.
    """
    try:
        # Preprocessing
        if date_col not in df.columns:
            return {"error": f"Column '{date_col}' not found in dataset."}
        if value_col not in df.columns:
            return {"error": f"Column '{value_col}' not found in dataset."}
        if periods <= 0:
            return {"error": "Forecast periods must be greater than zero."}
        if ExponentialSmoothing is None:
            return {"error": "Forecasting dependencies are not installed."}
            
        df_ts = df.copy()
        try:
            df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors='coerce')
        except Exception:
            return {"error": f"Column '{date_col}' could not be converted to date format."}

        # Clean string currency/commas before numeric conversion
        if df_ts[value_col].dtype == 'object' or df_ts[value_col].dtype.name == 'string':
            df_ts[value_col] = df_ts[value_col].astype(str).str.replace(r'[$,]', '', regex=True)

        df_ts[value_col] = pd.to_numeric(df_ts[value_col], errors='coerce')
        df_ts = df_ts.dropna(subset=[date_col, value_col])
        if len(df_ts) < 2:
            return {"error": f"Not enough valid dates or numeric values in '{date_col}' and '{value_col}' to generate a forecast."}


        df_ts = df_ts.set_index(date_col).resample('D')[value_col].sum().fillna(0)
        if len(df_ts) < 2:
            return {"error": "Not enough daily observations are available to generate a forecast."}
        
        # Fit model
        try:
            model = ExponentialSmoothing(df_ts, trend='add', seasonal=None, initialization_method="estimated")
            model_fit = model.fit()
        except TypeError:
            # Fallback for older statsmodels versions without initialization_method
            model = ExponentialSmoothing(df_ts, trend='add', seasonal=None)
            model_fit = model.fit(optimized=True)
        
        # Forecast
        forecast = model_fit.forecast(periods)
        
        forecast_dates = [df_ts.index[-1] + timedelta(days=i+1) for i in range(periods)]
        
        return {
            "dates": [d.strftime('%Y-%m-%d') for d in forecast_dates],
            "values": [round(float(v), 2) for v in forecast],
            "trend": "up" if forecast.iloc[-1] > forecast.iloc[0] else "down"
        }
    except Exception as e:
        return {"error": str(e)}

def predict_trend(df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> Dict[str, Any]:
    """
    Simple trend prediction using Linear Regression. Handles numeric and categorical features automatically.
    """
    try:
        if LinearRegression is None:
            return {"error": "Prediction dependencies are not installed."}
        if target_col not in df.columns:
            return {"error": f"Column '{target_col}' not found in dataset."}
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            return {"error": f"Feature columns not found: {', '.join(missing_features)}"}
        if not feature_cols:
            return {"error": "At least one feature column is required for prediction."}

        data = df.copy()
        
        # Target column must definitively be numeric
        if data[target_col].dtype == 'object' or data[target_col].dtype.name == 'string':
            data[target_col] = data[target_col].astype(str).str.replace(r'[$,£€]', '', regex=True)
        data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
        
        # Categorize feature columns
        numeric_features = []
        categorical_features = []
        for col in feature_cols:
            if data[col].dtype == 'object' or data[col].dtype.name == 'string':
                categorical_features.append(col)
                # Ensure they are strings, filling NaN with placeholder
                data[col] = data[col].fillna("Unknown").astype(str)
            else:
                numeric_features.append(col)
                data[col] = pd.to_numeric(data[col], errors='coerce')
                
        # Drop rows where target or numerical features are completely null
        data = data.dropna(subset=[target_col] + numeric_features)
        
        if len(data) < 2:
            return {"error": "Not enough valid numeric rows are available to train the prediction model."}
            
        # One-hot encode categorical features if any exist
        if categorical_features:
            # We use drop_first=True to avoid perfect multicollinearity in regression
            data = pd.get_dummies(data, columns=categorical_features, drop_first=True, dtype=float)
            # Gather all resulting dummy columns + numeric columns
            final_features = numeric_features + [c for c in data.columns if any(c.startswith(orig + '_') for orig in categorical_features)]
            X = data[final_features]
        else:
            X = data[feature_cols]
            
        y = data[target_col]
        
        model = LinearRegression()
        model.fit(X, y)
        
        return {
            "coefficients": {str(col): round(float(c), 4) for col, c in zip(X.columns, model.coef_)},
            "intercept": round(float(model.intercept_), 2),
            "r2_score": round(float(model.score(X, y)), 4)
        }
    except Exception as e:
        return {"error": str(e)}
