import pandas as pd
import glob
from app.services.forecasting_service import forecast_sales
from app.services.dataset_service import get_dataset_df

for f in glob.glob('uploads/*'):
    print(f"--- File: {f} ---")
    try:
        df = get_dataset_df(f)
        dcols = df.select_dtypes(include=['object', 'datetime']).columns
        ncols = df.select_dtypes(include='number').columns
        if len(dcols) > 0 and len(ncols) > 0:
            res = forecast_sales(df, dcols[0], ncols[0], 30)
            print("Result:", res if "error" in res else "SUCCESS")
    except Exception as e:
        print("Crash:", e)
