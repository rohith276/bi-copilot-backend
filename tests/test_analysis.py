import pytest
import pandas as pd
import numpy as np
from app.services import analysis_service, cleaning_service, anomaly_service
from app.schemas.query import QueryRequest, GroupBy

def test_basic_clean():
    data = {
        'A': [1, -2, np.nan, 4],
        'B': [' x ', 'y', ' ', 'z'],
        'C': ['2023-01-01', '2023-01-02', '2023-01-03', 'invalid']
    }
    df = pd.DataFrame(data)
    cleaned_df = cleaning_service.basic_clean(df)
    
    assert cleaned_df['A'].isnull().sum() == 1
    assert cleaned_df.loc[1, 'A'] == -2
    assert cleaned_df.loc[0, 'B'] == 'x'
    assert pd.isna(cleaned_df.loc[2, 'B'])

def test_group_by_count_keeps_categorical_values():
    df = pd.DataFrame({
        'category': ['A', 'A', 'B', 'B', 'B'],
        'value': [10, 20, 30, 40, 50],
    })

    result = analysis_service.process_query(
        df,
        QueryRequest(
            group_by=GroupBy(columns=['category'], agg_funcs={'category': 'count'}),
            limit=10,
        ),
    )

    counts = {row['category']: row['category_count'] for row in result['data']}
    assert counts == {'A': 2, 'B': 3}

def test_anomaly_detection():
    # Create data with a clear outlier
    data = {'val': [10, 11, 12, 10, 11, 100, 11, 12, 10]}
    df = pd.DataFrame(data)
    anomalies = anomaly_service.detect_anomalies(df, 'val')
    
    assert len(anomalies) > 0
    assert anomalies[0]['value'] == 100
    assert anomalies[0]['deviation'] == "High"

def test_anomaly_empty_df():
    df = pd.DataFrame({'val': []})
    anomalies = anomaly_service.detect_anomalies(df, 'val')
    assert anomalies == []
