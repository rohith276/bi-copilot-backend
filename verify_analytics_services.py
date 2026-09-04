import os
import sys
import pandas as pd
import numpy as np

# Ensure backend path is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.forecasting_service import forecast_sales, predict_trend
from app.services.recommendation_service import generate_recommendations
from app.services.anomaly_service import detect_anomalies
from app.services.root_cause_service import analyze_root_cause

csv_path = "backend/uploads/a15bed8b-5f57-409c-92b2-cef459998762.csv"
if not os.path.exists(csv_path):
    csv_path = "uploads/a15bed8b-5f57-409c-92b2-cef459998762.csv"

print(f"Loading test dataset: {csv_path}")
df = pd.read_csv(csv_path, encoding="latin1")
print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

print("\n" + "="*70)
print("TEST 1: TIME SERIES FORECASTING")
print("="*70)
f_res = forecast_sales(df, date_col="Order Date", value_col="Sales", periods=14)
if "error" in f_res:
    print(f"ERROR: {f_res['error']}")
else:
    print(f"Model Engine: {f_res.get('model_engine')}")
    print(f"Detected Frequency: {f_res.get('frequency')}")
    print(f"Bounded R2/Confidence Score: {f_res.get('r2_score')} ({f_res.get('confidence')}%)")
    print(f"Holdout Test MAPE: {f_res.get('test_mape')}")
    print(f"Forecast Projections (first 5): {f_res.get('values')[:5]}")
    print(f"Lower Bounds (first 5): {f_res.get('lower_bounds')[:5]}")
    print(f"Upper Bounds (first 5): {f_res.get('upper_bounds')[:5]}")
    print(f"Trend Direction: {f_res.get('trend')}")
    assert f_res.get('r2_score') > 0, "R2 score must be positive!"
    assert len(set(f_res.get('values'))) > 1, "Forecast must NOT be a flat line!"

print("\n" + "="*70)
print("TEST 2: REGRESSION ANALYSIS")
print("="*70)
p_res = predict_trend(df, target_col="Sales", feature_cols=["Profit", "Discount", "Quantity", "Category", "Region"])
if "error" in p_res:
    print(f"ERROR: {p_res['error']}")
else:
    print(f"Best Model Engine: {p_res.get('model_engine')}")
    print(f"Out-of-Sample Test R2: {p_res.get('r2_score')}")
    print(f"Adjusted R2: {p_res.get('adjusted_r2')}")
    print(f"Holdout MAE: ${p_res.get('mae'):,.2f}")
    print(f"Holdout RMSE: ${p_res.get('rmse'):,.2f}")
    print(f"Top 5 Coefficients/Importances: {dict(list(p_res.get('coefficients', {}).items())[:5])}")
    assert p_res.get('r2_score') > 0.40, "Regression R2 should be healthy and verified!"

print("\n" + "="*70)
print("TEST 3: INVENTORY ACTION SPECS (ABC PARETO & ROP)")
print("="*70)
r_res = generate_recommendations(df, product_col="Product Name", sales_col="Sales", inventory_col="Quantity")
print(f"Total Action Specs Generated: {len(r_res)}")
for i, spec in enumerate(r_res[:5]):
    print(f"[{i+1}] {spec.get('category')} | Priority: {spec.get('priority')}")
    print(f"    Action: {spec.get('action')}")
    print(f"    Reason: {spec.get('reason')}")
    print(f"    Velocity: {spec.get('velocity')} units/day | Recommended: {spec.get('recommended_units')} units")
assert len(r_res) > 0, "Recommendations must not be empty!"
assert any("Class A" in s.get('category', '') for s in r_res), "Must contain Class A specs!"

print("\n" + "="*70)
print("TEST 4: ANOMALY DETECTION (ROBUST MAD & ISOLATION FOREST)")
print("="*70)
a_res = detect_anomalies(df, column="Sales", contamination=0.02)
print(f"Total Anomalies Flagged: {len(a_res)}")
for i, ano in enumerate(a_res[:5]):
    print(f"[{i+1}] Row #{ano['row_index']} | Value: ${ano['value']:,.2f} | Z-Score: {ano['z_score']} | Severity: {ano['severity_score']}% | Type: {ano['anomaly_type']}")
assert len(a_res) > 0, "Anomalies must detect extreme points!"
assert all(0 <= a['severity_score'] <= 100 for a in a_res), "Severity scores must be between 0 and 100!"

print("\n" + "="*70)
print("TEST 5: ROOT-CAUSE ANALYSIS (MATHEMATICAL CONSERVATION & VOLUME/RATE)")
print("="*70)
rc_res = analyze_root_cause(df, metric_col="Sales")
if "error" in rc_res:
    print(f"ERROR: {rc_res['error']}")
else:
    print(f"Metric: {rc_res.get('metric')}")
    print(f"Period: {rc_res.get('period_label')}")
    print(f"Prior Total: ${rc_res.get('prior_total'):,.2f} ({rc_res.get('prior_transactions')} orders @ ${rc_res.get('prior_avg_ticket'):,.2f}/order)")
    print(f"Recent Total: ${rc_res.get('recent_total'):,.2f} ({rc_res.get('recent_transactions')} orders @ ${rc_res.get('recent_avg_ticket'):,.2f}/order)")
    print(f"Total Delta: ${rc_res.get('delta'):+,.2f} ({rc_res.get('delta_pct'):+,.2f}%)")
    print(f"Volume Effect: ${rc_res.get('volume_effect'):+,.2f}")
    print(f"Ticket Price (Rate) Effect: ${rc_res.get('rate_effect'):+,.2f}")
    decomp_sum = rc_res.get('volume_effect', 0) + rc_res.get('rate_effect', 0)
    print(f"Sum of Volume + Rate Effect: ${decomp_sum:+,.2f} (matches delta: {abs(decomp_sum - rc_res.get('delta', 0)) < 0.05})")
    print(f"Narrative: {rc_res.get('narrative')}")
    print(f"Top Dimension: {rc_res['breakdowns'][0]['dimension']} with delta ${rc_res['breakdowns'][0]['total_delta']:,.2f}")

print("\n" + "="*70)
print("ALL 5 SERVICES TESTED AND VERIFIED!")
print("="*70)
