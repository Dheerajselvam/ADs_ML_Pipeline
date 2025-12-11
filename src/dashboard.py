import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(page_title="Ads ML Dashboard", layout="wide")
st.title("Ads ML Dashboard — Demo")

# Load data
off = Path("reports/offline_metrics.json")
ab = Path("reports/ab_results.json")
rtb = Path("reports/rtb_report.json")
calib = Path("reports/calibration_table.csv")

if off.exists():
    st.header("Offline metrics")
    st.code(off.read_text())

if ab.exists():
    st.header("A/B test")
    st.json(json.loads(ab.read_text()))

if rtb.exists():
    st.header("RTB results")
    st.json(json.loads(rtb.read_text()))

if calib.exists():
    st.header("Calibration table (CSV)")
    df = pd.read_csv(calib)
    st.dataframe(df)

st.info("This dashboard is a simple binder to view the JSON reports. Add charts as needed.")
