import streamlit as st
import pandas as pd
import json
from pathlib import Path

def dashboard_main():
    st.set_page_config(page_title="Ads ML Dashboard", layout="wide")
    st.title("Ads ML Dashboard")

    # Load data
    all_off = Path("reports/all_offline_metrics.json")
    ab = Path("reports/ab_results.json")    
    rtb = Path("reports/rtb_report.json")
    mon = Path("reports/monitoring.json")
    
    if all_off.exists():
        st.header("All Offline metrics")
        st.code(all_off.read_text())

    if ab.exists():
        st.header("A/B test")
        st.json(json.loads(ab.read_text()))

    if rtb.exists():
        st.header("RTB results")
        st.json(json.loads(rtb.read_text()))
    
    if mon.exists():
        st.header("Monitoring results")
        st.json(json.loads(mon.read_text()))


    st.info("This dashboard is a simple binder to view the JSON reports. Add charts as needed.")

dashboard_main()