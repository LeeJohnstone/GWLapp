import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from itertools import cycle
import seaborn as sns



######################## SETUP #########################

st.set_page_config(
    page_title="Upload",
    layout="wide")

col1, col2 = st.columns([1, 2], gap='small')
with col1:
    st.image('SRK_Logo.png', width=800)
    st.title("Groundwater Level Analysis")
    st.write("Author: ABennett@srk.co.uk")
    st.write("Version: 01")
    st.write("Date: 12/01/2024") 
    st.write("")
    st.write("This application is a work in progress designed to assist with groundwater level interpretation.")
    st.write("")
    st.write("")

    
######################## UPLOADS #########################

# time-series upload 
ts_upload = st.file_uploader("Upload Time Series Data. Fields required ['DTime', 'SensorCode', 'WL', 'SL']. Accepts CSV or Parquet Files.", 
                         #label_visibility = "hidden",
                         accept_multiple_files=False)

if ts_upload is not None:
    try:
        df_ts = pd.read_csv(ts_upload, parse_dates=['DTime'], dayfirst=True)
    except:
        df_ts = pd.read_parquet(ts_upload)
        df_ts['DTime'] = pd.to_datetime(df_ts['DTime'])

    st.write(df_ts.head())
    st.write(df_ts.dtypes)
    # add the dfs to session state for use in other pages
    st.session_state['df_ts'] = df_ts

# xy upload
xy_upload = st.file_uploader("Upload Lat-Lon Data.  Fields required ['SensorCode', 'Lat', 'Lon'].  Accepts CSV files.",
                         accept_multiple_files=False)

if xy_upload is not None:
    df_xy = pd.read_csv(xy_upload)
    st.write(df_xy.head())
    st.write(df_xy.dtypes)
    st.session_state['df_xy'] = df_xy

# signatures upload (if not added, can be created via the Groundwater Signatures page, (fairly slow).
sig_upload = st.file_uploader("Upload GW Signature Data.  If unavailable, export from Groundwater Signatures page. Accepts CSV files.",
                         accept_multiple_files=False)

if sig_upload is not None:
    df_signatures = pd.read_csv(sig_upload, index_col=0)
    st.write(df_signatures.head())
    st.write(df_ts.dtypes)
    if 'df_signatures' not in st.session_state:
        st.session_state['df_signatures'] = df_signatures

# upload the various stresses 
stress_upload = st.file_uploader("Upload Stress Data.  Fields required ['StressID', 'DTime', 'Value', 'Units']. Accepts CSV files.",
                                accept_multiple_files=False)

if stress_upload is not None:
    df_stresses = pd.read_csv(stress_upload, parse_dates=['DTime'], dayfirst=True)
    st.write(df_stresses.head())
    st.write(df_ts.dtypes)
    if 'df_stresses' not in st.session_state:
        st.session_state['df_stresses'] = df_stresses
