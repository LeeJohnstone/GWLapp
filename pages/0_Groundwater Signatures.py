# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from itertools import cycle
import seaborn as sns
import GWLs_v01 as gwl

# set variables for column names
scode = 'SensorCode'
dtime = 'DTime'
val = 'WL'

# read in the files from session state
# df_xy = st.session_state['df_xy']
df_ts = st.session_state['df_ts']

# set mapbox token
px.set_mapbox_access_token('pk.eyJ1IjoiYWRhbW5iZW5uZXR0IiwiYSI6ImNsOGVldGwzODA5cWszcG1vZGJmejYyOXUifQ.7AjKZ8js-hrQR6b19M75Vg')
    
def main():

    ######################### SETUP ##########################

    # make landscape
    st.set_page_config(layout="wide")

    # page title
    st.markdown("<h1 style='text-align: left;'>Groundwater Signatures</h1>", unsafe_allow_html=True)

    ######################### SELECTIONS ##########################

    col1, col2, col3, col4 = st.columns(4, gap='small')
    with col1:# Dropdown menu to select a SensorCode
        freq = st.selectbox('Select a Frequency', ['W', 'M', 'D'])

    
    ######################### PROCESSING ############################
    
    # resample with given frequency from dropdown
    df_ts_rs = gwl.resample(df_ts, scode, dtime, val, freq, 'median')
    all_sensors = df_ts_rs[scode].unique()

    # get signatures if not in session state
    if 'df_signatures' not in st.session_state:
        # get signatures
        st.spinner(text='Getting Groundwater Signatures..', cache=False)
        df_signatures = gwl.get_signatures(df=df_ts_rs, sensors=all_sensors, scode=scode, dtime=dtime, val=val)
        st.session_state['df_signatures'] = df_signatures
    else:
        df_signatures = st.session_state['df_signatures']

    # normalise if if the normalised signatures haven't been calculated yet
    if 'df_signatures_norm' not in st.session_state:
        st.spinner(text='Normalising Groundwater Signatures..', cache=False)
        df_signatures_norm = df_signatures.copy()
        for col in df_signatures_norm.columns:
            df_signatures_norm[col] = (df_signatures[col].values-df_signatures.min(axis=1))/(df_signatures.max(axis=1)-df_signatures.min(axis=1))
            # save to session state
            st.session_state['df_signatures_norm'] = df_signatures_norm
    else:
        df_signatures_norm = st.session_state['df_signatures_norm']
    

    ######################## PLOTS #############################

    fig1 = gwl.plot_signatures(df_signatures, None)
    st.plotly_chart(fig1, use_container_width=True)
    fig2 = gwl.plot_signatures(df_signatures_norm, None)
    st.plotly_chart(fig2, use_container_width=True)

    ###################### EXPORTS ##############################
    
    def convert_df(df):
        return df.to_csv(index=True).encode('utf-8')
    csv = convert_df(df_signatures)
    csv2 = convert_df(df_signatures_norm)
    
    col1, col2 = st.columns(2, gap='small') 
    with col1:
        st.download_button(
            "Export GW Signatures",
            csv,
            f"GW_Signatures.csv",
            "text/csv",
            key='download-sigs',
            use_container_width=True
            )
    with col2:
        st.download_button(
            "Export Normalised GW Signatures",
            csv,
            f"Normalised_GW_Signatures.csv",
            "text/csv",
            key='download-sigs-norm',
            use_container_width=True
            )
        

if __name__ == "__main__":
    main()