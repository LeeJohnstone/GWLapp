# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from itertools import cycle
import seaborn as sns
import GWLs_v01 as gwl

# set variables
scode = 'SensorCode'
dtime = 'DTime'
val = 'WL'

# read in the files from session state
df_xy = st.session_state['df_xy']
df_ts = st.session_state['df_ts']


def main():

    ######################## SETUP #########################    

    # make landscape
    st.set_page_config(layout="wide")

    # app title
    st.markdown("<h1 style='text-align: left;'>Resample Time-Series</h1>", unsafe_allow_html=True)

    ######################### SELECTIONS ##########################

    col1, col2, col3, col4 = st.columns(4, gap='small')
    with col1:# Dropdown menu to select a SensorCode
        sensor_code = st.selectbox('Select a SensorCode', df_ts.sort_values(by=scode)[scode].unique())
    with col2:# Dropdown menu to select a SensorCode
        freq = st.selectbox('Select a Frequency', ['Raw', 'M', 'W', 'D', 'H'])
    with col3:# Dropdown menu to select a SensorCode
        mode = st.selectbox('Select a Mode', ['Normalised', 'Standardised', 'Raw'])
    with col4:
        stat = st.selectbox('Select a Statistic', ['median', 'min', 'max', 'mean'])
    
    ######################### PROCESS DATA ############################
    
    # filter on sensor code
    df_ts_s = df_ts.loc[df_ts[scode]==sensor_code].copy()
    
    # process value field based on dropdown
    if mode == 'Normalised':
        df_ts_s[val] = gwl.normalise(df_ts_s.loc[df_ts_s[scode]==sensor_code, val])
    elif mode == 'Raw':
        df_ts_s[val] = df_ts_s.loc[df_ts_s[scode]==sensor_code, val]
    elif mode == 'Standardised':
        df_ts_s[val] = gwl.standardise(df_ts_s.loc[df_ts_s[scode]==sensor_code, val])

    # resample with given frequency from dropdown
    if freq == 'Raw':
        df_ts_rs = df_ts_s    
    else:
        df_ts_rs = gwl.resample(df_ts_s, scode, dtime, val, freq, stat)
        df_ts_rs.dropna(subset=val, inplace=True)
    
    
    ######################## TIME-SERIES #############################


    # Create time-series plot
    fig1 = go.Figure()
    
    # plot the original frequency
    x = df_ts_s.loc[df_ts_s[scode]==sensor_code, dtime]
    y = df_ts_s.loc[df_ts_s[scode]==sensor_code, val]
    
    # add to fig
    fig1.add_trace(go.Scattergl(
        x=x,
        y=y,
        mode='markers',
        marker=dict(size=4, color='orange'),
        opacity=0.3,
        name=f'{sensor_code}: Original',
        )
    )

    # plot the resampled sensor of interest
    x = df_ts_rs.loc[df_ts_rs[scode]==sensor_code, dtime]
    y = df_ts_rs.loc[df_ts_rs[scode]==sensor_code, val]
    
    # add to fig
    fig1.add_trace(go.Scattergl(
        x=x,
        y=y,
        mode='lines+markers',
        marker=dict(size=6, color='white'),
        line=dict(width=2, color='white', dash='dot'),
        opacity=1,
        name=f'{sensor_code}: Resampled',
        )
    )


    # update layout
    fig1.update_layout(title=f'Time-Series of {sensor_code}: Resampled to Frequency={freq}: Statistic={stat}',
                       height=800, 
                       width=1100, 
                       font=dict(family='Arial', size=12),
                       legend=dict(title='Sensor Code'),
                       margin={"r":50,"t":50,"l":0,"b":0}
                       )
    fig1.update_yaxes(title=f'{mode} Groundwater Level [m]')
        # add signature plot
    
    
    st.plotly_chart(fig1, use_container_width=True)
    def convert_df(df):
        return df.to_csv(index=True).encode('utf-8')
    
    
    
    col1, col2 = st.columns(2, gap='small') 
    with col1:
        if freq == 'Raw':
            st.write('Set Frequency to Resample All Data')    
        else:
            if st.button('Resample All Data'):
                n = len(df_ts[scode].unique())
                rs_prog = st.progress(0, text=f'Resampling {n} sensors') 
                list_df = []
                for i, s in enumerate(df_ts[scode].unique()):
                    # filter on sensor code
                    df_ts_s_ = df_ts.loc[df_ts[scode]==s].copy()
                    rs_prog.progress((i+1)/n, text=f'Resampling Sensor {i+1}/{n}')
                    # process value field based on dropdown
                    if mode == 'Normalised':
                        df_ts_s_[val] = gwl.normalise(df_ts_s_[val])
                    elif mode == 'Raw':
                        pass
                    elif mode == 'Standardised':
                        df_ts_s_[val] = gwl.standardise(df_ts_s_[val])

                    # resample
                    df_ts_s_rs = gwl.resample(df_ts_s_, scode, dtime, val, freq, stat)

                    # append to list
                    list_df.append(df_ts_s_rs)
                
                # concat all 
                df_ts_rs_all = pd.concat(list_df)



                csv = convert_df(df_ts_rs_all)
                st.download_button(
                    "Export Resampled Time-Series for All Sensors",
                    csv,
                    f"df_ts_rs_{freq}_{stat}_{mode}.csv",
                    "text/csv",
                    key='download-df-rs',
                    use_container_width=True
                    )


if __name__ == "__main__":
    main()