# Import necessary libraries
import pandas as pd
import plotly.express as px
import streamlit as st
import datetime as dt
import GWLs_v01 as gwl
import plotly.graph_objects as go
from itertools import cycle
import seaborn as sns
import numpy as np
import colorcet as cc

# set mapbox token
px.set_mapbox_access_token('pk.eyJ1IjoiYWRhbW5iZW5uZXR0IiwiYSI6ImNsOGVldGwzODA5cWszcG1vZGJmejYyOXUifQ.7AjKZ8js-hrQR6b19M75Vg')

# load time-series
df_ts = st.session_state['df_ts']
df_xy = st.session_state['df_xy']

# load variables
dtime = 'DTime'
val = 'WL'
scode = 'SensorCode'

def main():

    # Make landscape
    st.set_page_config(layout="wide")

    # App title
    st.markdown("<h1 style='text-align: left;'>Groundwater Level Clustering</h1>", unsafe_allow_html=True)

    # Create two columns for the figures
    col1, col2, col3, col4, col5, col6 = st.columns(6, gap='small')


    with col1:# Dropdown menu to select a SensorCode
        mode = st.selectbox('Select Datatype', ["Raw", "Normalised", "Standardised"])
    with col2:# Dropdown menu to select a SensorCode
        n_clusters = st.selectbox('Select N Clusters', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    with col3:
        start_date = st.date_input('Start Date for Clustering',
                                   df_ts['DTime'].min(),
                                   min_value=df_ts['DTime'].min(),
                                   max_value=df_ts['DTime'].max(),
                                   format='YYYY-MM-DD',
                                   label_visibility='visible'
                                   )
    with col4:
        end_date = st.date_input('End Date for Clustering',
                                  df_ts['DTime'].max(),
                                  min_value=df_ts['DTime'].min(),
                                  max_value=df_ts['DTime'].max(),
                                  format='YYYY-MM-DD',
                                  label_visibility='visible'
                                    )
    with col5:
        freq = st.selectbox('Select frequency', ['W', 'M', 'D', 'Y', 'H'])
    with col6:
        stat = st.selectbox('Select statistic', ['median', 'mean', 'min', 'max'])

    # convert inputs to dtime
    start_dtime = dt.datetime(start_date.year, start_date.month, start_date.day)
    end_dtime = dt.datetime(end_date.year, end_date.month, end_date.day)

    col5, col6 = st.columns(2, gap='small')
    
    # set the color palette for plotting
    palette = cycle(sns.color_palette(cc.glasbey, n_colors=16).as_hex())
    dict_color = {}
    for i in np.arange(0, 15): 
        dict_color[i] = next(palette)
    

    # plot the sensor of interest
    fig1, df_groups = gwl.dtw_cluster(df_ts, scode, dtime, val, mode, n_clusters, start_dtime, end_dtime, freq, stat, dict_color=dict_color)
    fig1.update_layout(margin={"r":50,"t":50,"l":0,"b":0})
    height = fig1.layout.height
    

    # Filter df_lat_lon based on correlated sensors
    df_lat_lon_corr = df_xy[df_xy['SensorCode'].isin(df_groups['SensorCode'])]
    df_lat_lon_corr.sort_values(by='SensorCode', inplace=True)
    for g in df_groups['Group'].unique():
        df_lat_lon_corr.loc[df_lat_lon_corr['SensorCode'].isin(df_groups.loc[df_groups['Group']==g, 'SensorCode'].unique()), 'Group'] = g
    df_lat_lon_corr['GroupInt'] = [int(g) for g in df_lat_lon_corr['Group']]     
    df_lat_lon_corr = df_lat_lon_corr.sort_values(by='GroupInt')
    cds = [dict_color[float(i)-1] for i in df_lat_lon_corr['Group'].unique()]
    n_sensors = len(df_groups[scode].unique())
    # Create map
    fig2 = px.scatter_mapbox(df_lat_lon_corr, 
                             lat="Lat", 
                             lon="Lon", 
                             color="Group", 
                             zoom=14, 
                             height=height,
                             color_discrete_sequence=cds,
                             width=1100
                             )
    fig2.update_traces(marker=dict(size=18))
    fig2.update_layout(title=f'Map of Clusters with N Clusters = {n_clusters} and N Sensors = {n_sensors}',
                       legend=dict(title='Group'),
                       mapbox_style="satellite",
                       font=dict(family='Arial', size=12),
                       margin={"r":50,"t":50,"l":0,"b":0}
                       )

    # Display the figures in the columns
    col5.plotly_chart(fig1, use_container_width=True)
    col6.plotly_chart(fig2, use_container_width=True)

    st.dataframe(df_groups)
    st.session_state['df_groups'] = df_groups
    
    col7, col8 = st.columns(2, gap='small')

    def convert_df(df):
        return df.to_csv(index=True).encode('utf-8')

    with col7:
            csv = convert_df(df_groups)
            st.download_button(
                "Export Selected Cluster Groups",
                csv,
                f"df_groups_{freq}_{stat}_{mode}_{start_date}_{end_date}.csv",
                "text/csv",
                key='download-df-groups',
                use_container_width=True
                )

if __name__ == "__main__":
    main()