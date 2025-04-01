# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from itertools import cycle
import seaborn as sns
import GWLs_v01 as gwl
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import colorcet as cc
import io

# set variables
scode = 'SensorCode'
dtime = 'DTime'
val = 'WL'

# read in the files from session state
df_xy = st.session_state['df_xy']
df_ts = st.session_state['df_ts']

# column filters from xy
column_filter = [i.split('_')[1] for i in df_xy.columns if 'Info' in i]
column_filter.append('None')



# make a dict of sensor elevations
dict_scode_z = df_xy.set_index(scode)['Z'].to_dict()
def get_z(scode):
    try:
        z = dict_scode_z[scode]
    except:
        z = 0
    return z 

# set the color palette for plotting
palette = cycle(sns.color_palette(cc.glasbey, n_colors=50).as_hex())
dict_color = {}
for s in df_ts[scode].unique(): 
    dict_color[s] = next(palette)

# method call for dictionary of colors per sensor for consistency across plots
def get_dict_color(s):
    try:
        col = dict_color[s]
    except:
        col = 'white'
    return col

# set mapbox token
px.set_mapbox_access_token('pk.eyJ1IjoiYWRhbW5iZW5uZXR0IiwiYSI6ImNsOGVldGwzODA5cWszcG1vZGJmejYyOXUifQ.7AjKZ8js-hrQR6b19M75Vg')
    

def main():

    ######################### SETUP ##########################

    # make landscape
    st.set_page_config(layout="wide")

    # page title
    st.markdown("<h1 style='text-align: left;'>Data Explorer</h1>", unsafe_allow_html=True)

    ######################### SELECTIONS ##########################

    col1, col2, col3, col4, col5, col6 = st.columns(6, gap='small')
    #with col1:# Dropdown menu to select a SensorCode
    #    sensor_code = st.selectbox('Select a SensorCode', df_ts.sort_values(by=scode)[scode].unique())
    with col1:# Dropdown menu to select a SensorCode
        freq = st.selectbox('Select a Frequency', ['M', 'W', 'D', 'H', 'Raw'])
    with col2:# Dropdown menu to select a SensorCode
        mode = st.selectbox('Select a Mode', ['Normalised', 'Standardised', 'Raw'])
    with col3:
        show_sensor_z = st.selectbox('Show Sensor Elevations?', ['Yes', 'No'])
    with col4:
        cf = st.selectbox('Choose a Filter Column', column_filter)
    with col5:
        slx = st.multiselect('Select filters on Column', df_xy.loc[df_xy[scode].isin(df_ts[scode].unique()), f'Info_{cf}'].unique(), df_xy.loc[df_xy[scode].isin(df_ts[scode].unique()), f'Info_{cf}'].unique())
    with col6:
        showdry = st.selectbox('Show Dry Sensors', ['Yes', 'No'])
    ######################## MAP (FOLIUM) ############################
    
    # set center of map 
    centerloc = [
            df_xy['Lat'].max()-(df_xy['Lat'].max()-df_xy['Lat'].min())/2, 
            df_xy['Lon'].max()-(df_xy['Lon'].max()-df_xy['Lon'].min())/2
                ]

    # create map for interaction
    m = folium.Map(location=centerloc,
                   zoom_start=14,
                   title='Map of All Sensors')
    
    # add satellite imagery tile
    folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = False,
        control = True
       ).add_to(m)
    
    # add draw functionality which is used to select features inside rectangle
    Draw(export=False, 
         draw_options={'polyline':False, 
                       'circlemarker': False, 
                       'polygon':False, 
                       'circle':False, 
                       'marker':False}).add_to(m)
    
    # create a feature group to store markers
    fg = folium.FeatureGroup(name='Sensors')

    # filter on those sensors with time-series and filtered column matching selection
    df_xy_ = df_xy.loc[(df_xy[scode].isin(df_ts[scode].unique())) & (df_xy[f'Info_{cf}'].isin(slx))]

    # add all XY locs as markers (if in df_ts)
    for i in df_xy_.index:
        marker = folium.CircleMarker(
                location = [df_xy_.at[i, 'Lat'], df_xy_.at[i, 'Lon']],
                tooltip=df_xy_.at[i, scode],
                fill=True,
                fill_opacity=0.7,
                radius=5
            )
        fg.add_child(marker)
    
    
    # create holders for the figures
    col7, col8 = st.columns([2, 1], gap='small')
    
    # add folium to right hand side of page (Col 4)
    with col8:
        st.write("**Interactive Map to Select Sensors: Use 'Draw' Tool to Select Sensors.**")
        output = st_folium(m, 
                           width=700, 
                           height=450, 
                           return_on_hover=False,
                           feature_group_to_add=fg)

    # use the output from drawing to filter df_ts
        
    # if no drawing
    if output['last_active_drawing'] == None:
        # select arbitrary selection of points 
        list_sensors = df_xy_.loc[df_xy_[scode].isin(df_ts[scode].unique()), scode][0:5]
        df_xy_filt = df_xy_.loc[df_xy_[scode].isin(list_sensors)]
    else:
        # retrieve box coordinates
        lat_x1 = output['last_active_drawing']['geometry']['coordinates'][0][0][1]
        lon_y1 = output['last_active_drawing']['geometry']['coordinates'][0][0][0]
        lat_x2 = output['last_active_drawing']['geometry']['coordinates'][0][2][1]
        lon_y2 = output['last_active_drawing']['geometry']['coordinates'][0][2][0]

        # filter df xy on these locations
        df_xy_filt = df_xy_.loc[(df_xy_['Lat']>=lat_x1) & (df_xy_['Lon']>=lon_y1) & (df_xy_['Lat']<=lat_x2) & (df_xy_['Lon']<=lon_y2)]
        df_xy_filt = df_xy_filt.loc[df_xy_filt[scode].isin(df_ts[scode].unique())]
        # list of sensors
        list_sensors = df_xy_filt[scode].unique()

    ############## SECOND MAP (PLOTLY MAPBOX) ##################
    
    # set the color discrete selection based on the locations 
    cds = [get_dict_color(i) for i in df_xy_filt[scode]]

    # generate scatter mapbox
    fig3 = px.scatter_mapbox(df_xy_filt, 
                             lat="Lat", 
                             lon="Lon", 
                             color=scode,
                             color_discrete_sequence=cds, 
                             zoom=14, 
                             height=550,
                             width=750
                             )
    
    # update size of markers
    fig3.update_traces(marker=dict(size=20))
    # update layout
    fig3.update_layout(
                       legend=dict(title='Sensor Code', orientation='h'),
                       mapbox_style="satellite",
                       font=dict(family='Arial', size=14),
                       margin={"r":50,"t":0,"l":0,"b":0}
                       )
    
    # add to col4 below folium map
    with col8:
        st.write('**Map of Selected Sensors: Legend Matches Time-Series.**')
        st.plotly_chart(fig3)

    ######################## TIME-SERIES #############################

    # Create time-series plot
    fig1 = go.Figure()

    df_ts_ = df_ts.loc[df_ts['SL']+1<df_ts['WL']].copy()
    
    # iterate through sensors and add time-series
    for s in list_sensors:
        df = df_ts_.loc[df_ts_[scode]==s].copy() 
        # resample based on selected dropdown
        df = gwl.resample(df, scode, dtime, val, freq=freq, stat='median')
        x = df[dtime]
        # set y variable depending on the mode
        if mode == 'Normalised':
            y = gwl.normalise(df[val])
        elif mode == 'Standardised':
            y = gwl.standardise(df[val])
        else:
            y = df[val]
        # add WL to figure
        fig1.add_trace(go.Scattergl(
            x=x,
            y=y,
            mode='lines+markers',
            line=dict(width=2, dash='dot', color=get_dict_color(s)),
            marker=dict(size=4, color=get_dict_color(s)),
            name=f'{s}',
            )
        )

        if show_sensor_z == 'Yes':    
            # add sensorz to figure
            fig1.add_trace(go.Scattergl(
                x=[df_ts[dtime].min(), df_ts[dtime].max()],
                y=[get_z(s), get_z(s)],
                mode='lines',
                line=dict(width=2, color=get_dict_color(s)),
                name=f'{s}: Sensor Elevation',
                )
            )
        else:
            pass



    # update the layout
    fig1.update_layout(title=f'Time-Series of Selected Sensors',
                       height=1000, 
                       width=1100, 
                       font=dict(family='Arial'),
                       legend=dict(title='Sensor Code'),
                       margin={"r":50,"t":50,"l":0,"b":0}
                       )
    fig1.update_layout(font=dict(size=16))
    fig1.update_yaxes(title=f'{mode} Groundwater Level [m]')
        # add signature plot
    with col7:
        st.plotly_chart(fig1, use_container_width=True)
if __name__ == "__main__":
    main()