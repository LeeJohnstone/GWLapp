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
    return dict_scode_z.get(scode, 0)

# set the color palette for plotting
palette = cycle(sns.color_palette(cc.glasbey, n_colors=50).as_hex())
dict_color = {s: next(palette) for s in df_ts[scode].unique()}

def get_dict_color(s):
    return dict_color.get(s, 'white')

# set mapbox token
px.set_mapbox_access_token('pk.eyJ1IjoiYWRhbW5iZW5uZXR0IiwiYSI6ImNsOGVldGwzODA5cWszcG1vZGJmejYyOXUifQ.7AjKZ8js-hrQR6b19M75Vg')

def main():
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: left;'>Data Explorer</h1>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6 = st.columns(6, gap='small')
    freq = col1.selectbox('Select a Frequency', ['M', 'W', 'D', 'H', 'Raw'])
    mode = col2.selectbox('Select a Mode', ['Normalised', 'Standardised', 'Raw'])
    show_sensor_z = col3.selectbox('Show Sensor Elevations?', ['Yes', 'No'])
    cf = col4.selectbox('Choose a Filter Column', column_filter)
    slx = col5.multiselect('Select filters on Column', df_xy.loc[df_xy[scode].isin(df_ts[scode].unique()), f'Info_{cf}'].unique(), default=None)
    showdry = col6.selectbox('Show Dry Sensors', ['Yes', 'No'])

    centerloc = [
        df_xy['Lat'].mean(),
        df_xy['Lon'].mean()
    ]

    m = folium.Map(location=centerloc, zoom_start=14)

    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Esri Satellite', overlay=False, control=True).add_to(m)

    Draw(export=False, draw_options={
        'polyline': False, 'circlemarker': False, 'polygon': False,
        'circle': False, 'marker': False}).add_to(m)

    fg = folium.FeatureGroup(name='Sensors')

    df_xy_ = df_xy[df_xy[scode].isin(df_ts[scode].unique()) & df_xy[f'Info_{cf}'].isin(slx)]

    for i in df_xy_.index:
        marker = folium.CircleMarker(
            location=[df_xy_.at[i, 'Lat'], df_xy_.at[i, 'Lon']],
            tooltip=df_xy_.at[i, scode], fill=True, fill_opacity=0.7, radius=5)
        fg.add_child(marker)

    col7, col8 = st.columns([2, 1], gap='small')

    with col8:
        st.write("**Interactive Map to Select Sensors: Use 'Draw' Tool to Select Sensors.**")
        output = st_folium(m, width=700, height=450, return_on_hover=False, feature_group_to_add=fg)

    if output['last_active_drawing'] is None:
        list_sensors = df_xy_[scode].iloc[:5]
        df_xy_filt = df_xy_[df_xy_[scode].isin(list_sensors)]
    else:
        coords = output['last_active_drawing']['geometry']['coordinates'][0]
        lat_min, lat_max = sorted([coords[0][1], coords[2][1]])
        lon_min, lon_max = sorted([coords[0][0], coords[2][0]])

        df_xy_filt = df_xy_[(df_xy_['Lat'] >= lat_min) & (df_xy_['Lat'] <= lat_max) &
                            (df_xy_['Lon'] >= lon_min) & (df_xy_['Lon'] <= lon_max)]
        list_sensors = df_xy_filt[scode].unique()

    cds = [get_dict_color(i) for i in df_xy_filt[scode]]

    fig3 = px.scatter_mapbox(df_xy_filt, lat="Lat", lon="Lon", color=scode,
                             color_discrete_sequence=cds, zoom=14, height=550, width=750)
    fig3.update_traces(marker=dict(size=20))
    fig3.update_layout(legend=dict(title='Sensor Code', orientation='h'),
                       mapbox_style="satellite", font=dict(family='Arial', size=14),
                       margin={"r": 50, "t": 0, "l": 0, "b": 0})

    with col8:
        st.write('**Map of Selected Sensors: Legend Matches Time-Series.**')
        st.plotly_chart(fig3)

    fig1 = go.Figure()
    df_ts_ = df_ts[df_ts['SL'] + 1 < df_ts['WL']].copy()

    for s in list_sensors:
        df = df_ts_[df_ts_[scode] == s].copy()
        df[dtime] = pd.to_datetime(df[dtime], errors='coerce')

        if freq == 'Raw':
            df_rs = df
        else:
            df_rs = gwl.resample(df, scode, dtime, val, freq=freq, stat='median')

        x = df_rs[dtime]
        y = gwl.normalise(df_rs[val]) if mode == 'Normalised' else \
            gwl.standardise(df_rs[val]) if mode == 'Standardised' else df_rs[val]

        fig1.add_trace(go.Scattergl(x=x, y=y, mode='lines+markers',
                                    line=dict(width=2, dash='dot', color=get_dict_color(s)),
                                    marker=dict(size=4, color=get_dict_color(s)), name=f'{s}'))

        if show_sensor_z == 'Yes':
            fig1.add_trace(go.Scattergl(x=[df_ts[dtime].min(), df_ts[dtime].max()],
                                        y=[get_z(s), get_z(s)], mode='lines',
                                        line=dict(width=2, color=get_dict_color(s)),
                                        name=f'{s}: Sensor Elevation'))

    fig1.update_layout(title='Time-Series of Selected Sensors', height=1000, width=1100,
                       font=dict(family='Arial', size=16),
                       legend=dict(title='Sensor Code'),
                       margin={"r": 50, "t": 50, "l": 0, "b": 0})
    fig1.update_yaxes(title=f'{mode} Groundwater Level [m]')

    with col7:
        st.plotly_chart(fig1, use_container_width=True)

if __name__ == "__main__":
    main()
