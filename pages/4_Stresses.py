# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from itertools import cycle
import seaborn as sns
import GWLs_v01 as gwl
from plotly.subplots import make_subplots
from scipy.stats import spearmanr, pearsonr 

# set variables
scode = 'SensorCode'
dtime = 'DTime'
val = 'WL'
s_id = 'StressID'
s_val = 'Value'

# Read in the files
#df_xy = st.session_state['df_xy']
df_ts = st.session_state['df_ts']
df_stresses = st.session_state['df_stresses']
try:
    df_groups = st.session_state['df_groups'] 
except:
    df_groups = pd.DataFrame({'Nan': ['Empty']})
# set mapbox token
px.set_mapbox_access_token('pk.eyJ1IjoiYWRhbW5iZW5uZXR0IiwiYSI6ImNsOGVldGwzODA5cWszcG1vZGJmejYyOXUifQ.7AjKZ8js-hrQR6b19M75Vg')
    
def main():

    # Make landscape
    st.set_page_config(layout="wide")

    # App title
    st.markdown("<h1 style='text-align: left;'>Stresses vs Groundwater Levels</h1>", unsafe_allow_html=True)

    ######################### Selections ##########################

    col1, col2, col3, col4, col5, col6 = st.columns(6, gap='small')
    with col1:
        if 'Nan' in df_groups.columns:
            groups = ['All']
        else:
            groups = [str(i) for i in df_groups.sort_values(by='Group')['Group'].unique()]
            groups = ['All'] + groups

        group_id = st.selectbox('Select a Cluster Group', groups)
    
    if group_id == 'All':
        scodes = df_ts.sort_values(by=scode)[scode].unique()
    else:
        scodes = df_groups.loc[df_groups['Group'].astype(str)==group_id].sort_values(by=scode)[scode].unique()

    with col2:# Dropdown menu to select a SensorCode
        sensor_code = st.selectbox('Select a SensorCode', scodes)
    with col3:
         stress_id = st.selectbox('Select StressID', df_stresses[s_id].unique())
    with col4:# Dropdown menu to select a SensorCode
        freq = st.selectbox('Select a Frequency', ['M', 'W', 'D', 'H'])
        if freq == 'H' or freq == 'D' or freq == 'M':
            components = ['Observed']
        else:
            components = ['Observed', 'Seasonal-MA', 'Seasonal-STL', 'Trend-MA', 'Trend-STL', 'Residual-MA', 'Residual-STL']

    with col5:# Dropdown menu to select a SensorCode
        stress_statistic = st.selectbox('Select a Stress Statistic', ['mean', 'min', 'max', 'sum', 'median', 'cumsum', 'cumdep', 'seasonal'])
    with col6:
        component = st.selectbox('Select Time-Series Component', components)
    
    ######################### Processing ############################
    
    # resample with given frequency from dropdown
    df_ts_s = df_ts.loc[df_ts[scode]==sensor_code].copy()
    df_ts_rs = gwl.resample(df_ts_s, scode, dtime, val, freq, 'median')
    df_ts_rs[val] = df_ts_rs[val].interpolate()
    #df_ts_rs.dropna(subset=val, inplace=True)

    st.write(df_ts_rs.head())
 

    if component == 'Observed':
        df_ts_rs[val] = df_ts_rs[val]
    else:
        if freq == 'W':
            seasonal = 53
        elif freq == 'M':
            seasonal = 13
        elif freq == 'D':
            seasonal = 365
        ts = component.split('-')[0]
        method = component.split('-')[1] 
        df_ts_rs = gwl.seasonal_decomposition(sensor_code, df_ts_rs, scode, dtime, val, seasonal, method)
        df_ts_rs[val] = df_ts_rs[ts]

    # resample with given frequency from dropdown
    df_stress_s = df_stresses.loc[df_stresses[s_id]==stress_id].copy()
    stress_unit = df_stress_s['Units'].unique()[0]
    if stress_statistic == 'cumsum':
        df_stresses_rs = gwl.resample(df_stresses.loc[df_stresses[s_id]==stress_id], s_id, dtime, s_val, freq, 'sum')
        df_stresses_rs[s_val] = df_stresses_rs[s_val].cumsum()
    elif stress_statistic == 'cumdep':
        df_stresses_rs = gwl.resample(df_stresses.loc[df_stresses[s_id]==stress_id], s_id, dtime, s_val, freq, 'sum')
        df_stresses_rs['Dep'] = df_stresses_rs[s_val]-df_stresses_rs[s_val].mean()
        df_stresses_rs[s_val] = df_stresses_rs['Dep'].cumsum()
    else:
        df_stresses_rs = gwl.resample(df_stresses.loc[df_stresses[s_id]==stress_id], s_id, dtime, s_val, freq, stress_statistic)
    
    df_stresses_rs[s_val] = df_stresses_rs[s_val].interpolate()
    df_stresses_rs.dropna(subset=s_val, inplace=True)

    # merge the two datasets
    df_ts_vs_stress = pd.merge(df_ts_rs, df_stresses_rs, on=dtime)
    #st.write(df_ts_vs_stress)


    ###################### Figures ##################################

    # make a figure
    fig1 = make_subplots(specs=[[{'secondary_y':True}]])
    fig1.add_trace(
        go.Scattergl(
            x=df_ts_vs_stress[dtime],
            y=df_ts_vs_stress[val],
            name=sensor_code,
            mode='lines+markers'
        )
    )
    fig1.add_trace(
        go.Scattergl(
            x=df_ts_vs_stress[dtime],
            y=df_ts_vs_stress[s_val],
            name=stress_id,
            mode='lines+markers'
        ),
        secondary_y=True
    )

    fig1.update_yaxes(title='Groundwater Level [m]', secondary_y=False)
    fig1.update_yaxes(title=f'{stress_id} [{stress_unit}]', secondary_y=True)
    
    fig1.update_layout(title=f'{sensor_code} Groundwater Levels vs {stress_statistic.capitalize()} {stress_id}: Raw Data at Frequency = {freq}',
                       height=500,
                       font=dict(family='Arial', size=16),
                       )

    
    # make a figure 
    fig2 = go.Figure()

    fig2.add_trace(
        go.Scattergl(
            x=df_ts_vs_stress[s_val],
            y=df_ts_vs_stress[val],
            mode='markers',
            marker=dict(color='black'),
        )
    )
    r_p = gwl.sigfigs(pearsonr(df_ts_vs_stress[s_val], df_ts_vs_stress[val]).statistic, 2)
    r_s = gwl.sigfigs(spearmanr(df_ts_vs_stress[s_val], df_ts_vs_stress[val]).statistic, 2)
    

    fig2.update_xaxes(title=f'{stress_id} [{stress_unit}]')
    fig2.update_yaxes(title=f'Groundwater Level [m]')
    fig2.update_layout(title=f'{stress_id} vs {sensor_code} Groundwater Level',
                       height=500,
                       font=dict(family='Arial', size=16)
                       )
    fig2.add_annotation(x=0.9, y=0.9, xref='paper', yref='paper', text=f'Spearman R: {r_s} <br> Pearson R {r_p}')

    # make a figure
    fig3 = make_subplots(specs=[[{'secondary_y':True}]])
    fig3.add_trace(
        go.Scattergl(
            x=df_ts_vs_stress[dtime],
            y=df_ts_vs_stress[val].diff(),
            name=sensor_code,
            mode='lines+markers'
        )
    )
    fig3.add_trace(
        go.Scattergl(
            x=df_ts_vs_stress[dtime],
            y=df_ts_vs_stress[s_val].diff(),
            name=stress_id,
            mode='lines+markers'
        ),
        secondary_y=True
    )

    fig3.update_yaxes(title=f'Difference Groundwater Level [m/{freq}]', secondary_y=False)
    fig3.update_yaxes(title=f'Difference {stress_id} [{stress_unit}/{freq}]', secondary_y=True)
    
    fig3.update_layout(title=f'{sensor_code} Difference Groundwater Levels vs Difference {stress_statistic.capitalize()} {stress_id}: Raw Data at Frequency = {freq}',
                    height=500,
                    font=dict(family='Arial', size=16),
                    )
    # make a figure 
    fig4 = go.Figure()
    fig4.add_trace(
        go.Scattergl(
            x=df_ts_vs_stress[s_val].diff(),
            y=df_ts_vs_stress[val].diff(),
            mode='markers',
            marker=dict(color='black'),
        )
    )
    
    # add the diffs for dropna
    df_ts_vs_stress[f'{s_val}_Diff'] = df_ts_vs_stress[s_val].diff()
    df_ts_vs_stress[f'{val}_Diff'] = df_ts_vs_stress[val].diff()
    # drop na 
    df_ts_vs_stress_diff = df_ts_vs_stress.dropna(subset=[f'{s_val}_Diff', f'{val}_Diff'])
    # get stats
    r_p = gwl.sigfigs(pearsonr(df_ts_vs_stress_diff[f'{s_val}_Diff'], df_ts_vs_stress_diff[f'{val}_Diff']).statistic, 2)
    r_s = gwl.sigfigs(spearmanr(df_ts_vs_stress_diff[f'{s_val}_Diff'], df_ts_vs_stress_diff[f'{val}_Diff']).statistic, 2)
    

    fig4.update_xaxes(title=f'{stress_id} [{stress_unit}]')
    fig4.update_yaxes(title=f'Groundwater Level [m]')
    fig4.update_layout(title=f'{stress_id} vs {sensor_code} Groundwater Level',
                       height=500,
                       font=dict(family='Arial', size=16)
                       )
    fig4.add_annotation(x=0.9, y=0.9, xref='paper', yref='paper', text=f'Spearman R: {r_s} <br> Pearson R {r_p}')

    
    col7, col8 = st.columns([2, 1])
    with col7:
        st.plotly_chart(fig1, use_container_width=True)
    with col8:
        st.plotly_chart(fig2, use_container_width=True)        

    col9, col10 = st.columns([2, 1])
    with col9:
        st.plotly_chart(fig3, use_container_width=True)
    with col10:
        st.plotly_chart(fig4, use_container_width=True)

if __name__ == "__main__":
    main()