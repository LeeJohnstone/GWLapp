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
df_ts = st.session_state['df_ts']
df_stresses = st.session_state['df_stresses']

    
def main():

    # Make landscape
    st.set_page_config(layout="wide")

    # App title
    st.markdown("<h1 style='text-align: left;'>Rainfall vs Groundwater Levels: Lag-Times</h1>", unsafe_allow_html=True)

    ######################### Selections ##########################
    scodes = df_ts[scode].unique()
    components = ['Observed', 'Seasonal', 'Trend', 'Residual']

    col1, col2, col3, col4, col5, col6 = st.columns(6, gap='small')
    
    with col1:# Dropdown menu to select a SensorCode
        sensor_code = st.selectbox('Select a SensorCode', scodes)
    with col2:
        stress_id = st.selectbox('Select a Stress ID', df_stresses['StressID'].unique())
    with col3:
        sd_type = st.selectbox('Select a Seasonal Decomposition Method', ['MA', 'STL'])
    with col4:
         stress_c = st.selectbox('Select Stress Component', components)
    with col5:
        ts_c = st.selectbox('Select Time-Series Component', components)
    with col6: 
        freq = st.selectbox('Select a Frequency', ['W', 'D', 'M'])
    
    ######################### Processing ############################
    
    # resample with given frequency from dropdown
    df_ts_s = df_ts.loc[df_ts[scode]==sensor_code].copy()
    df_ts_rs = gwl.resample(df_ts_s, scode, dtime, val, freq, 'median')
    #df_ts_rs.dropna(subset=val, inplace=True)

    if ts_c == 'Observed':
        df_ts_rs[val] = df_ts_rs[val].interpolate()
    else:
        if freq == 'W':
            seasonal = 53
        elif freq == 'D':
            seasonal = 365
        elif freq == 'M':
            seasonal = 13
        df_ts_rs = gwl.seasonal_decomposition(sensor_code, df_ts_rs, scode, dtime, val, seasonal, sd_type)
        df_ts_rs[val] = df_ts_rs[ts_c]

    # resample with given frequency from dropdown
    df_stress_s = df_stresses.loc[df_stresses[s_id]=="Rain"].copy()
    stress_unit = df_stress_s['Units'].unique()[0]
    df_stresses_rs = gwl.resample(df_stresses.loc[df_stresses[s_id]=="Rain"], s_id, dtime, s_val, freq, 'sum')
    df_stresses_rs['Dep'] = df_stresses_rs[s_val]-df_stresses_rs[s_val].mean()
    df_stresses_rs[s_val] = df_stresses_rs['Dep'].cumsum()

    # get component of cumdep rainfall
    if stress_c == 'Observed':
        df_stresses_rs[s_val] = df_stresses_rs[s_val]
    else:
        if freq == 'W':
            seasonal = 53
        elif freq == 'D':
            seasonal = 365
        elif freq == 'M':
            seasonal = 13
        df_stresses_rs = gwl.seasonal_decomposition("Rain", df_stresses_rs, 'StressID', dtime, s_val, seasonal, sd_type)
        df_stresses_rs[s_val] = df_stresses_rs[stress_c]
       
    # drop nans
    df_stresses_rs.dropna(subset=s_val, inplace=True)

    # merge the two datasets
    df_ts_vs_stress = pd.merge(df_ts_rs, df_stresses_rs, on=dtime)

    # method to find optimal lags
    if freq == 'W':
        lag_range = 48
        lag_steps = 1
    if freq == 'D':
        lag_range = 182
        lag_steps = 1
    if freq == 'M':
        lag_range = 11
        lag_steps = 1
    df_timelag, optimal_r, optimal_lag = gwl.time_lagged_xy(df_ts_vs_stress, s_val, val, lag_range, lag_steps, freq) 
    df_timelag['PlusErrY'] = df_timelag['CI_U']-df_timelag['R']
    df_timelag['MinErrY'] = df_timelag['R']-df_timelag['CI_L']
    
    
    

    ###################### Figures ##################################

    # make a figure
    fig1 = make_subplots(specs=[[{'secondary_y':True}]])
    fig1.add_trace(
        go.Scattergl(
            x=df_ts_vs_stress[dtime],
            y=df_ts_vs_stress[val],
            name=sensor_code,
            opacity=0.5,
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
    fig1.add_trace(
        go.Scattergl(
            x=df_ts_vs_stress[dtime],
            y=df_ts_vs_stress[val].shift(optimal_lag),
            name=f'Optimal Lag Time: {optimal_lag}{freq}',
            mode='lines+markers',
            marker=dict(color='orange'),
            line=dict(color='orange')
        )

    )

    fig1.update_yaxes(title='Groundwater Level [m]', secondary_y=False)
    fig1.update_yaxes(title=f'{stress_id} [{stress_unit}]', secondary_y=True)
    
    fig1.update_layout(title=f'{sensor_code} Groundwater Levels vs {stress_id}: Raw Data at Frequency = {freq}',
                       height=500,
                       legend=dict(orientation='h'),
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
            name='Original'
        )
    )
    fig2.add_trace(
            go.Scattergl(
                x=df_ts_vs_stress[s_val],
                y=df_ts_vs_stress[val].shift(optimal_lag),
                mode='markers',
                marker=dict(color='orange'),
                name=f'Shifted {optimal_lag}{freq}'
            )
        )

    # original stats
    r_p = gwl.sigfigs(pearsonr(df_ts_vs_stress[s_val], df_ts_vs_stress[val]).statistic, 2)
    #r_s = gwl.sigfigs(spearmanr(df_ts_vs_stress[s_val], df_ts_vs_stress[val]).statistic, 2)
    
    # optimised stats
    #r_p_o = gwl.sigfigs(pearsonr(df_ts_vs_stress[s_val][:optimal_lag], df_ts_vs_stress[val].shift(optimal_lag).dropna()).statistic, 2)
    #r_s_o = gwl.sigfigs(spearmanr(df_ts_vs_stress[s_val][:optimal_lag], df_ts_vs_stress[val].shift(optimal_lag).dropna()).statistic, 2)

    fig2.update_xaxes(title=f'{stress_id} [{stress_unit}]')
    fig2.update_yaxes(title=f'Groundwater Level [m]')
    fig2.update_layout(title=f'{stress_id} vs {sensor_code} Groundwater Level',
                       height=500,
                       legend=dict(orientation='h'),
                       font=dict(family='Arial', size=16),
                       )
    fig2.add_annotation(x=0.05, 
                        y=0.95, 
                        xref='paper', 
                        yref='paper', 
                        text=f'Org Pearson R {r_p}', 
                        font=dict(size=14),
                        showarrow=False)
    fig2.add_annotation(x=0.05, 
                    y=0.88, 
                    xref='paper', 
                    yref='paper', 
                    text=f'Opt Pearson R {gwl.sigfigs(optimal_r, 2)}', 
                    font=dict(size=14, color='orange'),
                    showarrow=False)

    # lagged correlation 
    fig3 = px.scatter(df_timelag, x=f'Lag ({freq})', y='R', error_y_minus='MinErrY', error_y='PlusErrY')
    fig3.update_layout(title=f'Rain vs {sensor_code}: Time-lagged Correlation',
                       height=500,
                       font=dict(family='Arial', size=16),
                       margin=dict(l=80, r=80, b=80, t=100)
                       )
    fig3.update_traces(marker=dict(color='black'))

    # make a figure
    fig4 = make_subplots(specs=[[{'secondary_y':True}]])
    fig4.add_trace(
        go.Scattergl(
            x=df_ts_vs_stress[dtime],
            y=df_ts_vs_stress[val].diff(),
            name=sensor_code,
            mode='lines+markers'
        )
    )
    fig4.add_trace(
        go.Scattergl(
            x=df_ts_vs_stress[dtime],
            y=df_ts_vs_stress[s_val].diff(),
            name=stress_id,
            mode='lines+markers'
        ),
        secondary_y=True
    )
    fig4.add_trace(
            go.Scattergl(
                x=df_ts_vs_stress[dtime],
                y=df_ts_vs_stress[val].shift(optimal_lag).diff(),
                name='Optimal Lag',
                mode='lines+markers',
                marker=dict(color='orange'),
                line=dict(color='orange')
            ),
        )


    fig4.update_yaxes(title=f'Difference Groundwater Level [m/{freq}]', secondary_y=False)
    fig4.update_yaxes(title=f'Difference {stress_id} [{stress_unit}/{freq}]', secondary_y=True)
    
    fig4.update_layout(title=f'{sensor_code} Difference Groundwater Levels vs Difference {stress_id}: Raw Data at Frequency = {freq}',
                    height=500,
                    legend=dict(orientation='h'),
                    font=dict(family='Arial', size=16),
                    )
    # make a figure 
    fig5 = go.Figure()
    fig5.add_trace(
        go.Scattergl(
            x=df_ts_vs_stress[s_val].diff(),
            y=df_ts_vs_stress[val].diff(),
            mode='markers',
            marker=dict(color='black'),
            name='Original'
        )
    )
    # add the diffs for dropna
    df_ts_vs_stress[f'{s_val}_Diff'] = df_ts_vs_stress[s_val].diff()
    df_ts_vs_stress[f'{val}_Diff'] = df_ts_vs_stress[val].diff()
    # drop na 
    df_ts_vs_stress_diff = df_ts_vs_stress.dropna(subset=[f'{s_val}_Diff', f'{val}_Diff'])
    # get stats
    r_p = gwl.sigfigs(pearsonr(df_ts_vs_stress_diff[f'{s_val}_Diff'], df_ts_vs_stress_diff[f'{val}_Diff']).statistic, 2)
    

    fig5.update_xaxes(title=f'{stress_id} [{stress_unit}]')
    fig5.update_yaxes(title=f'Groundwater Level [m]')
    fig5.update_layout(title=f'{stress_id} vs {sensor_code} Groundwater Level',
                       height=500,
                       font=dict(family='Arial', size=16)
                       )
    fig5.add_annotation(x=0.05, 
                    y=0.95, 
                    xref='paper', 
                    yref='paper', 
                    text=f'Pearson R {r_p}', 
                    font=dict(size=14),
                    showarrow=False)


    col7, col8, col9 = st.columns([1.5, 0.8, 1], gap='large')
    with col7:
        st.plotly_chart(fig1, use_container_width=True)
    with col8:
        st.plotly_chart(fig2, use_container_width=True)    
    with col9:
        st.plotly_chart(fig3, use_container_width=True)    

    col10, col11, col12 = st.columns([1.5, 0.8, 1], gap='large')
    with col10:
        st.plotly_chart(fig4, use_container_width=True)
    with col11:
        st.plotly_chart(fig5, use_container_width=True)
    with col12:
        st.write('')
        st.write('')
        st.write('**DataFrame of Time Lagged Correlation**')
        st.dataframe(df_timelag, use_container_width=True)

if __name__ == "__main__":
    main()