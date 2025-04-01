# Import necessary libraries
import pandas as pd
import streamlit as st
import datetime as dt
import GWLs_v01 as gwl




def main():

    scode = 'SensorCode'
    val = 'WL'
    dtime = 'DTime'

    # load datasets
    df_ts = st.session_state['df_ts']
    df_xy = st.session_state['df_xy']
    df_xy = df_xy.loc[df_xy[scode].isin(df_ts[scode].unique())]
    df_signatures = st.session_state['df_signatures']
    df_groups = st.session_state['df_groups']
    df_sd_var_all = st.session_state['df_sd_var']


    # Make landscape
    st.set_page_config(layout="wide")

    # App title
    st.markdown("<h1 style='text-align: left;'>Export to Leapfrog CSV</h1>", unsafe_allow_html=True)
    st.write('')
    st.write('')
    st.write('User must have uploaded files: df_xy, df_ts and run the following pages: Groundwater Signatures, Seasonal Decomposition and Clustering.')

    # Create two columns for the figures
    col1, col2, col3, col4 = st.columns(4, gap='small')

    with col1:
        start_date = st.date_input('Start Date for Statistics',
                                   dt.datetime(2020, 1, 1), 
                                   min_value=df_ts['DTime'].min(),
                                   max_value=df_ts['DTime'].max(),
                                   format='YYYY-MM-DD',
                                   label_visibility='visible'
                                   )
    with col2:
        end_date = st.date_input('End Date for Statistics',
                                  dt.datetime(2022, 1, 1), 
                                  min_value=df_ts['DTime'].min(),
                                  max_value=df_ts['DTime'].max(),
                                  format='YYYY-MM-DD',
                                  label_visibility='visible'
                                    )    
    with col3:
        inc_sigs = st.selectbox('Include Signatures?', ['Yes', 'No'])
   
    with col4: 
        inc_sd = st.selectbox('Include Seasonal Decomposition Signatures?', ['Yes', 'No'])

    # convert inputs to dtime
    start_dtime = dt.datetime(start_date.year, start_date.month, start_date.day)
    end_dtime = dt.datetime(end_date.year, end_date.month, end_date.day)

    # get summary statistics for various time-series
    df_ts_stats = df_ts.loc[(df_ts[dtime]>=start_dtime) & (df_ts[dtime]<=end_dtime)][[scode, val]].dropna(subset=val).groupby(scode).describe()
    df_ts_stats.columns = [f'{val}_Count', f'{val}_Mean', f'{val}_Std', f'{val}_Min', f'{val}_25%', f'{val}_Median', f'{val}_75%', f'{val}_Max']
    
    # set index to allow join
    df_xy_gw = df_xy.loc[df_xy[scode].isin(df_ts[scode].unique())].set_index(scode).drop(columns='Unnamed: 0')
    df_groups = df_groups.set_index(scode)
    df_signatures_t = df_signatures.transpose().reset_index().rename(columns={'index': scode}).set_index(scode)
    df_sd_var_all = df_sd_var_all.set_index(scode)

    list_dfs = [df_xy_gw, df_groups, df_ts_stats]
    
    if inc_sigs == 'Yes':    
        # merge for export to leapfrog
        list_dfs.append(df_signatures_t)
    else:
        pass
    if inc_sd == 'Yes':
        list_dfs.append(df_sd_var_all)
    else:
        pass
    # merge for export to leapfrog
    df_lf = pd.concat(list_dfs, axis=1)

    
    st.write(df_lf)

    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')


    csv = convert_df(df_lf)

    st.download_button(
    "Export to CSV",
    csv,
    "GWLs_LeapfrogExport.csv",
    "text/csv",
    key='download-lf'
    )

if __name__ == "__main__":
    main()