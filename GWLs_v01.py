import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from tslearn.clustering import TimeSeriesKMeans
import pastas as ps
import datetime as dt

def sigfigs(value, figures):
    '''
    Method to return value with a number of significant figures

    Parameter 
    ---------
    figures: int - number of significant figures
    value: float - value to be used

    Returns
    -------
    float
    '''
    x = float('%.*g' % (figures, value))
    return x

def resample(df, scode, dtime, val, freq, stat):
    '''
    Resample on time-series within a dataframe at a given frequency and given metric

    Parameters
    ----------
    df: pd.DataFrame - dataframe of time-series data with fields [scode, dtime, val], note no other fields returned in df_rs
    scode: str - string of the sensor code field name
    dtime: str - string of the datetime field name (must be in dt.datetime format)
    val: str - string of the value field to resample
    freq: str - string of the frequency field to resample at ['D', 'W', 'M', 'Y' or '1H', '2H' etc.]
    stat: str - string of the stat to be used ['mean', 'min', 'max', 'median']

    Return
    ------
    df_rs: pd.DataFrame
    '''
    if stat == 'mean':
        df_rs = df.groupby(scode)[[dtime, val]].resample(on=dtime, rule=freq).mean()
    elif stat == 'min':
        df_rs = df.groupby(scode)[[dtime, val]].resample(on=dtime, rule=freq).mean()
    elif stat == 'max':
        df_rs = df.groupby(scode)[[dtime, val]].resample(on=dtime, rule=freq).max()
    elif stat == 'median':
        df_rs = df.groupby(scode)[[dtime, val]].resample(on=dtime, rule=freq).median()
    elif stat == 'sum':
        df_rs = df.groupby(scode)[[dtime, val]].resample(on=dtime, rule=freq).sum()
   
    df_rs = df_rs.reset_index()

    # set the resampled points to center of sampling period
    #if freq == 'H':
    #    df_rs[dtime] = [i - dt.timedelta(hours=0.5) for i in df_rs[dtime]]
    #if freq == 'D':
    #    df_rs[dtime] = [i - dt.timedelta(hours=12) for i in df_rs[dtime]]
    if freq == 'W':
        df_rs[dtime] = [i - dt.timedelta(days=3.5) for i in df_rs[dtime]]
    if freq == 'M':
        df_rs[dtime] = [i - dt.timedelta(days=15) for i in df_rs[dtime]]
    if freq == 'Y':
        df_rs[dtime] = [i - dt.timedelta(days=182.5) for i in df_rs[dtime]]
    

    return df_rs

def normalise(series):
    '''
    Normalise the dataset to between 0 and 1 relative to itself.
    
    Parameters
    ----------
    series: pd.Series, e.g. field of a dataframe

    Returns
    -------
    series: pd.Series, normalised between 0 and 1
    
    '''
    if series.min() == series.max():
        series = pd.Series([0.5 for v in series])
    else:
        series = series + 0-series.min()
        series = (series - series.min()) / (series.max()-series.min())

    return series

def standardise(series):
    '''
    Standardised the dataset so that it is centred on 0 by subtracting the mean

    Parameters
    ----------
    series: pd.Series, e.g.field name of a df

    Returns
    -------
    series: pd.Series, standardised with a mean of 0
    '''

    series_std = series - series.mean()
    return series_std

def get_signatures(df, sensors, scode, dtime, val):

    '''
    Parameters
    ----------
    df: pd.DataFrame of all TS data
    sensors: list - sensors to use
    dtime: str - datetime field
    val: str - value field

    Returns
    -------
    df_signatures: pd.DataFrame

    '''
    signatures = []
    for s in sensors:
        try:
            # filter on scode
            df_s = df.loc[df[scode]==s].set_index(dtime).copy()
            df_s = df_s[val].dropna()
            # calculate signatures
            sig = ps.stats.signatures.summary(df_s)
            sig.name=s
            # append results
            signatures.append(pd.DataFrame(sig))
        except:
            pass
    # concatenate all
    df_signatures = pd.concat(signatures, axis=1)
    
    return df_signatures

def plot_signatures(df_signatures, dict_color):
    # create a figure object
    fig = go.Figure()
    for s in df_signatures.columns:
        x = df_signatures.index
        y = df_signatures[s].values
        if dict_color == None:
            line = {}
        else:
            line = dict(color=dict_color[s])   
        fig.add_trace(
            go.Scattergl(x=x,
                        y=y,
                        mode='lines',
                        line=line,
                        opacity=0.3,
                        name=s)
                        )   
        
    return fig

def plot_sd_ts(s, df_sd, scode, dtime, units):
    # make a figure with secondary y
    fig = make_subplots(rows=4, 
                        cols=1,
                        shared_xaxes=True, 
                        vertical_spacing=0.02,
                        specs=[[{"secondary_y": True}], 
                              [{"secondary_y": False}], 
                              [{"secondary_y": False}], 
                              [{"secondary_y": False}]])
    
    # set dtime as index
    df_sd.set_index(dtime, inplace=True)

    # extract seasonal decomposition
    obs, trend, seasonal, resid = df_sd['Observed'], df_sd['Trend'], df_sd['Seasonal'], df_sd['Residual']
    names = ['Observed', 'Trend', 'Seasonal', 'Residual']
    
    # add trace of each component as a line for top 3 rows
    for r, ts in enumerate([obs, trend, seasonal, resid]):
        if r<3:
            fig.add_trace(
                go.Scattergl(x=ts.index,
                            y=ts.values,
                            name=f'{names[r]} [{units}]'
                ),
                row=r+1,
                col=1
                )
        else:
            fig.add_trace(
                go.Bar(x=ts.index,
                            y=ts.values,
                            name=f'{names[r]} [{units}]'
                ),
                row=r+1,
                col=1
                )
            
    # get axis ranges based on data
    #axmin = min(np.min(obs), np.min(trend), np.min(seasonal), np.min(resid))
    #axmax = max(np.max(obs), np.max(trend), np.max(seasonal), np.max(resid))

    # update y titles
    ytitles = [f'Observed [{units}]', 
               f'Trend [{units}]',
               f'Seasonal [{units}]',
               f'Residual [{units}]']
    for i, yt in enumerate(ytitles):
        fig.update_yaxes(title=yt, 
                            row=i+1, 
                            secondary_y=False,
                            #range=[axmin+0.05*axmin, axmax+-0.05*axmin]
                            )

    # update layout
    fig.update_layout(title=f'Seasonal Decomposition of TS: {s}',
                    height=1000, 
                    width=800,
                    font=dict(family='Arial', size=14),
                    legend=dict(orientation='h'))
    
    return fig

def dtw_cluster(df, scode, dtime, val, mode, n_clusters, start_date, end_date, freq, stat, dict_color):
    
    # resample data for a given period
    df = df.loc[(df[dtime]>start_date) & (df[dtime]<end_date)].copy()
    df = df.dropna(subset=val)
    df_rs = resample(df, scode, dtime, val, freq, stat)

    # pivot resampled data
    df_rs_pivot = df_rs.pivot(index=dtime, columns=scode, values=val)
    df_rs_pivot = df_rs_pivot.loc[:, df_rs_pivot.isnull().mean() < .2].interpolate(method='linear', limit_direction='both') # remove sensors if >10% NAN in period

    # standardise/normalise if required

    # create empty array with dimensions nsensors, nrecords, 1
    arr = np.zeros(shape=[len(df_rs_pivot.columns), len(df_rs_pivot.index), 1])

    # for each sensor
    for i, s in enumerate(df_rs_pivot.columns):
        # add the records to the array 
        s_arr = df_rs_pivot.loc[:, s].values
        if mode == 'Normalised':
            # normalise the array
            if np.min(s_arr) == np.max(s_arr):
                proc_arr = np.array([0.5 for i in list(s_arr)])
            else:
                proc_arr = (s_arr - np.min(s_arr)) / (np.max(s_arr)-np.min(s_arr))
        elif mode == 'Standardised':
            # standardise array
            proc_arr = s_arr - np.mean(s_arr)
        else:
            proc_arr = s_arr
        # add procesed array of WLs
        arr[i] = proc_arr.reshape(len(df_rs['DTime'].unique()), 1)


    # set color palette for consistent colours
    # run k means
    km = TimeSeriesKMeans(n_clusters=n_clusters, verbose=True)
    y_pred = km.fit_predict(arr)

    # get sensor code for each group
    list_grp = []
    list_s = []
    for i, g in enumerate(y_pred):
        list_grp.append(str(g+1))
        list_s.append(df_rs_pivot.columns[i])
    df_groups = pd.DataFrame(data={'Group': list_grp, 'SensorCode': list_s})


    # max 4 cols
    cols = 3
    rows = int(np.ceil(n_clusters/cols))

    # make the figure of clusters
    fig = make_subplots(cols=cols, 
                        rows=rows,
                        subplot_titles=[f'Cluster: {str(i+1)}' for i in np.arange(0, n_clusters)],
                        vertical_spacing=0.1)

    for yi in np.arange(0,n_clusters):

        row = int(np.ceil((yi+1)/cols))
        col = int(yi%cols+1)

        for xx in arr[y_pred == yi]:
            xx = xx.reshape(len(xx))
            fig.add_trace(
                go.Scattergl(
                    x=df_rs_pivot.index,
                    y=xx,
                    opacity=0.2,
                    marker=dict(color='grey'),
                    legendgroup=str(yi),
                    legendgrouptitle = dict(text=str(yi+1)),
                    showlegend=False,
                ),
                row=row,
                col=col
            )

        fig.add_trace(
            go.Scattergl(
                x=df_rs_pivot.index,
                y=km.cluster_centers_[yi].ravel(),
                opacity=1,
                marker=dict(color=dict_color[yi]),
                legendgroup=str(yi),
                legendgrouptitle = dict(text=f'Cluster {yi+1}'),
                name=f'Centerpoint'
            ),
            row=row,
            col=col
            )

        

    fig.update_layout(
        height=300*rows,
        width=1200,
        title='Clusters of GWLs',
        font=dict(family='Arial', size=12))

    return fig, df_groups

def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation. 
    Parameters
    ----------
    lag : int, default = 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
        
    x=datax, 
    y=datay.shift(lag)
    df = pd.DataFrame(data={'X':datax, 'Y':y})
    df = df.dropna()
    return pearsonr(x=df['X'], y=df['Y'])

def time_lagged_xy(df, x, y, lag_range, lag_steps, freq):
    '''
    Method to get the optimal lag time between two time-series.

    Parameters
    -----------
    df: pd.DataFrame
    x: string - field name for x variable
    y: string - field name for y variable
    lag_steps: number of lags to review
    freq: frequency of dataset
    
    '''

    # get correlation 
    pearsons = crosscorr(df[x], df[y], lag=0)
    r = pearsons.correlation

    # find time lagged results
    # empty holder for results
    list_lag = []
    list_r = []
    list_ci_l = []
    list_ci_u = []

    # loop through lags in 5 day steps, upper limit is 10 months
    for lag in np.arange(-lag_range, 0, lag_steps):
        try:
            # get time-lagged correlation
            result = crosscorr(df[x], 
                        df[y], 
                        lag=lag)
            
            # add results to lists
            list_lag.append(lag)
            list_r.append(result.correlation)
            list_ci_l.append(result.confidence_interval().low)
            list_ci_u.append(result.confidence_interval().high)
        except:
            pass

    # get df of timelag results
    df_timelag = pd.DataFrame(data={f'Lag ({freq})':list_lag, 
                                    'R': list_r,
                                    'CI_L':list_ci_l,
                                    'CI_U': list_ci_u})

    # get best correlation lag
    optimal_lag = df_timelag.loc[df_timelag['CI_L'].idxmax(), f'Lag ({freq})']
    optimal_r = df_timelag.loc[df_timelag['CI_L'].idxmax(), 'R']

    # if unshifted is best, then set no lag
    if r > optimal_r:
        optimal_r = r
        optimal_lag = 0

    return df_timelag, optimal_r, optimal_lag