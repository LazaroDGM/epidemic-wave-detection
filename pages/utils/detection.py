import numpy as np
import pandas as pd
import wavedetection as wd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from scipy.optimize import least_squares

def location_data_in_weeks(location_data):
    location_data = location_data[['date', 'new_cases']]
    location_data['date'] = pd.to_datetime(location_data['date'])
    location_data.set_index('date', inplace=True)

    weeks = location_data.resample('W').sum()
    weeks.reset_index(0, inplace=True)

    return weeks

def plot_new_real_cases(weeks, location):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weeks['date'], y=weeks['new_cases'], name= 'Datos OWID'))
    fig.update_layout(title=f'Nuevos Casos Confirmados, {location}',
                    xaxis_title='Semana',
                    yaxis_title='Casos')
    return fig

def plot_smooth_new_cases(weeks, window, location):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weeks['date'], y=weeks['new_cases'], name= 'Datos OWID'))
    fig.add_trace(go.Scatter(x=weeks['date'], y=weeks['smooth'],
                            name= f'Datos Suavizados MM{window // 2}',
                            line= {'color':'red'}))
    fig.update_layout(title=f'Nuevos Casos Confirmados, {location}',
                    xaxis_title='Semana',
                    yaxis_title='Casos')
    return fig

def plot_diff_new_cases(weeks):
    positive = weeks[weeks['sign'] == 1]
    negative = weeks[weeks['sign'] == -1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weeks['date'], y=weeks['diff'],
                            line_color='black', name= 'Diferencias Finitas'))
    fig.add_trace(go.Scatter(x=positive['date'],
                            y=positive['diff'],
                            name= 'Positivo',
                            mode='markers',
                            marker_color='red'))
    fig.add_trace(go.Scatter(x=negative['date'],
                            y=negative['diff'],
                            name= 'Negativo',
                            mode='markers',
                            marker_color='blue'))
    fig.add_hline(y=0, line_dash='dash')
    fig.update_layout(title=f'Diferencias Finitas de la Serie de Tiempo Suavizada',
                    xaxis_title='Semana',
                    yaxis_title='Casos')
    return fig

def plot_monotony(weeks, location):
    positive = weeks[weeks['sign'] == 1]
    negative = weeks[weeks['sign'] == -1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weeks['date'], y=weeks['new_cases'],
                            line_color='black', name= 'Datos OWID'))
    fig.add_trace(go.Scatter(x=positive['date'],
                            y=positive['new_cases'],
                            name= 'Monotona Creciente',
                            mode='markers',
                            marker_color='red'))
    fig.add_trace(go.Scatter(x=negative['date'],
                            y=negative['new_cases'],
                            name= 'Monotona Decreciente',
                            mode='markers',
                            marker_color='blue'))
    fig.update_layout(title=f'Nuevos Casos Confirmados, {location}',
                    xaxis_title='Semana',
                    yaxis_title='Casos')
    return fig

def plot_first_waves(weeks, location):
    fig = go.Figure()
    fig.select_coloraxes()
    for i, wave in weeks.groupby('wave'):
        fig.add_trace(go.Scatter(x=wave['date'],
                                y=wave['new_cases'],
                                name= f'Ola {i+1}',
                                mode='lines',
                        ))
        fig.update_layout(title=f'Olas Epidemicas, Fase I, {location}',
                        xaxis_title='Semana',
                        yaxis_title='Casos')
    return fig

def plot_scaled_data(weeks, scale):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weeks['date'], y=weeks['new_cases_accumulated_norm'],
                            line_color='black', name= 'Datos OWID'))
    fig.update_layout(title=f'Casos Diarios por cada {scale} habitantes',
                    xaxis_title='Semana',
                    yaxis_title='Casos')
    return fig

def plot_richards_fitting(weeks, params, scale):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weeks['date'], y=weeks['new_cases_accumulated_norm'],
                            line_color='gray', name= 'Datos OWID'))

    for i, wave in weeks.groupby('wave'):
        rich = wd.richards(wave.index, *params.iloc[i])
        fig.add_trace(go.Scatter(x=wave['date'],
                                y=rich,
                                name= f'Ola {i+1}',
                                fill= 'tozeroy'))
    fig.update_layout(title=f'Casos Diarios por cada {scale} habitantes, Richards',
                    xaxis_title='Semana',
                    yaxis_title='Casos')
    return fig

def plot_gompertz_fitting(weeks, params, scale):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weeks['date'], y=weeks['new_cases_accumulated_norm'],
                            line_color='gray', name= 'Datos OWID'))

    for i, wave in weeks.groupby('wave'):
        rich = wd.gompertz(wave.index, *params.iloc[i])
        fig.add_trace(go.Scatter(x=wave['date'],
                                y=rich,
                                name= f'Ola {i+1}',
                                fill= 'tozeroy'))
    fig.update_layout(title=f'Casos Diarios por cada {scale} habitantes, Gompertz',
                    xaxis_title='Semana',
                    yaxis_title='Casos')
    return fig

def plot_first_fitting_separately(weeks, params_richards, params_gompertz, scale):
    figures = []
    for i, wave in weeks.groupby('wave'):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wave['date'],
                                    y=wd.richards(wave.index, *params_richards.iloc[i]) - wave['new_cases_accumulated_norm'].iloc[0],
                                    line_color='blue',
                                    name= 'Richards'))
        fig.add_trace(go.Scatter(x=wave['date'],
                                    y=wd.gompertz(wave.index, *params_gompertz.iloc[i]) - wave['new_cases_accumulated_norm'].iloc[0],
                                    line_color='red',
                                    name='Gompertz'))
        fig.add_trace(go.Scatter(x=wave['date'],
                                    y=wave['new_cases_accumulated_norm'] - wave['new_cases_accumulated_norm'].iloc[0],
                                    mode='markers',
                                    marker_color= 'black',
                                    name= 'Datos OWID'))
        fig.update_layout(title=f'Casos Diarios Acumulados por cada {scale} habitantes, Ola {i + 1}',
                        xaxis_title='Semana',
                        yaxis_title='Casos')
        figures.append(fig)
    return figures

def plot_second_waves():
    pass
