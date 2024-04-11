import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def sign(x):
    if x >= 0:
        return 1
    elif x<0:
        return -1
    else:
        return np.nan

def waves(data, p, max_window= 25):
    '''
    Algoritmo para detectar olas epidemicas

    Parametros
    ---
    `data`: `DataFrame` de `pandas`, con los datos de los casos diarios.
    Debe tener una columna nombrada "new_cases", esta informacion.

    `p`: `2*p` indica el tamano minimo de la ola.

    `max_window`: indica la ventana maxima permitida para el suavizado de los datos
    con Media Movil Centrada.

    Retorno
    ---
    Una copia del `DataFrame` original con las columnas anadidas:
        - "smooth": datos suavizados con Media Movil Centrada
        - "diff": diferencias finitas de los datos suavizados
        - "sign": signo de las diferencias finitas
        - "same_sign": numero del intervalo donde las diferencias finitas contiguas tienen el mismo signo
        - "wave": numero del intervalo de cada ola epidemica
    '''

    data = data.copy()
    window = 3
    while window <= max_window:
        data['smooth'] = data['new_cases'].rolling(window= window, center=True).mean()

        diff = data['smooth'].to_numpy()
        diff[1:] = diff[1:] - diff[:-1]
        data['diff'] = diff

        data['sign'] = data['diff'].apply(sign)
        data.loc[:window-1, 'sign'] = data['sign'].iloc[window-1]#data.loc[window-1, 'sign']
        data.loc[data.shape[0]-window + 2:, 'sign'] = 1 #data['sign'].iloc[-window+1]
        
        data['same_sign'] = 0
        for i in range(1, data.shape[0]):
            if data.loc[i-1, 'sign'] == data['sign'].iloc[i]:
                data.loc[i, 'same_sign'] = data['same_sign'].iloc[i-1]
            else:
                data.loc[i, 'same_sign'] = data['same_sign'].iloc[i-1] + 1
        if data.groupby('same_sign')['same_sign'].count().min() < p:
            window += 2
            continue
        data['wave'] = 0
        for i in range(1, data.shape[0]):
            if data['sign'].iloc[i-1] == -1 and data['sign'].iloc[i] == 1:
                data.loc[i, 'wave'] = data['wave'].iloc[i-1] + 1
            else:
                data.loc[i, 'wave'] = data['wave'].iloc[i-1]
        return data
    print('Se superó la ventana máxima permitida para la Media Móvil')
    return data

def richards(t, alpha, gamma, tau, K, c):
    return K / np.power((1 + alpha * np.exp(- alpha * gamma * (t - tau))), 1 / alpha) + c

def gompertz(t, beta, tau, K, c=0):
    return K * np.exp(- np.exp(- beta * (t - tau))) + c

def residuals_richards(alpha, gamma, tau, K, c, times, real):
    approx = richards(times, alpha, gamma, tau, K, c)
    return real - approx

def residuals_richards_parameters_vector(params, times, real):
    return residuals_richards(
        params[0],
        params[1],
        params[2],
        params[3],
        params[4],
        times,
        real
    )

def residuals_gompertz(beta, tau, K, c, times, real):
    approx = gompertz(times, beta, tau, K, c)
    return real - approx

def residuals_gompertz_parameters_vector(params, times, real):
    return residuals_gompertz(
        params[0],
        params[1],
        params[2],
        params[3],
        times,
        real
    )