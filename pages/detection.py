import streamlit as st
import numpy as np
import pandas as pd
import wavedetection as wd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error
from scipy.optimize import least_squares
import pages.utils.detection as ud

st.write('# Deteccion de Olas Epidemicas')

st.write('## Lectura de los datos')
st.write('Los datos utilizados fueron extraidos de https://ourworldindata.org/coronavirus .')



data = pd.read_csv(f'data/owid-covid-data.csv')
data['date'] = pd.to_datetime(data['date'])

location = st.selectbox(
    'Seleccione un pais',
    data['location'].unique(), index=list(data['location'].unique()).index('Mexico'))

location_data = data[data['location']==location]

show_data = st.checkbox('Mostrar todos los datos originales', value=False)

if show_data:
   st.dataframe(location_data)


# ----------- WEEKS ------------ #

weeks = ud.location_data_in_weeks(location_data)

weeks_interval = st.select_slider('Intervalo de Tiempo',
            options= weeks['date'],
            value=(weeks['date'].iloc[0], weeks['date'].iloc[-1]))
weeks = weeks[(weeks['date'] >= weeks_interval[0]) & (weeks['date'] <= weeks_interval[1])]

fig = ud.plot_new_real_cases(weeks, location)
st.plotly_chart(fig)

# -------------------------------------

st.write('## Suavizado de los datos')

window = st.slider('Seleccione la ventana de suavizado para la Media Movil',
                   min_value=3,
                   max_value=55,
                   step=2,value=3)
weeks = wd.first_waves_processing(weeks, window)

fig = ud.plot_smooth_new_cases(weeks, window, location)
st.plotly_chart(fig)


fig = ud.plot_diff_new_cases(weeks)
st.plotly_chart(fig)


fig = ud.plot_monotony(weeks, location)
st.plotly_chart(fig)

fig = ud.plot_first_waves(weeks, location)
st.plotly_chart(fig)

# -----------------

st.write('## Reduccion de la escala de los datos')


scale = st.number_input('Escala', 1, int(1e9), int(1e6), 1)

weeks['new_cases_accumulated'] = weeks['new_cases'].cumsum()
weeks['new_cases_accumulated_norm'] = weeks['new_cases_accumulated'] / scale

fig = ud.plot_scaled_data(weeks, scale)
st.plotly_chart(fig)


df_params_richards = wd.wave_richards_fitting(weeks, 'new_cases_accumulated_norm')
params_richards = ['alpha', 'gamma', 'tau_R', 'K_R', 'c_R']

st.write('Parametros de los modelos Richards ajustados a los datos')
st.dataframe(df_params_richards, use_container_width=True)


fig = ud.plot_richards_fitting(weeks, df_params_richards[params_richards], scale)
st.plotly_chart(fig)

# ---------------------------------------

params_gompertz = ['beta', 'tau_G', 'K_G', 'c_G']
df_params_gompertz = wd.wave_gompertz_fitting(weeks, 'new_cases_accumulated_norm')

st.write('Parametros de los modelos Gompertz ajustados a los datos')
st.dataframe(df_params_gompertz, use_container_width=True)

fig = ud.plot_gompertz_fitting(weeks, df_params_gompertz[params_gompertz], scale)
st.plotly_chart(fig)

show_first_fitting_separately = st.checkbox('Ver Comparacion por Olas del Ajuste entre Richards y Gompertz', value=False)

if show_first_fitting_separately:
   
   for fig in ud.plot_first_fitting_separately(weeks,
                  params_richards=df_params_richards[params_richards],
                  params_gompertz=df_params_gompertz[params_gompertz],
                  scale=scale):
      st.plotly_chart(fig)

st.write('## Segunda')

#'''
epsilon = 0.005
i = 0
while i < weeks['wave'].max():
   
   wave_i = weeks[weeks['wave'] == i]
   wave_i_inc = weeks[weeks['wave'] == i + 1]

   
   last_solution = df_params_gompertz[params_gompertz].loc[i].to_numpy()
   last_r = np.inf
   for m in range(1, wave_i_inc.index.shape[0]):
      temp_index = np.concatenate([wave_i.index.to_numpy(), wave_i_inc.index.to_numpy()[:m]])
      
      median = np.median(temp_index)
      min_cases = weeks['new_cases_accumulated_norm'].loc[temp_index[0]]
      total_cases = weeks['new_cases_accumulated_norm'].loc[temp_index[-1]]
      sol = least_squares(fun= wd.residuals_gompertz_parameters_vector,
                  x0= np.array([1, median, total_cases, min_cases]),
                  args= [temp_index, weeks['new_cases_accumulated_norm'].loc[temp_index]],
                  bounds=(0, np.inf)
                  )
      print('hola')
      r = mean_squared_error(wd.gompertz(temp_index, *sol.x), weeks['new_cases_accumulated_norm'].loc[temp_index], squared=False)
      
      st.write(r)

      if r > epsilon:
         break
      last_solution = sol.x
      last_r = r

   
   m = m-1
   index = np.concatenate([wave_i.index.to_numpy(), wave_i_inc.index.to_numpy()[:m]])
   weeks['wave'].loc[index] = i

   st.write(f'Se annadieron {m} observaciones a la Ola {i+1}')
   st.write(weeks)

   fix_wave_i = weeks[weeks['wave'] == i]

   st.write(sol.x)
   fig = go.Figure()
   fig.add_trace(go.Scatter(x=fix_wave_i['date'],
                            y=fix_wave_i['new_cases_accumulated_norm'] - fix_wave_i['new_cases_accumulated_norm'].iloc[0],
                            mode='markers',
                            marker_color= 'black',
                            name= 'Datos OWID'))
   fig.add_trace(go.Scatter(x=fix_wave_i['date'],
                            y=wd.gompertz(index, *last_solution) - fix_wave_i['new_cases_accumulated_norm'].iloc[0],
                            line_color='red',
                            name='Gompertz'))
   fig.update_layout(title=f'Casos Diarios Acumulados por cada {scale} habitantes, Ola {i + 1}',
                   xaxis_title='Semana',
                   yaxis_title='Casos',
                   shapes= [
                  dict(
                     type="rect",
                     x0=fix_wave_i['date'].iloc[-m],
                     x1=fix_wave_i['date'].iloc[-1],
                     y0=0,
                     y1=fix_wave_i['new_cases_accumulated_norm'].max(),
                     fillcolor="green",
                     opacity=0.6,
                     line_width=0,
                     layer="below"
                  )] if m > 0 else None)
   st.plotly_chart(fig)

   i+=1

   #'''

#    for j in range(4):
#      wd.richards(wave.index, *df_params_richards[params_richards].loc[i])
#      rmse = mean_squared_error(wd.richards(wave.index, *df_params_richards[params_richards].loc[i]),
#                        wave['new_cases_accumulated_norm'],
#                        squared=True)
#      sol = least_squares(fun= wd.residuals_richards_parameters_vector,
#                     x0= np.array([1, 1, median, total_cases, min_cases]),
#                     args= [wave.index, wave['new_cases_accumulated_norm']],
#                     bounds=(-1, np.inf)
#                     )
#    df_params_richards.loc[i] = [mean_squared_error(wd.richards(wave.index, sol.x[0], sol.x[1], sol.x[2], sol.x[3], sol.x[4]), wave['new_cases_accumulated_norm'], squared=True)
#                                 , sol.x[0], sol.x[1], sol.x[2], sol.x[3], sol.x[4]]
#