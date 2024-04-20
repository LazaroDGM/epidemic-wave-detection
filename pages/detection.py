import streamlit as st
import numpy as np
import pandas as pd
import wavedetection as wd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from scipy.optimize import least_squares

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

location_data = location_data[['date', 'new_cases']]
location_data['date'] = pd.to_datetime(location_data['date'])
location_data.set_index('date', inplace=True)

weeks = location_data.resample('W').sum()
weeks.reset_index(0, inplace=True)

weeks_interval = st.select_slider('Intervalo de Tiempo',
          options= weeks['date'],
          value=(weeks['date'].iloc[0], weeks['date'].iloc[-1]))

weeks = weeks[(weeks['date'] >= weeks_interval[0]) & (weeks['date'] <= weeks_interval[1])]

fig = go.Figure()
fig.add_trace(go.Scatter(x=weeks['date'], y=weeks['new_cases'], name= 'Datos OWID'))
fig.update_layout(title=f'Nuevos Casos Confirmados, {location}',
                   xaxis_title='Semana',
                   yaxis_title='Casos')
st.plotly_chart(fig)

# -------------------------------------

st.write('## Suavizado de los datos')

window = st.slider('Seleccione la ventana de suavizado para la Media Movil',
                   min_value=3,
                   max_value=55,
                   step=2,value=3)

weeks['smooth'] = weeks['new_cases'].rolling(window= window, center=True).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(x=weeks['date'], y=weeks['new_cases'], name= 'Datos OWID'))
fig.add_trace(go.Scatter(x=weeks['date'], y=weeks['smooth'],
                         name= f'Datos Suavizados MM{window // 2}',
                         line= {'color':'red'}))
fig.update_layout(title=f'Nuevos Casos Confirmados, {location}',
                   xaxis_title='Semana',
                   yaxis_title='Casos')
st.plotly_chart(fig)


diff = weeks['smooth'].to_numpy()
diff[1:] = diff[1:] - diff[:-1]
weeks['diff'] = diff

weeks['sign'] = weeks['diff'].apply(lambda x: 1 if x>=0 else -1)
weeks.loc[:window-1, 'sign'] = weeks['sign'].loc[window-1]#data.loc[window-1, 'sign']
weeks.loc[data.shape[0]-window + 2:, 'sign'] = 1

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
st.plotly_chart(fig)


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
st.plotly_chart(fig)

weeks['wave'] = 0
for i in range(1, weeks.shape[0]):
   if weeks['sign'].iloc[i-1] == -1 and int(weeks['sign'].iloc[i]) == 1:
      weeks.loc[i, 'wave'] = weeks['wave'].iloc[i-1] + 1
   else:
      weeks.loc[i, 'wave'] = weeks['wave'].iloc[i-1]

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
st.plotly_chart(fig)

# -----------------

st.write('## Reduccion de la escala de los datos')

weeks['new_cases_accumulated'] = weeks['new_cases'].cumsum()

scale = st.number_input('Escala', 1, int(1e9), int(1e6), 1)

weeks['new_cases_accumulated_norm'] = weeks['new_cases_accumulated'] / scale

fig = go.Figure()
fig.add_trace(go.Scatter(x=weeks['date'], y=weeks['new_cases_accumulated_norm'],
                         line_color='black', name= 'Datos OWID'))
fig.update_layout(title=f'Casos Diarios por cada {scale} habitantes',
                   xaxis_title='Semana',
                   yaxis_title='Casos')
st.plotly_chart(fig)


df_params_richards = pd.DataFrame(columns=['RMSE', 'alpha', 'gamma', 'tau', 'K', 'c'])

for i, wave in weeks.groupby('wave'):
    median = np.median(wave.index)
    min_cases = wave['new_cases_accumulated_norm'].iloc[0]
    total_cases = wave['new_cases_accumulated_norm'].iloc[-1]
    sol = least_squares(fun= wd.residuals_richards_parameters_vector,
                  x0= np.array([1, 1, median, total_cases, min_cases]),
                  args= [wave.index, wave['new_cases_accumulated_norm']],
                  bounds=(-1, np.inf)
                  )
    df_params_richards.loc[i] = [mean_squared_error(wd.richards(wave.index, sol.x[0], sol.x[1], sol.x[2], sol.x[3], sol.x[4]), wave['new_cases_accumulated_norm'], squared=True)
                                 , sol.x[0], sol.x[1], sol.x[2], sol.x[3], sol.x[4]]

st.write('Parametros de los modelos Richards ajustados a los datos')
st.dataframe(df_params_richards, use_container_width=True)

params_richards = ['alpha', 'gamma', 'tau', 'K', 'c']

fig = go.Figure()
fig.add_trace(go.Scatter(x=weeks['date'], y=weeks['new_cases_accumulated_norm'],
                         line_color='gray', name= 'Datos OWID'))

for i, wave in weeks.groupby('wave'):
   rich = wd.richards(wave.index, *df_params_richards[params_richards].loc[i])
   fig.add_trace(go.Scatter(x=wave['date'],
                           y=rich,
                           name= 'Datos OWID',
                           fill= 'tozeroy'))
fig.update_layout(title=f'Casos Diarios por cada {scale} habitantes, Richards',
                   xaxis_title='Semana',
                   yaxis_title='Casos')
   
st.plotly_chart(fig)

# ---------------------------------------

params_gompertz = ['beta', 'tau', 'K', 'c']
df_params_gompertz = pd.DataFrame(columns=['RMSE', 'beta', 'tau', 'K', 'c'])

for i, wave in weeks.groupby('wave'):
    median = np.median(wave.index)
    min_cases = wave['new_cases_accumulated_norm'].iloc[0]
    total_cases = wave['new_cases_accumulated_norm'].iloc[-1]
    sol = least_squares(fun= wd.residuals_gompertz_parameters_vector,
                  x0= np.array([1, median, total_cases, min_cases]),
                  args= [wave.index, wave['new_cases_accumulated_norm']],
                  bounds=(0, np.inf)
                  )
    df_params_gompertz.loc[i] = [mean_squared_error(wd.gompertz(wave.index, sol.x[0], sol.x[1], sol.x[2], sol.x[3]), wave['new_cases_accumulated_norm'], squared=True)
                                 , sol.x[0], sol.x[1], sol.x[2], sol.x[3]]

st.write('Parametros de los modelos Gompertz ajustados a los datos')
st.dataframe(df_params_gompertz, use_container_width=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=weeks['date'], y=weeks['new_cases_accumulated_norm'],
                         line_color='gray', name= 'Datos OWID'))
for i, wave in weeks.groupby('wave'):
   fig.add_trace(go.Scatter(x=wave['date'],
                           y=wd.gompertz(wave.index, *df_params_gompertz[params_gompertz].loc[i]),
                           name= 'Datos OWID',
                           fill= 'tozeroy'))
fig.update_layout(title=f'Casos Diarios por cada {scale} habitantes, Gompertz',
                   xaxis_title='Semana',
                   yaxis_title='Casos')
   
st.plotly_chart(fig)

st.write('## Primera reconstruccion de las Olas Epidemicas')

for i, wave in weeks.groupby('wave'):
   fig = go.Figure()
   fig.add_trace(go.Scatter(x=wave['date'],
                            y=wd.richards(wave.index, *df_params_richards[params_richards].loc[i]) - wave['new_cases_accumulated_norm'].iloc[0],
                            line_color='blue',
                            name= 'Richards'))
   fig.add_trace(go.Scatter(x=wave['date'],
                            y=wd.gompertz(wave.index, *df_params_gompertz[params_gompertz].loc[i]) - wave['new_cases_accumulated_norm'].iloc[0],
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
   st.plotly_chart(fig)


st.write('## Segunda')


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