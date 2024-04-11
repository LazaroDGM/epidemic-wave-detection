import streamlit as st
import numpy as np
import wavedetection as wd
import plotly.express as px 



col1, col2 = st.columns([0.3, 0.7], gap='large')

alpha = col1.slider('Alpha value',
                0.00001,
                20.0,
                (2.0)
                )

gamma = col1.slider('Gamma value',
                0.0,
                5.0,
                (2.0)
                )

tau_R = col1.slider('Tau value',
                0.0,
                10.0,
                (5.0),
                key= 'tau_R'
                )

K_R = col1.slider('K value',
                0.0,
                1000.0,
                (400.0),
                key= 'K_R'
                )

lim_inf = 0
lim_sup = 10

t = np.linspace(lim_inf, lim_sup, 1000)
y = wd.richards(t, alpha, gamma, tau_R, K_R, 0)


fig = px.line(x=t, y=y, title='Richard Model') 
fig.add_vline(x=tau_R, line_dash="dash")
fig.add_hline(y=K_R * np.power(1+alpha, -1/alpha), line_dash="dash", line_color="red")

col2.plotly_chart(fig, use_container_width=True)


# --------------------------------------

col1, col2 = st.columns([0.3, 0.7], gap='large')

beta = col1.slider('$\\beta$ value',
                0.000001,
                20.0,
                (2.0)
                )

tau_G = col1.slider('Tau value',
                0.0,
                10.0,
                (5.0),
                key= 'tau_G'
                )

K_G = col1.slider('K value',
                0.0,
                1000.0,
                (400.0),
                key= 'K_G'
                )

t = np.linspace(lim_inf, lim_sup, 1000)
y = wd.gompertz(t, beta, tau_G, K_G, 0)

fig = px.line(x=t, y=y, title='Gompertz Model') 
fig.add_vline(x=tau_G, line_dash="dash")
fig.add_hline(y=K_G * np.exp(-1), line_dash="dash", line_color="red")

col2.plotly_chart(fig, use_container_width= True)