import streamlit as st
import tensorflow 
import numpy as np
np.random.seed(4)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import datetime as dt
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler
import streamlit.components.v1 as components
import time


html_temp = """
    <link href="https://fonts.googleapis.com/css2?family=Grandstander:wght@800&family=Roboto:wght@900&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@800&display=swap" rel="stylesheet">
		<div style=" font-family: 'Nunito';font-size: 14pt;background-color:DARKSLATEBLUE;padding:1px 1px 1px 1px;border-radius:10px">
		<h1 style="color:white;text-align:center;">PAQUETE DE SIMULACION PARA PREDICCION DE LA DEMANDA ELECTRICA DIARIA</h1>
		</div>
		"""
# components.html("<p style='color:red;'> Streamlit Components is Awesome</p>")
components.html(html_temp,height=220)









st.write('(dsfgghjhjhjgjghjadhdkjshfkjsdhfkjfhkjfhkdshkjs)')
components.html("<p style='color:magenta;'>(dsfgghjhjhjgjghjadhdk)</p>")

st.sidebar.title('Model Selection Panel')
#st.markdown('seleccione region')
st.sidebar.markdown('Choose your model and its parameters')





#desired_label = st.selectbox('Seleccione region:', ['Seleccion','Huila', 'Tolima'])


def load_data():
 data = pd.read_excel('HLAD_ORLIST.xlsx')
 return data

criterion=st.radio('Seleccione region', ('Huila', 'Tolima'), key='criterion')


if  criterion == 'Huila':
    df = pd.read_excel('HLAD_ORLIST.xlsx')
    st.dataframe(df.head(8))
   
       
if  criterion == 'Tolima':
    df = pd.read_excel('TOLIMA.xlsx')
    st.dataframe(df.head(10))
    
    
st.success('CURVA DE CARGA')








metrics = st.selectbox('Select prediction methods : ', ['ARIMA(UNI-PREDCIT)', 'MACHINLEARNING(UNI-PREDCIT)','MACHINLEARNING(MULTI-PREDCIT)(premium)'])




if metrics == 'MACHINLEARNING(MULTI-PREDCIT)(premium)': 
   metrics2 = ["Carga","Temperatura","irradiancia"]
   activities = st.multiselect("Selecionar Parametros",metrics2)
#################################################################################################################
############################################################################################################### 
# #################################################################################################################
###############################################################################################################    
if metrics == 'MACHINLEARNING(UNI-PREDCIT)': 
    metrics3 = ["Carga"]
    activities = st.multiselect("Selecionar Parametros",metrics3)

    Run=st.button("run")
    pics= []
    if  Run:
        progress_bar=st.progress(0)
        for i in range(100):
                if i <= 20:
                  time.sleep(0.3) # Sleep for 3 seconds
                  progress_bar.progress(i)
                elif i <= 50:
                  time.sleep(0.05) # Sleep for 3 seconds
                  progress_bar.progress(i)
                elif i <= 80:
                  time.sleep(0.1) # Sleep for 3 seconds
                  progress_bar.progress(i)
                elif i > 80:
                  time.sleep(0.03) # Sleep for 3 seconds
                  progress_bar.progress(i)
        if i == 99:
          pics = {"Cat": "pred01.png",
            "Puppy": "pred02.png",
            "Sci-fi city": "pred03.png"}
          st.image("sss.png", use_column_width=True, caption='historico')
          st.image("sss.png", use_column_width=True, caption='2019_12')
          st.image("sss.png", use_column_width=True, caption='2019-12-21 00:00:00')
     


    
 

   
    
#################################################################################################################
###############################################################################################################  
# #################################################################################################################
###############################################################################################################      

if metrics == 'ARIMA(UNI-PREDCIT)': 
    metrics4 = ["Carga"]
    activities = st.multiselect("Selecionar Parametros",metrics4)

activities

values = st.slider('Price range', float(df.TEMP.min()), 1000., (50., 300.))


  








st.markdown("## Party time!")
st.write("Yay! You're done with this tutorial of Streamlit. Click below to celebrate.")
btn = st.button("Celebrate!")
if btn:
    st.balloons()





#n_estimators = st.number_input('The number of trees in the forest', 100, 5000, step=10, key='n_estimators')

#st.button('Classify', key='classify')




 

# Reuse this data across runs!
#read_and_cache_csv = st.cache(pd.read_csv)

#BUCKET = "https://streamlit-self-driving.s3-us-west-2.amazonaws.com/"
#data = read_and_cache_csv(BUCKET + "labels.csv.gz", nrows=1000)
#st.write(data[data.label == desired_label])


