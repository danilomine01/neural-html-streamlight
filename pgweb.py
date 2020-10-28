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
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@100&display=swap" rel="stylesheet">
		<div style=" font-family: 'Poppins';font-size: 2pt;background-color:white;padding:1px 1px 1px 1px;border-radius:10px">
		<h1 style="color:MAGENTA;text-align:center;">Company: Electric Data Consulting Group SAS </h1>
        <h1 style="color:#004A7C;text-align:center;"> Electric Ing. Cristopher Danilo Avila C. cdavilac@unal.edu.co </h1>
        </div>
		"""

components.html("""
			<style>
			* {box-sizing: border-box}
			body {font-family: Verdana, sans-serif; margin:0}
			.mySlides {display: none}
			img {vertical-align: middle;}
			/* Slideshow container */
			.slideshow-container {
			  max-width: 1000px;
			  position: relative;
			  margin: auto;
			}
			/* Next & previous buttons */
			.prev, .next {
			  cursor: pointer;
			  position: absolute;
			  top: 50%;
			  width: auto;
			  padding: 16px;
			  margin-top: -22px;
			  color: white;
			  font-weight: bold;
			  font-size: 18px;
			  transition: 0.6s ease;
			  border-radius: 0 3px 3px 0;
			  user-select: none;
			}
			/* Position the "next button" to the right */
			.next {
			  right: 0;
			  border-radius: 3px 0 0 3px;
			}
			/* On hover, add a black background color with a little bit see-through */
			.prev:hover, .next:hover {
			  background-color: rgba(0,0,0,0.8);
			}
			/* Caption text */
			.text {
			  color: #f2f2f2;
			  font-size: 15px;
			  padding: 8px 12px;
			  position: absolute;
			  bottom: 8px;
			  width: 100%;
			  text-align: center;
			}
			/* Number text (1/3 etc) */
			.numbertext {
			  color: #f2f2f2;
			  font-size: 12px;
			  padding: 8px 12px;
			  position: absolute;
			  top: 0;
			}
			/* The dots/bullets/indicators */
			.dot {
			  cursor: pointer;
			  height: 15px;
			  width: 15px;
			  margin: 0 2px;
			  background-color: #bbb;
			  border-radius: 50%;
			  display: inline-block;
			  transition: background-color 0.6s ease;
			}
			.active, .dot:hover {
			  background-color: #717171;
			}
			/* Fading animation */
			.fade {
			  -webkit-animation-name: fade;
			  -webkit-animation-duration: 1.5s;
			  animation-name: fade;
			  animation-duration: 1.5s;
			}
			@-webkit-keyframes fade {
			  from {opacity: .4} 
			  to {opacity: 1}
			}
			@keyframes fade {
			  from {opacity: .4} 
			  to {opacity: 1}
			}
			/* On smaller screens, decrease text size */
			@media only screen and (max-width: 300px) {
			  .prev, .next,.text {font-size: 11px}
			}
			</style>
			</head>
			<body>
			<div class="slideshow-container">
			<div class="mySlides fade">
			  <div class="numbertext">1 / 3</div>
			  <img src="https://www.w3schools.com/howto/img_mountains_wide.jpg" style="width:100%">
			  <div class="text"></div>
			</div> 
			<div class="mySlides fade">
			  <div class="numbertext">2 / 3</div>
			  <img src="https://www.w3schools.com/howto/img_snow_wide.jpg" style="width:100%">
			  <div class="text"> </div>
			</div>
			<div class="mySlides fade">
			  <div class="numbertext">3 / 3</div>
			  <img src="https://www.w3schools.com/howto/img_nature_wide.jpg" style="width:100%">
			  <div class="text"> </div>
			</div>
			<a class="prev" onclick="plusSlides(-1)">&#10094;</a>
			<a class="next" onclick="plusSlides(1)">&#10095;</a>
			</div>
			<br>
			<div style="text-align:center">
			  <span class="dot" onclick="currentSlide(1)"></span> 
			  <span class="dot" onclick="currentSlide(2)"></span> 
			  <span class="dot" onclick="currentSlide(3)"></span> 
			</div>
			<script>
			var slideIndex = 1;
			showSlides(slideIndex);
			function plusSlides(n) {
			  showSlides(slideIndex += n);
			}
			function currentSlide(n) {
			  showSlides(slideIndex = n);
			}
			function showSlides(n) {
			  var i;
			  var slides = document.getElementsByClassName("mySlides");
			  var dots = document.getElementsByClassName("dot");
			  if (n > slides.length) {slideIndex = 1}    
			  if (n < 1) {slideIndex = slides.length}
			  for (i = 0; i < slides.length; i++) {
			      slides[i].style.display = "none";  
			  }
			  for (i = 0; i < dots.length; i++) {
			      dots[i].className = dots[i].className.replace(" active", "");
			  }
			  slides[slideIndex-1].style.display = "block";  
			  dots[slideIndex-1].className += " active";
			}
			</script>
			""",height=300)


components.html(html_temp,height=70)



html_temp = """
    <link href="https://fonts.googleapis.com/css2?family=Grandstander:wght@800&family=Roboto:wght@900&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@800&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@100&display=swap" rel="stylesheet">
		<div style=" font-family: 'Nunito';font-size: 22pt;background-color:white;padding:1px 1px 1px 1px;border-radius:10px">
		<h1 style="color:#00587a;text-align:center;">Electric Data Consulting Group SAS &#9410</h1>
        </div>
		"""

components.html(html_temp,height=200)


html_temp = """
    <link href="https://fonts.googleapis.com/css2?family=Grandstander:wght@800&family=Roboto:wght@900&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@800&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Open+Sans:ital,wght@1,300&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@100&display=swap" rel="stylesheet">
		<div style=" font-family: 'Nunito';font-size: 10pt;background-color:#2d6187;padding:1px 1px 1px 1px;border-radius:10px">
		<h1 style="color:white;text-align:center;">PAQUETE DE SIMULACIÓN PARA PREDICCIÓN DE LA DEMANDA ELECTRICA </h1>
        <div style=" font-family: 'Poppins';font-size: 8pt;background-color:WHITE;padding:1px 1px 1px 1px;border-radius:10px">
        <h1 style="color:MAGENTA;text-align:center;">Whith Artificial Neural Network and Machine Learning </h1>
		</div>
		"""
components.html(html_temp,height=190)  

components.html(""""Este paquete de simulación ofrece una herramienta efectiva para el análisis de datos ofreciendo diversos métodos y vararías opciones de potencia de cómputo según sus Necesidades"    """,height=70)


st.sidebar.title('Motores de Prediccion')



st.sidebar.write('puede tomar varios minutos dependiendo del dataset ...... ')

criterion=st.radio('Seleccione region par Analizar', ('Electrificadora del Huila', 'Electrificadora del Tolima'), key='criterion')

if  criterion == 'Electrificadora del Huila':
     
     dataset_train = pd.read_excel('HLAD_ORLIST2020.xlsx')
     st.dataframe( dataset_train.head(8))
   
       
if  criterion == 'Electrificadora del Tolima':
   
    dataset_train = pd.read_excel('TOLIMA.xlsx')
    st.dataframe(dataset_train.head(10))
    
    
st.success('Carga historica diaria por hora [MWh]')



cols = list(dataset_train)[1:3]

datelist_train = dataset_train['Fecha']
datelist_train = list(datelist_train.dt.to_pydatetime())

st.write('Dimensiones del Dataset = {}'.format(dataset_train.shape))
st.write('Unidades de tiempo Registradas= {}'.format(len(datelist_train)))
st.write('Parametros tomados para el calculo de prediccion {}'.format(cols))


dataset_train = dataset_train[cols].astype(str)
for i in cols:
    for j in range(0, len(dataset_train)):
        dataset_train[i][j] = dataset_train[i][j].replace(',', '')

dataset_train = dataset_train.astype(float)
dataset_train


training_set = dataset_train.values




sc = StandardScaler()
training_set_scaled = sc.fit_transform(training_set)
#st.write(training_set_scaled.shape)
#st.write('Normalizacion datos')
#training_set_scaled

sc_predict = StandardScaler()
sc_predict.fit_transform(training_set[:,0:1])

metrics = st.selectbox('Seleccione método de predicción:', ['Machine Learning(UNI-PRED)','Machine Learning(MULTI-PRED)(premium)'])
if metrics == 'Machine Learning(UNI-PRED)': 
   values = st.slider('Numero de Neuronas Adicionales a Activar (para mas de 100 suscribase y compre potecnia premiun)', float(dataset_train.TEMP.min()), 100., (10., 46.))
   metrics2 = ["Carga"]
   activities = st.multiselect("Seleccione Parametros serie de datos ",metrics2)
   if activities == ["Carga"]:
    Run=st.button("run")
    pics= []
    if  Run:
        progress_bar=st.progress(0)
        for i in range(100):
                if i <= 20:
                  time.sleep(0.2) # Sleep for 3 seconds
                  progress_bar.progress(i)
                elif i <= 50:
                  time.sleep(0.01) # Sleep for 3 seconds
                  progress_bar.progress(i)
                elif i <= 80:
                  time.sleep(0.01) # Sleep for 3 seconds
                  progress_bar.progress(i)
                elif i > 80:
                  time.sleep(0.01) # Sleep for 3 seconds
                  progress_bar.progress(i)
        if i == 99:
          st.success('MULTI-PREDCIT .... ready')
          pics = {"Cat": "unipredict01",
            "Puppy": "unipredict02.png",
            "pussy": "unipredict03.png",
            "Sci-fi city": "tt.png"}
          time.sleep(2) # Sleep for 3 seconds
          st.image("tt.png", use_column_width=True, caption='historico Temperatura')
          st.image("unipredict01.png", use_column_width=True, caption='historico carga')
          time.sleep(1) # Sleep for 3 seconds
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
          
          st.image("unipredict02.png", use_column_width=True)
          time.sleep(0.5) # Sleep for 3 seconds
          st.image("unipredict03.png", use_column_width=True)



















  
   def graficar_predicciones(real, prediccion):
       plt.figure(figsize=(20,8))
       plt.plot(real[0:len(prediccion)],color='m', label='Valor real ')
       plt.plot(prediccion, color='blue', label='Predicción ')
       plt.ylim(1.1 * np.min(prediccion)/2, 1.1 * np.max(prediccion))
       plt.xlabel('Tiempo')
       plt.ylabel('kWh')
       plt.legend()
       plt.grid(True)
       plt.show()
   
   
   
   

   dataset = pd.read_excel('HLAD_ORLIST2020.xlsx',index_col='Fecha', parse_dates=['Fecha'])
   dataset.head()
   
   #dataset['2019-12-26 00:00:00':].plot(figsize=(10,5))
   
   set_entrenamiento = dataset['2019-12-01 00:00:00':'2019-12-23 00:00:00'].iloc[:,0:1]
   set_validacion = dataset['2019-12-23 00:00:00':].iloc[:,0:1]
   set_entrenamiento['Carga (MW)'].plot(legend=True,figsize=(20,4))
   set_validacion['Carga (MW)'].plot(legend=True,figsize=(20,4))
   plt.axvline(x = '2019-12-20 00:00:00', color='c', linewidth=2, linestyle='--')
   plt.axvline(x = '2020-01-01 00:00:00', color='r', linewidth=2, linestyle='--')
   plt.grid(True)
   plt.legend(['Datos de aprendizaje de 2019-10-30 23:00:00 a 2019-12-20 00:00:00', 'Comprobacion 2019-12-20 00:00:00 a 2020-01-8 00:00:00' ])
   plt.show()

   sc = MinMaxScaler(feature_range=(0,1))
   set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)
   
 
   time_step = 80
   X_train  = []
   Y_train   = []
   m = len(set_entrenamiento_escalado)
   
   Run_rednet_2=st.sidebar.button("Run GPU Machine Learning(UNI-PRED) ")
   if Run_rednet_2:
      for i in range(time_step,m):
      
       X_train.append(set_entrenamiento_escalado[i-time_step:i,0])
       
   
    
       Y_train.append(set_entrenamiento_escalado[i,0])
       X_train, Y_train = np.array(X_train), np.array(Y_train)
       X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
   
   

       dim_entrada = (X_train.shape[1],1)
       dim_salida = 1
       na = 50


   
       modelo = Sequential()
       modelo.add(LSTM(units=na, input_shape=dim_entrada))
       modelo.add(Dense(units=dim_salida))
       modelo.compile(optimizer='rmsprop', loss='mse')
       modelo.fit(X_train,Y_train,epochs=5,batch_size=32)
     
       
       x_test = set_validacion.values
       x_test = sc.transform(x_test)
       
       X_test = []
       for i in range(time_step,len(x_test)):
           X_test.append(x_test[i-time_step:i,0])
       X_test = np.array(X_test)
       X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
       
       prediccion = modelo.predict(X_test)
       prediccion = sc.inverse_transform(prediccion)
       
 
       plt.figure(figsize=(20,8))
       plt.plot(prediccion, color='b', label='Predicción ',linewidth=1.5)
       plt.plot(set_validacion.values, color='m', label='Predicción ',linewidth=1.5)
       plt.tick_params(labelsize = 10)
       plt.grid(True)
       plt.title('_',fontdict={'fontsize':30})
       modelo.summary()
       
       plt.title('Predicciones de Demanda Electrica [MWh]', family='Arial', fontsize=12)
       plt.xlabel('Tiempo', family='Arial', fontsize=10)
       plt.ylabel('Carga Electrica [MWh]', family='Arial', fontsize=10)
       plt.xticks(rotation=45, fontsize=8)
       plt.grid(True)
       plt.show()
       
       
       #modelo.summary()

#################################################################################################################
############################################################################################################### 
# #################################################################################################################
###############################################################################################################    
if metrics == 'Machine Learning(MULTI-PRED)(premium)': 
   values = st.slider('Numero de Neuronas Adicionales a Activar (para mas de 100 suscribase y compre potecnia premiun)', float(dataset_train.TEMP.min()), 100., (10., 46.))
   metrics2 = ["Carga","Temperatura","irradiancia"]
   activities = st.multiselect("Seleccione Parametros serie de datos ",metrics2)
   
  
   if activities == ["Carga","Temperatura"]:
    Run1=st.button("run")
    pics= []
    if  Run1:
        progress_bar=st.progress(0)
        for i in range(100):
                if i <= 20:
                  time.sleep(0.2) # Sleep for 3 seconds
                  progress_bar.progress(i)
                elif i <= 50:
                  time.sleep(0.01) # Sleep for 3 seconds
                  progress_bar.progress(i)
                elif i <= 80:
                  time.sleep(0.01) # Sleep for 3 seconds
                  progress_bar.progress(i)
                elif i > 80:
                  time.sleep(0.01) # Sleep for 3 seconds
                  progress_bar.progress(i)
        if i == 99:
          st.success('MULTI-PRED ...... ready')
          pics = {"Cat": "pred01.png",
            "Puppy": "pred02.png",
            "Sci-fi city": "pred03.png"}
          time.sleep(2) # Sleep for 3 seconds
          st.image("pred01.png", use_column_width=True, caption='historico')
          time.sleep(1) # Sleep for 3 seconds
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
          
          st.image("pred02.png", use_column_width=True, caption='2019_12')
          time.sleep(0.5) # Sleep for 3 seconds
          st.image("pred03.png", use_column_width=True, caption='2019-12-21 00:00:00')
#################################################################################################################
###############################################################################################################  
##################################################################################################################
###############################################################################################################      
Run_rednet=st.sidebar.button("Run GPU Machine Learning(MULTI-PRED)(premium)")
if Run_rednet:
 X_train = []
 y_train = []
 
 n_future = 60   
 n_past = 90    
 
 for i in range(n_past, len(training_set_scaled) - n_future +1):
     X_train.append(training_set_scaled[i - n_past:i, 0:dataset_train.shape[1] - 1])
     y_train.append(training_set_scaled[i + n_future - 1:i + n_future, 0])
 
 X_train, y_train = np.array(X_train), np.array(y_train)
 
 #st.write('X_train shape == {}.'.format(X_train.shape))
 #st.write('y_train shape == {}.'.format(y_train.shape))
 
 
 from tensorflow.keras.models import Sequential
 from tensorflow.keras.layers import Dense
 from tensorflow.keras.layers import LSTM
 from tensorflow.keras.layers import Dropout
 from tensorflow.keras.optimizers import Adam
 
 
 model =tensorflow.keras.Sequential()
 model.add(LSTM(units=64, return_sequences=True, input_shape=(n_past, dataset_train.shape[1]-1)))
 model.add(LSTM(units=10, return_sequences=False))
 model.add(Dropout(0.25))
 model.add(Dense(units=1, activation='linear'))
 model.compile(optimizer = Adam(learning_rate=0.01), loss='mean_squared_error')
 
 
 
 
 
 
 
  # %%time
 es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
 rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
 mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
 
 tb = TensorBoard('logs')
 
 history = model.fit(X_train, y_train, shuffle=True, epochs=30, callbacks=[es, rlr, mcp, tb], validation_split=0.2, verbose=1, batch_size=256)
 
 
 datelist_future = pd.date_range(datelist_train[-1], periods=n_future, freq='1H').tolist()
 
 '''
 Remeber, we have datelist_train from begining.
 '''
 
 
 datelist_future_ = []
 for this_timestamp in datelist_future:
          datelist_future_.append(this_timestamp.date())
 predictions_future = model.predict(X_train[-n_future:])
 predictions_train = model.predict(X_train[n_past:])
 def datetime_to_timestamp(x):
     '''
         x : a given datetime value (datetime.date)
     '''
     return datetime.strptime(x.strftime('%Y%m%d %H:%M:%S'), '%Y%m%d %H:%M:%S') #'%Y%m%d %H:%M:%S'
 
 
 y_pred_future = sc_predict.inverse_transform(predictions_future)
 y_pred_train = sc_predict.inverse_transform(predictions_train)
 
 PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=['Carga (MW)']).set_index(pd.Series(datelist_future))
 PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=['Carga (MW)']).set_index(pd.Series(datelist_train[2 * n_past + n_future -1:]))
 
 
 PREDICTION_TRAIN.index = PREDICTION_TRAIN.index.to_series().apply(datetime_to_timestamp)
 
 PREDICTION_TRAIN.head(3)
 
 from pylab import rcParams
 rcParams['figure.figsize'] = 14, 5
 START_DATE_FOR_PLOTTING = '2019-01-01 00:00:00'	
 plt.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE['Carga (MW)'], color='r', label='Predicion 2020', linewidth=2)
 plt.plot(PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index, PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:]['Carga (MW)'], color='m', label='Tren de predicciones', linewidth=2)
 plt.plot(dataset_train.loc[START_DATE_FOR_PLOTTING:].index, dataset_train.loc[START_DATE_FOR_PLOTTING:]['Carga (MW)'], color='b', label='Carga real historica', linewidth=2)
 
 plt.axvline(x = min(PREDICTIONS_FUTURE.index), color='c', linewidth=2, linestyle='--')
 
 plt.grid(which='major', color='#cccccc', alpha=1)
 plt.legend(shadow=True)
 plt.title('Predicition de potencia', family='Arial', fontsize=12)
 plt.xlabel('tiempo', family='Arial', fontsize=10)
 plt.ylabel('Carga (MW)', family='Arial', fontsize=10)
 plt.xticks(rotation=45, fontsize=8)
 plt.show()
 # %%
