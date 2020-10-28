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



#
# Funciones auxiliares
#
def graficar_predicciones(real, prediccion):
    plt.figure(figsize=(20,8))
    plt.plot(real[0:len(prediccion)],color='m', label='Valor real de la acción')
    plt.plot(prediccion, color='blue', label='Predicción de la acción')
    plt.ylim(1.1 * np.min(prediccion)/2, 1.1 * np.max(prediccion))
    plt.xlabel('Tiempo')
    plt.ylabel('kWh')
    plt.legend()
    plt.grid(True)
    plt.show()




#
# Lectura de los datos
#
dataset = pd.read_excel('HLAD_ORLIST2020.xlsx',index_col='Fecha', parse_dates=['Fecha'])
dataset.head()

#dataset['2019-12-26 00:00:00':].plot(figsize=(10,5))

set_entrenamiento = dataset['2019-09-01 00:00:00':'2019-12-25 00:00:00'].iloc[:,0:1]
set_validacion = dataset['2019-12-25 00:00:00':].iloc[:,0:1]
set_entrenamiento['Carga (MW)'].plot(legend=True,figsize=(14, 5))
set_validacion['Carga (MW)'].plot(legend=True,figsize=(14, 5))
plt.axvline(x = '2019-12-23 00:00:00', color='c', linewidth=2, linestyle='--')
plt.axvline(x = '2020-01-01 00:00:00', color='r', linewidth=2, linestyle='--')
plt.grid(True)
plt.legend(['Datos de aprendizaje de 2019-10-30 23:00:00 a 2019-12-20 00:00:00', 'Comprobacion 2019-12-20 00:00:00 a 2020-01-8 00:00:00' ])
plt.show()
 
# Normalización del set de entrenamiento
sc = MinMaxScaler(feature_range=(0,1))
set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)


# La red LSTM tendrá como entrada "time_step" datos consecutivos, y como salida 1 dato (la predicción a
# partir de esos "time_step" datos). Se conformará de esta forma el set de entrenamiento
time_step = 30
X_train  = []
Y_train   = []
m = len(set_entrenamiento_escalado)


for i in range(time_step,m):
    # X: bloques de "time_step" datos: 0-time_step, 1-time_step+1, 2-time_step+2, etc
    X_train.append(set_entrenamiento_escalado[i-time_step:i,0])
    

    # Y: el siguiente dato
    Y_train.append(set_entrenamiento_escalado[i,0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

# Reshape X_train para que se ajuste al modelo en Keras
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


#
# Red LSTM
#
dim_entrada = (X_train.shape[1],1)
dim_salida = 1
na = 80

modelo = Sequential()
modelo.add(LSTM(units=na, input_shape=dim_entrada))
modelo.add(Dense(units=dim_salida))
modelo.compile(optimizer='rmsprop', loss='mse')
modelo.fit(X_train,Y_train,epochs=30,batch_size=32)


#
# Validación (predicción del valor de las acciones)
#
x_test = set_validacion.values
x_test = sc.transform(x_test)

X_test = []
for i in range(time_step,len(x_test)):
    X_test.append(x_test[i-time_step:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

prediccion = modelo.predict(X_test)
prediccion = sc.inverse_transform(prediccion)

# Graficar resultados
plt.figure(figsize=(14, 5))
plt.plot(prediccion, color='b', label='Predicción de la acción',linewidth=1.5)
plt.plot(set_validacion.values, color='m', label='Predicción de la acción',linewidth=1.5)
plt.tick_params(labelsize = 10)
plt.grid(True)
plt.title('daily sale graph test_id=505 ',fontdict={'fontsize':30})
modelo.summary()

plt.title('Predicciones de Demanda Electrica [MWh]', family='Arial', fontsize=12)
plt.xlabel('Tiempo', family='Arial', fontsize=10)
plt.ylabel('Carga Electrica [MWh]', family='Arial', fontsize=10)
plt.xticks(rotation=45, fontsize=8)
plt.grid(True)
plt.show()



modelo.summary()



