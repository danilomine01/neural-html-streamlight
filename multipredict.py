# Import modules and packages
import streamlit
import tensorflow 
import numpy as np
np.random.seed(4)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM , Dropout
from tensorflow.keras.optimizers import Adam
import datetime as dt
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard



#%matplotlib inline


# Importing Training Set
dataset_train = pd.read_excel('HLAD_ORLIST.xlsx')


# Select features (columns) to be involved intro training and predictions
cols = list(dataset_train)[1:3]
dataset_train

# Extract dates (will be used in visualization)
datelist_train = dataset_train['Fecha']

datelist_train = list(datelist_train.dt.to_pydatetime())

print('Training set shape == {}'.format(dataset_train.shape))
print('All timestamps == {}'.format(len(datelist_train)))
print('Featured selected: {}'.format(cols))

dataset_train = dataset_train[cols].astype(str)
for i in cols:
    for j in range(0, len(dataset_train)):
        dataset_train[i][j] = dataset_train[i][j].replace(',', '')

dataset_train = dataset_train.astype(float)
dataset_train

# Using multiple features (predictors)
training_set = dataset_train.values

print('Shape of training set == {}.'.format(training_set.shape))
training_set

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
training_set_scaled = sc.fit_transform(training_set)
training_set.shape

sc_predict = StandardScaler()
sc_predict.fit_transform(training_set[:,0:1])


# Creating a data structure with 90 timestamps and 1 output
X_train = []
y_train = []

n_future = 60   # Number of days we want top predict into the future
n_past = 100    # Number of past days we want to use to predict the future

for i in range(n_past, len(training_set_scaled) - n_future +1):
    X_train.append(training_set_scaled[i - n_past:i, 0:dataset_train.shape[1] - 1])
    y_train.append(training_set_scaled[i + n_future - 1:i + n_future, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

print('X_train shape == {}.'.format(X_train.shape))
print('y_train shape == {}.'.format(y_train.shape))

# Initializing the Neural Network based on LSTM
model =Sequential()

# Adding 1st LSTM layer
model.add(LSTM(units=64, return_sequences=True, input_shape=(n_past, dataset_train.shape[1]-1)))

# Adding 2nd LSTM layer
model.add(LSTM(units=10, return_sequences=False))

# Adding Dropout
model.add(Dropout(0.25))

# Output layer
model.add(Dense(units=1, activation='linear'))

# Compiling the Neural Network
model.compile(optimizer = Adam(learning_rate=0.01), loss='mean_squared_error')



##%%time
es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)

tb = TensorBoard('logs')

history = model.fit(X_train, y_train, shuffle=True, epochs=5, callbacks=[es, rlr, mcp, tb], validation_split=0.2, verbose=1, batch_size=256)

# Generate list of sequence of days for predictions
datelist_future = pd.date_range(datelist_train[-1], periods=n_future, freq='1H').tolist()

'''
Remeber, we have datelist_train from begining.
'''

# Convert Pandas Timestamp to Datetime object (for transformation) --> FUTURE
datelist_future_ = []
for this_timestamp in datelist_future:
    datelist_future_.append(this_timestamp.date())

  # Perform predictions
predictions_future = model.predict(X_train[-n_future:])

predictions_train = model.predict(X_train[n_past:])


# Inverse the predictions to original measurements

# ---> Special function: convert <datetime.date> to <Timestamp>
def datetime_to_timestamp(x):
    '''
        x : a given datetime value (datetime.date)
    '''
    return datetime.strptime(x.strftime('%Y%m%d %H:%M:%S'), '%Y%m%d %H:%M:%S') #'%Y%m%d %H:%M:%S'


y_pred_future = sc_predict.inverse_transform(predictions_future)
y_pred_train = sc_predict.inverse_transform(predictions_train)

PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=['Carga (MW)']).set_index(pd.Series(datelist_future))
PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=['Carga (MW)']).set_index(pd.Series(datelist_train[2 * n_past + n_future -1:]))

# Convert <datetime.date> to <Timestamp> for PREDCITION_TRAIN
PREDICTION_TRAIN.index = PREDICTION_TRAIN.index.to_series().apply(datetime_to_timestamp)

PREDICTION_TRAIN.head(3)

n=2
for x in range(n):
 # Set plot size 
 from pylab import rcParams
 rcParams['figure.figsize'] = 14, 5

# Plot parameters
 START_DATE_FOR_PLOTTING = '2019-11-23 00:00:00'	
 plt.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE['Carga (MW)'], color='r', label='Predicion 2020', linewidth=2)
 plt.plot(PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index, PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:]['Carga (MW)'], color='m', label='Tren de predicciones', linewidth=2)
 #plt.plot(dataset_train.loc[START_DATE_FOR_PLOTTING:].index, dataset_train.loc[START_DATE_FOR_PLOTTING:]['Carga (MW)'], color='b', label='Carga real historica', linewidth=2)
 


 plt.axvline(x = min(PREDICTIONS_FUTURE.index), color='c', linewidth=2, linestyle='--')

 plt.grid(which='major', color='#cccccc', alpha=1)
 plt.legend(shadow=True)
 plt.title('Predicition de potencia', family='Arial', fontsize=12)
 plt.xlabel('tiempo', family='Arial', fontsize=10)
 plt.ylabel('Carga (MW)', family='Arial', fontsize=10)
 plt.xticks(rotation=45, fontsize=8)
 plt.show()
 
 # Parse training set timestamp for better visualization
 dataset_train = pd.DataFrame(dataset_train, columns=cols)
 dataset_train.index = datelist_train
 dataset_train.index = pd.to_datetime(dataset_train.index)



