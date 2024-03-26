import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from matplotlib.pylab import rcParams
from datetime import datetime
import warnings
from pylab import rcParams
from sklearn.model_selection import train_test_split as split
import warnings
import itertools
warnings.filterwarnings("ignore")
from IPython import display
from matplotlib import pyplot
import os
import re
import seaborn as sns
import plotly.express as px
import warnings
from matplotlib.patches import Patch
import streamlit as st
import matplotlib.pyplot as plt

st.title("Predictive Analytics for HVAC: Data Science Stratagies for Power Consumption Forecasting in Chiller Systems")
data = pd.read_csv('HVAC Energy Data.csv')

bt = st.sidebar.radio("Menu", ["Home", "Upload Data", "PreProcess Data", "Training LSTM Model","Forecasting using LSTM Model", "Training BILSTM Model","Forecasting using BILSTM Model"])

if bt == "Home":
	st.header("Abstract")
	st.text(""" Heating, ventilating, and air conditioning (HVAC) systems play a significant role in ensuring occupant comfort 
and are one of the main energy consumers in buildings because more than half of the total building power is consumed by HVAC systems.
 The Chiller is an essential HVAC component that cools and dehumidifies the air in a wide variety of commercial, industrial, and institutional facilities. 
The Chiller consumes 25–40% of the total amount of electricity in a building. 
Over the last few decades, with a large increase in construction works, total energy consumption has increased while energy resources remain limited and hence the Energy demand management is crucial.
 Predicting and forecasting power consumption are essential parts of energy management systems. 
Predicting power consumption is used to evaluate engine performance in advanced power control and optimization and helps building managers make enhanced energy efficiency decisions. 
Forecasting power consumption is used to allocate electrical utility, safe and secure system operation, maintenance scheduling for energy savings, and also guidance for system energy optimization
""")

if bt=="Upload Data":
	st.header(" Details of the Dataset")
	st.text(""" The dataset used for this obtained from open source dataset 
published in Kaggle which is collected from a commercial building located in Singapore,
 from 18/08/2019 00:00 to 01/06/2020 13:00 which refined to 13,561 data samples after
 removing outliers and missing values. 
Feature Names:
•	Timestamp
•	Chilled Water Rate (L/sec) 
•	Cooling Water Temperature (C)
•	Building Load (RT)
•	Total Energy (kWh) 
•	Temperature (F)
•	Dew Point (F) 
•	Humidity (%)
•	Wind Speed (mph) 
•	Pressure (in) 
•	Hour of Day (h)  
•	Day of Week

""")
	st.subheader(" First Five samples of Data")
	st.text(data.head())
data['Date'] = pd.to_datetime(data['Local Time (Timezone : GMT+8h)'], infer_datetime_format=True)
print(data.info())
data=data.set_index(['Date'])
print(data.head())
data.reset_index(inplace=True)
data_feature_selected = data.drop(axis=1,labels=['Chilled Water Rate (L/sec)','Cooling Water Temperature (C)','Building Load (RT)','Outside Temperature (F)','Dew Point (F)','Humidity (%)','Wind Speed (mph)','Pressure (in)'])
col_order = ['Date','Chiller Energy Consumption (kWh)']
data_feature_selected = data_feature_selected.reindex(columns=col_order)

y= data_feature_selected["Chiller Energy Consumption (kWh)"]
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
y=scaler.fit_transform(np.array(y).reshape(-1,1))
##splitting dataset into train and test split
training_size=int(len(y)*0.65)
test_size=len(y)-training_size
train_data,test_data=y[0:training_size,:],y[training_size:len(y),:1]
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]    
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)

import numpy
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


if bt=="PreProcess Data":
	st.header("PreProcessing the Dataset")
	st.subheader("Checking for Missing Values")
	st.text(data.isna().sum())
	st.subheader("Dropping of all the features except date-time stamp and output variable")
	st.text(data_feature_selected)
	st.subheader("Performing MinMaxScaling on the output varaible")
	st.text(y)
	st.subheader("Splitting Data into Train and Test Sets")
	st.text("training_size,test_size")
	st.text(training_size)
	st.text(test_size)
	st.subheader("Shape of Xtrain")
	st.text(X_train.shape)
	st.subheader("Shape of ytrain")
	st.text(y_train.shape)
	st.subheader("Shape of Xtest")
	st.text(X_test.shape)
	st.subheader("Shape of ytest")
	st.text(ytest.shape)
	lx= ["No. of Train Samples", "No. of Test Samples"]
	lt= [training_size,test_size]
	fig= plt.figure()
	plt.title(" Splitting of Dataset")
	plt.bar(lx,lt)
	st.pyplot(fig)
	fig= plt.figure()
	plt.title(" Splitting of Dataset")
	plt.pie(lt,labels=lx,autopct='%.1f%%')
	st.pyplot(fig)


X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout ,BatchNormalization
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.initializers import RandomNormal, Constant

# # Build the LSTM Stack model
model=Sequential()
# Adding first LSTM layer
model.add(LSTM(150,return_sequences=True,input_shape=(100,1)))
model.add(Dropout(0.2)) # Dropout regularisation
# second LSTM layer 
model.add(LSTM(150,return_sequences=True))
# Adding third LSTM layer 
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.2))
# Adding fourth LSTM layer
model.add(LSTM(150))
model.add(Dropout(0.2))
# Adding the Output Layer
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=30, 
        verbose=1, mode='auto', restore_best_weights=True)
import tensorflow as tf
model = tf.keras.models.load_model('lstm_mod.h5')


### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = numpy.empty_like(y)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(y)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(y)-1, :] = test_predict

import time

if bt=="Training LSTM Model":
	st.subheader("Model Training for 20 Epochs")
	prg = st.progress(0)
	for i in range(100):
		time.sleep(0.1) 
		prg.progress(i+1)
	st.subheader("Model Structure of LSTM Network")
	model.summary(print_fn=lambda x: st.text(x))
	st.subheader(" Variation of Loss function with epochs")
	
	st.subheader("Root Mean Square Error for the test Data is")
	st.text(math.sqrt(mean_squared_error(y_train,train_predict)))
	st.subheader("Root Mean Square Error for the test Data is")
	st.text(mean_absolute_error(ytest, test_predict))
	st.subheader("plot the baseline and predictions for Test Data")
	fig, ax = plt.subplots(figsize=(20,10))
	plt.plot(scaler.inverse_transform(y))
	plt.plot(trainPredictPlot)
	plt.plot(testPredictPlot)
	plt.legend(['Actual Value','trainPredictPlot','testPredictPlot'])
	plt.xlabel('Time Steps')
	plt.ylabel('Energy Consumption')
	st.pyplot(fig)


# Future forecasting

len(test_data), len(train_data) # 2021-11-02 test-data last date
x_input=test_data[4666:].reshape(1,-1)
print(x_input.shape)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
lst_output1= scaler.inverse_transform(lst_output)  

if bt=="Forecasting using LSTM Model":
	st.subheader("prediction for next 30 days")
	st.text(lst_output1)
	import matplotlib.pyplot as plt
	fig = plt.figure()
	plt.plot(lst_output1)
	plt.xlabel('Time Steps')
	plt.ylabel('Energy Consumption')
	st.pyplot(fig)

print(" Using BILSTM Model")

# BILSTM
import math
from random import random
from numpy import array
from numpy import cumsum
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
import tensorflow as tf

tf.random.set_seed(1234)
modell = Sequential()
modell.add(Bidirectional(LSTM(100, activation='relu', input_shape=(100,1)))) #elu
#modell.add(Bidirectional(LSTM(50, dropout=0.5)))

#modell.add(Bidirectional(LSTM(100, dropout=0.5)))
#modell.add(BatchNormalization(momentum=0.6))
modell.add(Dense(1))
modell.compile(loss='mean_squared_error', optimizer='adam') #rmsprop adam

from tensorflow.keras.callbacks import EarlyStopping
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=30, 
        verbose=1, mode='auto', restore_best_weights=True)

new_model = tf.keras.models.load_model('BILSTM_mod.h5')

# Show the model architecture
#new_model.summary()

train_predict=modell.predict(X_train)
test_predict=modell.predict(X_test)

##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)
### Test Data RMSE
from sklearn.metrics import mean_squared_error

if bt=="Training BILSTM Model":
	st.subheader("Model Training for 50 Epochs")
	prg = st.progress(0)
	for i in range(100):
		time.sleep(0.1) 
		prg.progress(i+1)
	st.subheader("Model Structure of BILSTM Network")
	new_model.summary(print_fn=lambda x: st.text(x))
	st.subheader(" Variation of Loss function with epochs")
	st.text(math.sqrt(mean_squared_error(ytest,test_predict)))
	st.text(math.sqrt(mean_squared_error(y_train,train_predict)))

import math
from sklearn.metrics import mean_squared_error

### Test Data RMSE

import matplotlib.pyplot as plt

len(test_data), len(train_data) # 2021-11-02 test-data last date
x_input=test_data[4666:].reshape(1,-1)
print(x_input.shape)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = new_model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = new_model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
lst_output1= scaler.inverse_transform(lst_output)
if bt=="Forecasting using BILSTM Model":
	st.subheader("Prediction for next 30 days using BILSTM")
	st.text(lst_output1)
	import matplotlib.pyplot as plt
	fig = plt.figure()
	plt.plot(lst_output1)
	plt.xlabel('Time Steps')
	plt.ylabel('Energy Consumption')
	st.pyplot(fig)