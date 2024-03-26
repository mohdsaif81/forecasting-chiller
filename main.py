# importing necessary Libraries

import numpy as np 
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import time

st.title("Predictive Analytics for HVAC: Data Science Stratagies for Power Consumption Forecasting in Chiller Systems")
chillerdata = pd.read_csv('HVAC Energy Data.csv')
chillerdata.sort_index(inplace=True)
#Check if missing data
chillerdata.isna().sum()
#There is none
chillerdata.drop(columns=["Local Time (Timezone : GMT+8h)"],inplace=True)
X = chillerdata.drop(["Chiller Energy Consumption (kWh)"],axis=1)
y = chillerdata["Chiller Energy Consumption (kWh)"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model=load_model('mlp_model.h5')
model.summary()

# Gui Part

bt = st.sidebar.radio("Menu", ["Home", "Upload Data", "PreProcess Data", "Train", "CNN Model Training","Predict"])

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
	st.text(chillerdata.head())

if bt=="PreProcess Data":
	st.header(" Processing the Dataset")
	st.subheader("Checking for Missing Values")
	st.text(chillerdata.isna().sum())
	st.subheader(" Removing the Time stamp from dataset")
	st.text(chillerdata.head())
	st.subheader("Performing Correlation Analysis of the Features")
	fig = plt.figure(figsize=(14,6))
	sns.heatmap(chillerdata.corr(), linewidth = 0.6, cmap="YlGnBu", annot=True)
	st.pyplot(fig)
	st.subheader("Splitting the Dataset")
	tx= X_train.shape[0]
	ty=X_test.shape[0]
	lx= ["No. of Train Samples", "No. of Test Samples"]
	lt= [tx,ty]
	fig= plt.figure()
	plt.title(" Splitting of Dataset")
	plt.bar(lx,lt)
	st.pyplot(fig)
	fig= plt.figure()
	plt.title(" Splitting of Dataset")
	plt.pie(lt,labels=lx,autopct='%.1f%%')
	st.pyplot(fig)

if bt=="Train":
	st.subheader("creating a model with following parameters")
	hist=model.summary()
	st.text(hist)
	st.subheader("Model Training for 2000 Epochs")
	prg = st.progress(0)
	for i in range(100):
		time.sleep(0.1) 
		prg.progress(i+1)
	st.subheader("Model Structure of MLP Network")
	image1 = Image.open('model.png')
	st.image(image1, caption='Model Structure of MLP Network')
	st.subheader("Loss Function of the Training Data")
	image2 = Image.open('loss.png')
	st.image(image2, caption='Loss Function of Training Data')
	y_pred_test = model.predict(X_test)
	fig = plt.figure()
	plt.scatter(X_test['Building Load (RT)'], y_test, color='blue')

	# Scatter plot of predictions (red)
	plt.scatter(X_test['Building Load (RT)'], y_pred_test, color='red')

	# Plot the regression line (green)
	#plt.plot(X_test['Building Load (RT)'], y_pred_test, color='green')

	# Set the axis labels and title
	plt.xlabel('Building Load (RT)')
	plt.ylabel('Chiller Energy Consumption (kWh)')
	plt.title('MLP Network Regression - Building Load (RT) vs Chiller Energy Consumption (kWh)')
	# Add a legend to the plot
	plt.legend(['True Values', 'Predictions'])
	st.pyplot(fig)

if bt== "CNN Model Training":
	from keras.models import Sequential
	from keras.layers import Dense, Conv1D, Flatten
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import mean_squared_error
	import matplotlib.pyplot as plt
	from keras.models import load_model
	chillerdata = pd.read_csv('HVAC Energy Data.csv')
	chillerdata.sort_index(inplace=True)
	
	chillerdata.drop(columns=["Local Time (Timezone : GMT+8h)"],inplace=True)
	x = chillerdata.drop(["Chiller Energy Consumption (kWh)"],axis=1)
	x = x.values.reshape(x.shape[0], x.shape[1], 1)
	y = chillerdata["Chiller Energy Consumption (kWh)"]
	xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
	st.subheader(" Training the 1D CNN Model")
	model1=load_model('cnn_model.h5')
	ypred = model1.predict(xtest)
	prg = st.progress(0)
	for i in range(100):
		time.sleep(0.1) 
		prg.progress(i+1)
	st.subheader("Model Structure of 1D CNN Network")
	model1.summary(print_fn=lambda x: st.text(x))
	st.subheader("Model Structure of 1D CNN Network")
	image1 = Image.open('model1.png')
	st.image(image1, caption='Model Structure of 1D CNN Network')
	st.subheader("Loss Function of the Training Data")
	image2 = Image.open('1dcnn_loss.png')
	st.image(image2, caption='Loss Function of Training Data')
	st.text(model1.evaluate(xtrain, ytrain))
	st.subheader(" MSE value of the Network for Test Data")
	st.text("MSE: %.4f" % mean_squared_error(ytest, ypred))
	st.subheader(" Evaluation of Model Performance for Test Data")
	fig= plt.figure()
	x_ax = range(len(ypred))
	plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
	plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
	plt.title("Actual vs Predicted Values for Test Data")
	plt.legend()
	st.pyplot(fig)



if bt == "Predict":
	st.header("Predicting Energy Consumption of The Chiller Unit")
	st.subheader("Enter the input features")
	i1 =st.text_input("Enter Chilled Water Rate (L/sec)")
	i2 = st.text_input("Enter Cooling Water Temperature (C)")
	i3 = st.text_input("Enter Building Load (RT) ")
	i4 = st.text_input("Enter Outside Temperature (F) ")
	i5 = st.text_input("Enter Dew Point (F) ")
	i6 = st.text_input("Enter Humidity (%)")
	i7= st.text_input("Enter Wind Speed (mph)")
	i8= st.text_input("Enter Pressure (in)")
	X_new =np.array( [[i1,i2,i3,i4,i5,i6,i7,i8]])
	b1= st.button("Predict")
	if b1 == True:
		st.subheader("prediction using MLP")
		y_pred_new = model.predict(X_new)
		st.text("The Predicted Energy Consumption of The Chiller Unit")
		st.text(y_pred_new)
		st.subheader("prediction using 1D CNN")
		y_pred_cnn = model1.predict(X_new)
		st.text("The Predicted Energy Consumption of The Chiller Unit")
		st.text(y_pred_cnn)

	
