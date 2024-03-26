import numpy as np 
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from tensorflow.keras.models import load_model
from PIL import Image
import time
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

chillerdata = pd.read_csv('HVAC Energy Data.csv')
chillerdata.sort_index(inplace=True)
#Check if missing data
chillerdata.isna().sum()
#There is none
chillerdata.drop(columns=["Local Time (Timezone : GMT+8h)"],inplace=True)
x = chillerdata.drop(["Chiller Energy Consumption (kWh)"],axis=1)
x = x.values.reshape(x.shape[0], x.shape[1], 1)
y = chillerdata["Chiller Energy Consumption (kWh)"]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv1D(32, 2, activation="relu", input_shape=(8,1)))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam")
model.summary()
history = model.fit(xtrain, ytrain, batch_size=12,epochs=500, verbose=0)
model.save("cnn_model.h5")
# Plot the loss curve
plt.plot(history.history['loss'], label='Training Loss')
plt.legend()
plt.savefig('loss.png')
plt.show()
ypred = model.predict(xtest)
print(model.evaluate(xtrain, ytrain))
print("MSE: %.4f" % mean_squared_error(ytest, ypred))

x_ax = range(len(ypred))
plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()
tf.keras.utils.plot_model(
    model,
    to_file="model.png",
    dpi=200
)
