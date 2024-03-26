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

chillerdata = pd.read_csv('HVAC Energy Data.csv')
chillerdata.sort_index(inplace=True)
chillerdata.drop(columns=["Local Time (Timezone : GMT+8h)"],inplace=True)

X = chillerdata.drop(["Chiller Energy Consumption (kWh)"],axis=1)
y = chillerdata["Chiller Energy Consumption (kWh)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(62, input_shape=(8,), activation='relu'),
    tf.keras.layers.Dense(62, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(X_train, y_train, epochs=2000, validation_data=(X_test, y_test))
# Plot the loss curve
plt.plot(history.history['loss'], label='Training Loss')
plt.legend()
plt.savefig('loss.png')
plt.show()
model.save('mlp_model.h5')
tf.keras.utils.plot_model(
    model,
    to_file="model.png",
    dpi=200
)
