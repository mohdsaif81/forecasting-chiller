import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

chillerdata = pd.read_csv('HVAC Energy Data.csv')
chillerdata.sort_index(inplace=True)
print(chillerdata.head())
print(chillerdata.info())
print(chillerdata.describe())

# Create a dataframe
df = chillerdata.drop(columns=["Local Time (Timezone : GMT+8h)"])


# Plot the variation of y with respect to x
plt.scatter(df['Chilled Water Rate (L/sec)'], df['Chiller Energy Consumption (kWh)'])
plt.xlabel('Chilled Water Rate (L/sec)')
plt.ylabel('Chiller Energy Consumption (kWh)')
plt.title('Variation of Chiller Energy Consumption (kWh) with respect to Chilled Water Rate (L/sec)')
plt.show()

# Plot the variation of y with respect to x
plt.scatter(df['Cooling Water Temperature (C)'], df['Chiller Energy Consumption (kWh)'])
plt.xlabel('Cooling Water Temperature (C)')
plt.ylabel('Chiller Energy Consumption (kWh)')
plt.title('Variation of Chiller Energy Consumption (kWh) with respect to Cooling Water Temperature (C)')
plt.show()

# Plot the variation of y with respect to x
plt.scatter(df['Building Load (RT)'], df['Chiller Energy Consumption (kWh)'])
plt.xlabel('Building Load (RT)')
plt.ylabel('Chiller Energy Consumption (kWh)')
plt.title('Variation of Chiller Energy Consumption (kWh) with respect to Building Load (RT)')
plt.show()

# Plot the variation of y with respect to x
plt.scatter(df['Outside Temperature (F)'], df['Chiller Energy Consumption (kWh)'])
plt.xlabel('Outside Temperature (F)')
plt.ylabel('Chiller Energy Consumption (kWh)')
plt.title('Variation of Chiller Energy Consumption (kWh) with respect to Outside Temperature (F)')
plt.show()

# Plot the variation of y with respect to x
plt.scatter(df['Dew Point (F)'], df['Chiller Energy Consumption (kWh)'])
plt.xlabel('Dew Point (F)')
plt.ylabel('Chiller Energy Consumption (kWh)')
plt.title('Variation of Chiller Energy Consumption (kWh) with respect to Dew Point (F)')
plt.show()

# Plot the variation of y with respect to x
plt.scatter(df['Humidity (%)'], df['Chiller Energy Consumption (kWh)'])
plt.xlabel('Humidity (%)')
plt.ylabel('Chiller Energy Consumption (kWh)')
plt.title('Variation of Chiller Energy Consumption (kWh) with respect to Humidity (%)')
plt.show()

# Plot the variation of y with respect to x
plt.scatter(df['Wind Speed (mph)'], df['Chiller Energy Consumption (kWh)'])
plt.xlabel('Wind Speed (mph)')
plt.ylabel('Chiller Energy Consumption (kWh)')
plt.title('Variation of Chiller Energy Consumption (kWh) with respect to Wind Speed (mph)')
plt.show()

# Plot the variation of y with respect to x
plt.scatter(df['Pressure (in)'], df['Chiller Energy Consumption (kWh)'])
plt.xlabel('Pressure (in)')
plt.ylabel('Chiller Energy Consumption (kWh)')
plt.title('Variation of Chiller Energy Consumption (kWh) with respect to Pressure (in)')
plt.show()

fig = plt.figure(figsize=(14,6))
sns.heatmap(df.corr(), linewidth = 0.6, cmap="YlGnBu", annot=True)
plt.show()

l= ["Chilled Water Rate (L/sec)","Cooling Water Temperature (C)", "Building Load (RT)", "Temperature (F)", "Dew Point (F)", "Humidity (%)", "Wind Speed (mph)", "Pressure (in)"] 
v= [0.85,0.31,0.92,0.56,0.13,0.54,0.42,0.15]
plt.title("Dependency of Energy Consumption on each of input Variable")
plt.bar(l,v)
plt.show()
