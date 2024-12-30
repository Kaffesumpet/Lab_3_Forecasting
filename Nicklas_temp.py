import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

#######################################################################################
# Jag var tvungen att uppdatera mitt Conda, installera pdarima i anaconda powershell - Nicklas
# conda install -c conda-forge pmdarima
#########################################################################################
# Appen gör så att en tabell skrivs ut först, 
# sedan efter varje minut skrivs ut en ny tabell vilket innehåller forecast
#########################################################################################

# Code provided by Auday
# Comments and fixes by Nicklas
# Jag tror koden härstammar från google colab, därav lite fixes här och där

data = pd.read_csv('dataset.csv', index_col='date')

# Display the first few rows to verify changes
data.head()

data.describe()

# Splitting the dataset into training and testing sets
train_size = int(len(data) * 0.7)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# Check for stationarity in the CO2 data
adf_result = adfuller(train['CO2'])

# Plot the temperature data
plt.figure(figsize=(10, 6))
plt.plot(train['CO2'], label='Train')
plt.plot(test['CO2'], label='Test')
plt.title('CO2 Levels Over Time')
plt.xlabel('Date')
plt.ylabel('CO2')
plt.legend()
plt.show()

adf_result # Fattar inte riktigt vad denna gör

train = train["CO2"]
# Use auto_arima to find the best ARIMA model for our data
auto_model = auto_arima(train, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)

# Display the summary of the best model found
auto_model.summary()


#######################################################################################
# Re-importing necessary libraries and re-loading the data due to execution state reset - Auday
# Jag tror inte vi behöver göra reimports av libraries, utan endast av datan
# import pandas as pd
# import matplotlib.pyplot as plt
#######################################################################################

# Reload data
data = pd.read_csv('dataset.csv', index_col='date')

data.index.freq = '1min'

data.index = pd.DatetimeIndex(data.index).to_period('1min')

temp_data = data['CO2']

# Split the data into train and test sets (70% train, 30% test)
split_point = int(0.7 * len(temp_data))
train, test = temp_data[:split_point], temp_data[split_point:]

# Fit the ARIMA(2,1,2) model to the training data
model = ARIMA(train, order=(4,0,2))
fitted_model = model.fit()

# Make predictions on the test data
forecast = fitted_model.forecast(steps=len(test))

# Before plotting, convert the PeriodIndex back to DatetimeIndex
train.index = train.index.to_timestamp()
test.index = test.index.to_timestamp()
forecast.index = forecast.index.to_timestamp()

# Compare the forecasted values with the actual values
plt.figure(figsize=(10, 6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, forecast, label='Forecast')
plt.title('CO2 Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Temperature Levels')
plt.legend()
plt.show()

print(forecast.shape, test.shape)

# # Mean Absolute Error (MAE)
MAE = np.mean(abs(forecast - test))
print('Mean Absolute Error (MAE): ' + str(np.round(MAE, 2)))
