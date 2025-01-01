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
# Appen gör så att en tabell skrivs ut först
# sedan efter en minut skrivs ut en ny tabell vilket innehåller forecast
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
print(f"ADF Statistic: {adf_result[0]}, p-value: {adf_result[1]}")

######
# The ADF test results indicate that data is stationary since the p-value (0.00046),
# which significantly below the standard threshold of 0.05.
######

# Plot the temperature data
# plt.figure(figsize=(10, 6))
# plt.plot(train['CO2'], label='Train')
# plt.plot(test['CO2'], label='Test')
# plt.title('CO2 Levels Over Time')
# plt.xlabel('Date')
# plt.ylabel('CO2')
# plt.legend()
# plt.show()


train = train["CO2"]


#auto_model = auto_arima(
#    train, 
#    seasonal=False, 
#    trace=True, 
#    error_action='ignore', 
#    suppress_warnings=True,
#    stepwise=False,  # Disable stepwise search for a full grid search
#    max_p=6,         
#    max_d=3,         
#    max_q=6
#)
## Display the summary of the best model found
#auto_model.summary()


#######################################################################################
# Re-importing necessary libraries and re-loading the data due to execution state reset - Auday
# Jag tror inte vi behöver göra reimports av libraries, utan endast av datan
# import pandas as pd
# import matplotlib.pyplot as plt
#######################################################################################
# För att nästa del av koden ska köra så måste du stänga ner figure 1 # 
#######################################################################################

# Reload data
data = pd.read_csv('dataset.csv', index_col='date')

data.index.freq = '1min'
data.index = pd.DatetimeIndex(data.index).to_period('1min')
temp_data = data['CO2']

# Split the data into train and test sets (70% train, 30% test)
split_point = int(0.7 * len(temp_data))
train, test = temp_data[:split_point], temp_data[split_point:]
                                      
# Use auto_arima to find the best ARIMA model for our data
# 2: The number of autoregressive terms (AR).
# 1: The degree of differencing (I).
# 2: The number of moving average terms (MA).                     
# Fit the ARIMA(2,1,2) model to the training data (auday use 4.0.2)
# Testing with other numbers seems to be either indifferent or cause flatlining
# 6,0,2 gave result
order = (4, 0, 12)
model = ARIMA(train, order=order)
fitted_model = model.fit()
print(fitted_model.summary())

# Make predictions on the test data
forecast_arima = fitted_model.forecast(steps=len(test))

# Make predictions using the stepwise ARIMA model

train.index = train.index.to_timestamp()
test.index = test.index.to_timestamp()
forecast_arima.index = forecast_arima.index.to_timestamp()

# Plot comparison of forecasts and actual values
plt.figure(figsize=(10, 6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test', color='green')
plt.plot(forecast_arima.index, forecast_arima, label=f'ARIMA {order} Forecast', linestyle='--')
plt.title('CO2 Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('CO2 Levels')
plt.legend()
plt.show()

# Evaluate Mean Absolute Error (MAE)
mae_arima = np.mean(abs(forecast_arima - test))

print(f"ARIMA(4,0,2) MAE: {mae_arima:.2f}")

# # Mean Absolute Error (MAE)
# MAE = np.mean(abs(forecast - test))
# print('Mean Absolute Error (MAE): ' + str(np.round(MAE, 2)))
