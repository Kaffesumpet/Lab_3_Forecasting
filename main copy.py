import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA


data = pd.read_csv('dataset.csv', index_col='date')

data.head()

data.describe()

train_size = int(len(data) * 0.7)
train, test = data.iloc[:train_size], data.iloc[train_size:]

adf_result = adfuller(train['CO2'])

print(adf_result)

train = train["CO2"]

auto_model = auto_arima(train, 
                        seasonal=True, 
                        trace=True, 
                        error_action='ignore', 
                        suppress_warnings=True)

data = pd.read_csv('dataset.csv', index_col='date')

data.index.freq = '1min'

data.index = pd.DatetimeIndex(data.index).to_period('1min')

temp_data = data['CO2']

# Split the data into train and test sets (70% train, 30% test)
split_point = int(.7 * len(temp_data))
train, test = temp_data[:split_point], temp_data[split_point:]
                                                           
# Fit the ARIMA(2,1,2) model to the training data
model = ARIMA(train, order=(auto_model.order))
fitted_model = model.fit()

# Make predictions on the test data
forecast = fitted_model.forecast(steps=len(test))
print(forecast)

# Before plotting, convert the PeriodIndex back to DatetimeIndex
train.index = train.index.to_timestamp()
test.index = test.index.to_timestamp()
forecast.index = forecast.index.to_timestamp()

# Compare the forecasted values with the actual values
plt.figure(figsize=(10, 6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, forecast, label=f'Forecast {auto_model.order}')
plt.title('CO2 Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Temperature Levels')
plt.legend()
plt.show()

print(forecast.shape, test.shape)

# # Mean Absolute Error (MAE)
MAE = np.mean(abs(forecast - test))
print('Mean Absolute Error (MAE): ' + str(np.round(MAE, 2)))
