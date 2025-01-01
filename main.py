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

# Split the data into train and test sets (70% train, 30% test) per instructions.
split_point = int(.7 * len(temp_data))
train, test = temp_data[:split_point], temp_data[split_point:]
                                                           
# Model uses the output of Arima Auto, which is (2, 1, 2). 
model = ARIMA(train, order=(auto_model.order))
fitted_model = model.fit()

# Stepwise Update and Forecast
# [1, 2, 7, 11, 14, 19, 22, 38, 77, 133, 154, 209, 266, 418, 1463, 2926]
batch_size = 418
forecasts = []
true_values = []

# Start with the initial training data
current_train = train.copy()

# Iterate in batches
for i in range(0, len(test), batch_size):  
    batch = test.iloc[i:i+batch_size]
    print(f'{i} out of 2926')
    
    # Fit the ARIMA model on the current training data
    model = ARIMA(current_train, order=auto_model.order)
    fitted_model = model.fit()

    # Forecast the next 'batch_size' values
    forecast = fitted_model.forecast(steps=batch_size)
    forecasts.extend(forecast)  # Append the forecasted values

    # Update the training data with the actual observed values from the current batch
    current_train = pd.concat([current_train, batch])

    # Store the true values for later evaluation
    true_values.extend(batch)

# Ensure the indices are in the correct format
train.index = train.index.to_timestamp()
test.index = test.index.to_timestamp()

forecasts = pd.Series(forecasts, index=test.index)

plt.figure(figsize=(10, 6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(forecasts.index, forecasts, label='Stepwise Forecast')
plt.title('CO2 Stepwise Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Temperature Levels')
plt.legend()
plt.show()

# Calculate Mean Absolute Error (MAE)
MAE = np.mean(abs(forecasts - true_values))
print('Mean Absolute Error (MAE): ' + str(np.round(MAE, 2)))
