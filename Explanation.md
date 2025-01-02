[Intro]
Heya!!!!!!! (LinkVoice done by Mona)
Terraria Oof sound by Thomas
Nice by Filip
Radio Silence provided Nicklas

In this lab, we aim to explore and forecast sensor data, specifically CO2 levels, by leveraging time series modeling techniques. The dataset consists of CO2 measurements, along with other environmental factors like temperature, humidity, and light. Our primary goal is to apply the ARIMA model to forecast CO2 levels and evaluate its performance using various methods.

The first task involves analyzing the stationarity of the CO2 data, which is a crucial step before applying time series models. We use the Augmented Dickey-Fuller (ADF) test to check for stationarity, as non-stationary data requires transformation (like differencing) to ensure the model's accuracy. Once stationarity is confirmed, we apply the auto_arima function to automatically identify the best ARIMA model parameters for the CO2 dataset.

Following this, we split the data into training and test sets, with 70% of the data used for training and 30% for testing. We implement a stepwise training approach for forecasting, updating the model iteratively as we progress through the test data. The forecasts are then compared to the actual observed values, and the Mean Absolute Error (MAE) is calculated to assess the model's performance.

Finally, we visualize the forecasts through lineplot, scatterplots and correlation matrices to explore relationships between various variables (temperature, humidity, light, and CO2) and help better understand the data.

This lab aims to demonstrate the application of time series forecasting on sensor data, providing insights into how predictive models can be used for real-world data analysis and automation tasks.

[ADFTest]
The ADF (Augmented Dickey-Fuller) test checks the stationarity of the CO2 data in the training set. Stationarity is a crucial requirement for time series models like ARIMA, as it ensures the data's statistical properties (e.g., mean and variance) remain consistent over time. The test outputs several statistics, including the test statistic, p-value, and critical values, which help determine whether the null hypothesis of non-stationarity can be rejected. A p-value below a significance threshold (e.g., 0.05) indicates stationarity, meaning the data is suitable for modeling without transformations like differencing.

[ARIMA] #Directly from TEACHER, vet ej om detta ska vara med
Autoregressive (AR): This component uses lagged values of the time series to predict future values. The number of lagged values used is represented by the parameter 'p' in the ARIMA model.
Integrated (I): This part involves differencing the time series to achieve stationarity. The degree of differencing is denoted by 'd'. A stationary time series is one whose statistical properties like mean and variance do not change over time.
Moving Average (MA): The MA component models the error of the model as a combination of previous error terms. The parameter 'q' represents the size of the moving average window.

[AutoARIMAModel]
This step uses the auto_arima function to automatically find the best ARIMA model for the training data, CO2. The model is fitted with seasonal settings, meaning it accounts for repeating patterns or trends over time.

The trace=True parameter outputs the tested models and their respective AIC (Akaike Information Criterion) scores, which measure model quality. A lower AIC score indicates a better fit. By setting error_action='ignore' and suppress_warnings=True, the function handles errors or warnings gracefully, skipping problematic configurations. The resulting model balances simplicity and predictive accuracy, tailored to the dataset's characteristics.

[StepwiseUpdate]
The stepwise update and forecast process involves iteratively fitting an ARIMA model to the training data, forecasting a small batch of future values, and updating the model with actual observed values from the test set. At each iteration, a batch of test data is selected, and the ARIMA model is trained on the current training dataset (current_train). The model then forecasts the next batch_size values, which are stored in the forecasts list. The training data is updated by appending the observed test values from the current batch, enabling the model to incorporate the latest real-world data for improved future predictions. The true observed values are stored for evaluation, ensuring accuracy can be assessed, such as by calculating the Mean Absolute Error (MAE).

[CO2FvA]
This line chart shows CO2 levels over time, comparing actual data (split into training and testing sets) with predictions from a stepwise forecasting model. The blue line represents the training data, the orange line corresponds to the test set, and the green line shows the stepwise forecast.

The chart demonstrates forecasts with different batch sizes (77, 22, and 2). Smaller batch sizes allow the model to update more frequently, leading to improved alignment between the forecast and test data, as seen in the reduced prediction error. This iterative approach proves effective in capturing trends and fluctuations, but accuracy comes at the cost of computional power. Batch size 2 requires 1463 calculations while batch size 77 only require 38.

[CorrelationMatrix]
This correlation matrix visually represents the relationships between four variables:
Temperature, Humidity, Light, and CO2.

Positive correlations are shown in warm colors, while negative correlations are in cool colors. For example, Temperature and Light have a strong positive correlation of 0.71, indicating they increase together. In contrast, Temperature and Humidity have a moderate negative correlation of -0.47, meaning as one rises, the other tends to decrease. CO2 shows weak correlations with the other variables, slight positives with Temperature and Light and a very slight negative with Humidity.
This like means that there are other factors other than Temperature, Humidity and Light that has a stronger relationship with CO2. Perhaps Temperature and Light indicate that perhaps time of day or trafic?

[ScatterPlots]
This scatterplot matrix provides a detailed view of pairwise relationships between Temperature, Humidity, Light, and CO2. Each scatterplot shows how two variables interact, while histograms along the diagonal illustrate the distribution of each variable individually. For instance, the plot between Temperature and Light highlights a positive trend, aligning with the correlation we observed earlier. Other plots, such as Temperature versus Humidity, show a negative relationship. AKA, same as correlation matrix.

[MAE]
The Mean Absolute Error (MAE) represents the average of the absolute differences between the predicted and actual values in the dataset. In the case of batch size 2, the MAE is 11.99, meaning the model’s predictions for CO2 levels, on average, differ by about 11.99 units from the observed values. In contrast, with batch size 77, the MAE increases to 54.61, indicating a significantly larger prediction error. This further highlights the impact of batch size on model accuracy—smaller batch sizes allow the model to update more frequently, leading to better alignment with the observed data, while larger batch sizes may reduce the model's sensitivity to new data and result in less accurate forecasts.

[Outro]
Kebabpizza