Lab 3 - Forecasting sensor data

Background:
Sensors produce autonomous data which can be collected continuously or periodically. Sensors work like a minicomputer with its own set of operations. Working with sensor data will help to automate actions.

Purpose:
The purpose of this assignment is to learn how to analyse and forecast sensor data. Sensors produce data which can be collected over a period. This historical data can be utilized to analyse and perform predictions.

ARIMA:
Autoregressive (AR): This component uses lagged values of the time series to predict future values. The number of lagged values used is represented by the parameter 'p' in the ARIMA model.
Integrated (I): This part involves differencing the time series to achieve stationarity. The degree of differencing is denoted by 'd'. A stationary time series is one whose statistical properties like mean and variance do not change over time.
Moving Average (MA): The MA component models the error of the model as a combination of previous error terms. The parameter 'q' represents the size of the moving average window.

Tasks:

1. You need to analyse multiple columns of the provided dataset. 
   1.1. [ ] Show scatterplots and 
   1.2. [ ] Correlation matrices of chosen columns. 
   1.3. [ ] Explain what you understood.

2. [X] Use “adfuller” to understand if your data is stationary

3. [X] Use “auto_arima” to find the best ARIMA model parameters

4. [X] Forecasting: Train and test CO2 data with 70% training and 30% test. You can increase the training size up to 80%.

5. [ ] Plot your forecast with actual values as shown in the below figure. Use stepwise approach for better forecast values.

6. [ ] Show MAE value of the test

7. [ ] Discuss your results and provide your reflection


Presentation: Record a video in which you explain your results and upload it to your YouTube channel. Maximum time of recording should be 10 minutes.

Lab 3 needs to be submitted no later than Jan/13 23.59 in the submission folder on Learn/Assignment/Labs/Lab3.
Your target should be something like this:

HINT: Use Step wise training approach while testing

1. Initial Training: Train the ARIMA model on the initial training set.
2. Stepwise Update and Forecast:
   • For each data point in the test set, forecast the next step.
   • Update the model with the actual observed value.
   • Store the forecast and the true value for later evaluation.
   You can try batch updates

---

Material: All material for the tasks can be found in Lab2 folder Assignments/Lab 3.
Grade:
This assignment is graded as U/G.
G will be given if all the tasks are completed successfully.
U will be given if any task is not completed
Note: Everyone in the group should talk during the presentation.

Reference:
How to interpret ADFULLER test
