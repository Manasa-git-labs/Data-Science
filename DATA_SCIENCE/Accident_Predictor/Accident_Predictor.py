#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
plt.style.use('ggplot')
import warnings
import itertools
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


# In[ ]:


df = pd.read_csv('datasets_331993_662385_accident_UK.csv')
df.head()
#df.tail()


# In[ ]:


df.info()


# **Convert the Date coloumn to Date type** 

# In[ ]:


df['Date'] = pd.to_datetime(df['Date'])
df.head()


# **Sorting the data by Date**

# In[ ]:


df = df.sort_values(by=['Date'])
df.head(5)


# **Set the Date for index**

# In[ ]:


accident = df.set_index('Date')
accident.index


# **Extract the average number of accidents in each month**

# In[ ]:


y = accident['Total_Accident'].resample('MS').mean()
y.head()


# **Visualize average number of accidents in each month**

# In[ ]:


y.plot(figsize=(15, 6))
plt.show()


# In[ ]:


from pylab import rcParams
import statsmodels.api as sm
rcParams['figure.figsize'] = 16, 10
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()


# **Different parameter combinations for seasonal ARIMA**

# In[ ]:


p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Different parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# **In this step, parameter selection for ARIMA Time Series Model is done.Our goal is to find the optimal set of parameters that yields the best performance for our model**

# In[ ]:


for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()
            
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# **Fitting the ARIMA model**

# In[ ]:


mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])


# In[ ]:


results.plot_diagnostics(figsize=(16, 8))
plt.show()


# **Evaluation of Forecasts**

# To understand the accuracy of the forecasts, predicted no of accidents is compared to real no of time series, the forecast is set to start from 2017-01-01 to end of date.

# In[ ]:


pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Total Accidents')
plt.legend()
plt.show()


# **Checking the accuracy of the model using MSE** 
# MSE is the average of the square of the difference between the observed and pridicted values.

# In[ ]:


y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))


# **Visualising Forecasts**

# We can observe from the graph the number of road accidents in UK will be declined in next years

# In[ ]:


pred_uc = results.get_forecast(steps=100)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Toatal Accident')
plt.legend()
plt.show()


# **Applying Prophet**

# Sort the values by Date

# In[ ]:


df = df.sort_values(by=['Date'])
df.head()


# Since Prophet requires the variable names in time series to be 
# 
# 
# *   y- Target
# *   ds- Datetime
# 
# 

# In[ ]:


df = df.rename(columns={'Date': 'ds',
                        'Total_Accident': 'y'})
df.head()


# **Visualising the number of road accident for each day**

# In[ ]:


ax = df.set_index('ds').plot(figsize=(15, 8))
ax.set_ylabel('Total Accident')
ax.set_xlabel('Date')

plt.show()


# **Fitting the Prophet model**

# Setting the uncertainity interval to 95%(the Prophet default is 80%)

# In[ ]:


from fbprophet import Prophet
my_model = Prophet(interval_width=0.95)
my_model.fit(df)


# To create forecast with our model we need some futre dates.
# Prphet provides make_future_dataframe to do so. Input the no of future periods and frequency. Created for 36 months/3 years.

# In[ ]:


future_dates = my_model.make_future_dataframe(periods=36, freq='MS')
future_dates.tail()


# yhat is forecast value

# In[ ]:


forecast = my_model.predict(future_dates)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# **Graph of actuals and forecast**

# In[ ]:


plt.figure(figsize=(10,8))
my_model.plot(forecast,
              uncertainty=True)


# **plot_components provides a graph of Trend and Seasonality**

# In[ ]:


my_model.plot_components(forecast)


# **Model Evaluation** 

# 

# In[ ]:


from fbprophet.diagnostics import cross_validation
df_cv = cross_validation(my_model, initial='730 days', period='180 days', horizon = '365 days')
df_cv.head()


# In[ ]:


from fbprophet.diagnostics import performance_metrics
df_p = performance_metrics(df_cv)
df_p.head()


# In[ ]:




