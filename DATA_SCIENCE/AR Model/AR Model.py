#!/usr/bin/env python
# coding: utf-8

# In[10]:



#AR_mod Fixed

import pandas as pd
from matplotlib import pyplot
from statsmods.tsa.ar_mod import AR
from sklearn.metrics import mean_squared_error
t_s =pd.read_csv('daily-min-temp.csv',header=0,index_col=0)
#split the dataset
r=t_s.values
trainseries,test_series = r[1:len(r)-8],r[len(r)-8:]

#train for autoregression

mod = AR(trainseries)
mod_fitting = mod.fit()

print('Lag: %s'% mod_fitting.k_ar)
print('Coefficients: %s' % mod_fitting.params)


#make Prediction

predictn =mod_fitting.predict(start=len(trainseries),end= (len(trainseries)+len(test_series)-1),dynamic=False)
for i in range (len(predictn)):
     print('predicted=%f, expected=%f' % (predictn[i], test_series[i]))
     error = mean_squared_error(test_series, predictn)
     print('Test MSE: %.3f' % error)
# plot results
pyplot.plot(test_series)
pyplot.plot(predictn, color='green')
pyplot.show()


# In[11]:


import pandas as pd
#from pandas import datetime
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from pandas import DataFrame
from statsmods.tsa.arima_mod import ARIMA
from sklearn.metrics import mean_squared_error
 


#def parser(x):
	#return datetime.strptime('190'+x, '%Y-%m')
t_s = pd.read_csv('daily-min-temp.csv', header=0,index_col=0)
print(t_s.head())
t_s.plot()
pyplot.show()
autocorrelation_plot(t_s)
pyplot.show()
# fit model
mod = ARIMA(t_s, order=(6,2,0))
mod_fit = mod.fit(disp=0)
print(mod_fit.summary())
# plot residual errors
resd = DataFrame(mod_fit.resid)
resd.plot()
pyplot.show()
resd.plot(kind='kde')
pyplot.show()
print(resd.describe())

X = t_s.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictn = list()
for t in range(len(test)):
	mod = ARIMA(history, order=(5,1,0))
	mod_fit = mod.fit(disp=0)
	output = mod_fit.forecast()
	yhat = output[0]
	predictn.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictn)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictn, color='green')
pyplot.show()


# In[12]:


import pandas as pd
from numpy import mean
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
t_s=pd.read_csv('daily-min-temp.csv',header=0,index_col=0)

#prepare situation

t=t_s.values
window =6
history = [t[i] for i in range(window)]
test_series = [t[i] for i in range(window, len(t))]
predictn=list()

# walk forward over time steps in test
for r in range(len(test_series)):
	length = len(history)
	yhat = mean([history[i] for i in range(length-window,length)])
	ob = test_series[r]
	predictn.append(yhat)
	history.append(ob)
	print('predicted=%f, expected=%f' % (yhat, ob))
error = mean_squared_error(test_series, predictn)
# plot
pyplot.plot(test_series)
pyplot.plot(predictn, color='green')
pyplot.show()
# zoom plot
pyplot.plot(test_series[0:50])
pyplot.plot(predictn[0:50], color='green')
pyplot.show()

