#!/usr/bin/env python
# coding: utf-8

# In[ ]:



0


# **MLP IMPLIMENTATION**

# In[ ]:


dataset


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
seednum=7
np.random.seed(seednum)
dataset = pd.read_csv("CreditCardDefaults10k.csv", delimiter = "," , skiprows=2)
X=dataset.iloc[:,0:23]
Y=dataset.iloc[:,23]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
model=Sequential()
model.add(Dense(18,input_dim=23,init='uniform',activation='relu'))
model.add(Dense(8,init='uniform',activation='relu'))
model.add(Dense(1,init='uniform',activation='sigmoid'))
opt = SGD(lr=1, momentum=0.9, decay=0.01)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history=model.fit(X_train,Y_train,nb_epoch=100,batch_size=10,verbose=1,validation_data=(X_test, Y_test))
#score = model.evaluate(X_test, Y_test, accuracy=True, verbose=0)


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel("accuracy")
plt.ylabel("epoch")
plt.title("Credit_card dataset")
plt.legend(['train','test'],loc='upper right')
plt.show()
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel("loss")
plt.ylabel("epoch")
plt.title("Credit_card dataset")
plt.legend(['train','test'],loc='upper right')
plt.show()


# In[ ]:





# **MLP & Regularisation**

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras import regularizers
seednum=7
np.random.seed(seednum)
dataset = pd.read_csv("CreditCardDefaults10k.csv", delimiter = "," , skiprows=2)
X=dataset.iloc[:,0:23]
Y=dataset.iloc[:,23]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
model = Sequential()
model.add(Dense(23, input_dim=23, activation='relu',activity_regularizer=regularizers.l2(0.001)))
model.add(Dense(18, activation='relu',activity_regularizer=regularizers.l2(0.05)))
model.add(Dense(1, activation='sigmoid',activity_regularizer=regularizers.l2(0.5)))
# compile the keras model
opt = SGD(lr=1, momentum=0.9, decay=0.01)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
history=model.fit(X_train,Y_train,nb_epoch=100,batch_size=100,verbose=1,validation_data=(X_test, Y_test))


# 

# In[ ]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel("accuracy")
plt.ylabel("epoch")
plt.title("Credit_card dataset")
plt.legend(['train','test'],loc='upper right')
plt.show()
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel("loss")
plt.ylabel("epoch")
plt.title("Credit_card dataset")
plt.legend(['train','test'],loc='upper right')
plt.show()


# 

# 

# **WORK FOR REFERNCE**

# In[ ]:





# In[ ]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.model_selection import train_test_split, cross_val_score 
from statistics import mean 


# In[ ]:


dataset = np.loadtxt("CreditCardDefaults10k.csv", delimiter = "," , skiprows=2)


# In[ ]:




# Bulding and fitting the Linear Regression model 
linearModel = LinearRegression() 
linearModel.fit(X_train, y_train) 

# Evaluating the Linear Regression model 
print(linearModel.score(X_test, y_test)) 


# In[ ]:


from keras import regularizers
model.add(Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import plot_model

seednum=7
np.random.seed(seednum)
dataset = np.loadtxt("CreditCardDefaults10k.csv", delimiter = "," , skiprows=2)
#print(dataset)

inputFeatures = dataset[:,0:23]
Output = dataset[:,23]

print(inputFeatures)
print(Output)
architecture=[]
accuracyy=[]
for i in range(21,25) :
   for j in range(18,20):
     result = ""
     model=Sequential()
     model.add(Dense(i,input_dim=23,init='uniform',activation='relu'))
     model.add(Dense(1,init='uniform',activation='sigmoid'))
     model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
     model.fit(inputFeatures,Output,nb_epoch=100,batch_size=100)
     scores=model.evaluate(inputFeatures,Output)
     z=scores[1]*100
     print("%s: %.2f%%" % (model.metrics_names[1],z))
     accuracyy.append(z)
     result += " 18:" +str(i) + " : "+ str(j)+":1"
     architecture.append(result)
     j=j+1
i=i+1


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(architecture,accuracyy)
plt.xlabel("ARCHITECTURE")
plt.ylabel("ACCURACY")
plt.title("Credit_card dataset")
plt.show()


# In[ ]:


print(accuracyy)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import plot_model

seednum=7
np.random.seed(seednum)
dataset = np.loadtxt("CreditCardDefaults10k.csv", delimiter = "," , skiprows=2)
#print(dataset)

inputFeatures = dataset[:,0:23]
Output = dataset[:,23]

print(inputFeatures)
print(Output)
architecture=[]
accuracyy=[]
for i in range(21,25) :
   for j in range(18,20):
     result = ""
     model=Sequential()
     model.add(Dense(i,input_dim=23,init='uniform',activation='softmax'))
     model.add(Dense(j,init='uniform',activation='tanh'))
     model.add(Dense(1,init='uniform',activation='sigmoid'))
     model.compile(loss='binary_crossentropy',optimizer='rmsprop ',metrics=['accuracy'])
     model.fit(inputFeatures,Output,nb_epoch=100,batch_size=100)
     scores=model.evaluate(inputFeatures,Output)
     z=scores[1]*100
     print("%s: %.2f%%" % (model.metrics_names[1],z))
     accuracyy.append(z)
     result += " 18:" +str(i) + " : "+ str(j)+":1"
     architecture.append(result)
     j=j+1
i=i+1


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(architecture,accuracyy)
plt.xlabel("ARCHITECTURE")
plt.ylabel("ACCURACY")
plt.title("Credit_card dataset")
plt.show()


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import plot_model

seednum=7
np.random.seed(seednum)
dataset = np.loadtxt("CreditCardDefaults10k.csv", delimiter = "," , skiprows=2)
#print(dataset)

inputFeatures = dataset[:,0:23]
Output = dataset[:,23]

print(inputFeatures)
print(Output)
architecture=[]
accuracyy=[]
for i in range(21,25) :
   for j in range(18,20):
     result = ""
     model=Sequential()
     model.add(Dense(i,input_dim=23,init='uniform',activation='relu'))
     model.add(Dense(j,init='uniform',activation='relu'))
     model.add(Dense(1,init='uniform',activation='sigmoid'))
     model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
     model.fit(inputFeatures,Output,nb_epoch=10,batch_size=10)
     scores=model.evaluate(inputFeatures,Output)
     z=scores[1]*100
     print("%s: %.2f%%" % (model.metrics_names[1],z))
     accuracyy.append(z)
     result += " 18:" +str(i) + " : "+ str(j)+":1"
     architecture.append(result)
     j=j+1
i=i+1


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(architecture,accuracyy)
plt.xlabel("ARCHITECTURE")
plt.ylabel("ACCURACY")
plt.title("Credit_card dataset")
plt.show()


# In[ ]:


print(accuracyy)
print(architecture)


# In[ ]:


print(max(accuracyy))

