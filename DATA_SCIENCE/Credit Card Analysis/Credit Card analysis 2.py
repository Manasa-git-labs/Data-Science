#!/usr/bin/env python
# coding: utf-8

# In[29]:


from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import plot_model
from keras import regularizers

seednum=7
np.random.seed(seednum)
dataset = np.loadtxt("CreditCardDefaults10k.csv", delimiter = "," , skiprows=2)


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
     model.add(Dense(i,input_dim=23,init='uniform',activation='relu',activity_regularizer=regularizers.l1(0.03)))
     model.add(Dense(j,init='uniform',activation='relu',activity_regularizer=regularizers.l1(0.03)))
     model.add(Dense(1,init='uniform',activation='sigmoid',activity_regularizer=regularizers.l1(0.03)))
     model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
     model.fit(inputFeatures,Output,validation_split =0.33,nb_epoch=40,batch_size=10) 
     scores=model.evaluate(inputFeatures,Output)
     z=scores[1]*100
     print("%s: %.2f%%" % (model.metrics_names[1],z))
     accuracyy.append(z)
     result += " 18:" +str(i) + " : "+ str(j)+":1"
     architecture.append(result)
     j=j+1
i=i+1



# In[31]:


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[33]:


import matplotlib.pyplot as plt
plt.plot(architecture,accuracyy)
plt.xlabel("ARCHITECTURE")
plt.ylabel("ACCURACY")
plt.title("Credit_card dataset")
plt.show()


# In[ ]:


print(accuracyy)
print(architecture)


# In[17]:


print(max(accuracyy))


# In[5]:


from keras.utils import plot_model
plot_model(model, to_file='model.png')


# In[6]:


import matplotlib.pyplot as plt

history = model.fit(inputFeatures,Output,validation_split =0.33,nb_epoch=10,batch_size=10) 

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# **Implementing regularisarion**

# In[7]:



print(history.history.keys())


# In[9]:


history = model.fit(x_train, y_train, nb_epoch=10, validation_split=0.2, shuffle=True)


# In[8]:


import keras
from matplotlib import pyplot as plt
history = model1.fit(train_x, train_y,validation_split = 0.1, epochs=50, batch_size=4)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[ ]:


from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import plot_model


# In[ ]:


print(max(accuracyy))


# In[ ]:





# In[ ]:





# In[22]:


# MLP with automatic validation set
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = np.loadtxt("CreditCardDefaults10k.csv", delimiter = "," , skiprows=2)
# split into input (X) and output (Y) variables
X = dataset[:,0:23]
Y = dataset[:,23]
# create model
model = Sequential()
model.add(Dense(i,input_dim=23,init='uniform',activation='relu',activity_regularizer=regularizers.l1(0.03)))
model.add(Dense(j,init='uniform',activation='relu',activity_regularizer=regularizers.l1(0.03)))
model.add(Dense(1,init='uniform',activation='sigmoid',activity_regularizer=regularizers.l1(0.03)))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10)


# In[26]:


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

