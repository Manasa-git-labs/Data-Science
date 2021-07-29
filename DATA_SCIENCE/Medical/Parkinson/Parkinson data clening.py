#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import files
uploaded = files.upload()


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import KFold


# In[ ]:


import io
parkinson_df= pd.read_csv(io.BytesIO(uploaded['parkinsons2.csv']))


# In[ ]:


parkinson_df.head().transpose()


# In[ ]:


X = parkinson_df
X=X.replace([np.inf, -np.inf], np.nan).fillna(value=0)
scaler = MinMaxScaler()
scaler.fit(X)
X=scaler.transform(X)


# In[ ]:


parkinson_df.columns


# In[ ]:


parkinson_df.shape


# In[ ]:


parkinson_df.info()


# In[ ]:


parkinson_df.describe().transpose()


# In[ ]:


parkinson_df[parkinson_df.isnull().any(axis=1)]


# In[ ]:


parkinson_df.boxplot(figsize=(24,8))


# In[ ]:


parkinson_df.corr()


# In[ ]:





# In[ ]:


parkinson_df['status'].value_counts().sort_index()


# In[ ]:


X = parkinson_df.drop(['MDVP:Fhi(Hz)','NHR','status'],axis=1)
Y = parkinson_df['status']


# In[ ]:


#Splitting the data into train and test in 70/30 ratio with random state as 2.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2)


# In[ ]:


LR = LogisticRegression()
LR.fit(X_train, Y_train)


# In[ ]:


Y1_predict = LR.predict(X_test)
Y1_predict


# In[ ]:


Y_acc = metrics.accuracy_score(Y_test,Y1_predict)
print("Accuracy of the model is {0:2f}".format(Y_acc*100))
Y_cm=metrics.confusion_matrix(Y_test,Y1_predict)
print(Y_cm)


# In[ ]:


#Sensitivity
TPR=Y_cm[1,1]/(Y_cm[1,0]+Y_cm[1,1])
print("Sensitivity of the model is {0:2f}".format(TPR))


# In[ ]:


#Specificity
TNR=Y_cm[0,0]/(Y_cm[0,0]+Y_cm[0,1])
print("Specificity of the model is {0:2f}".format(TNR))


# In[ ]:


Y_CR=metrics.classification_report(Y_test,Y1_predict)
print(Y_CR)


# In[ ]:


fpr,tpr, _ = roc_curve(Y_test, Y1_predict)
roc_auc = auc(fpr, tpr)

print("Area under the curve for the given model is {0:2f}".format(roc_auc))
plt.figure()
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()


# In[ ]:


X = parkinson_df.drop(['MDVP:Fhi(Hz)','NHR','status'],axis=1)
Y = parkinson_df['status']


# In[ ]:


# K-fold cross validation for the given model:
#Since the dataset contains 197 rows, we are taking the number of splits as 3
kf=KFold(n_splits=3,shuffle=True,random_state=2)
acc=[]
for train,test in kf.split(X,Y):
    M=LogisticRegression()
    Xtrain,Xtest=X.iloc[train,:],X.iloc[test,:]
    Ytrain,Ytest=Y[train],Y[test]
    M.fit(Xtrain,Ytrain)
    Y_predict=M.predict(Xtest)
    acc.append(metrics.accuracy_score(Ytest,Y_predict))
    print(metrics.confusion_matrix(Ytest,Y_predict))
    print(metrics.classification_report(Ytest,Y_predict))
print("Cross-validated Score:{0:2f} ".format(np.mean(acc)))


# Accuracy for each fold

# In[ ]:


acc


# 

# #Error

# In[ ]:


error=1-np.array(acc)
error


# Variance Error of the model

# In[ ]:


np.var(error,ddof=1)

