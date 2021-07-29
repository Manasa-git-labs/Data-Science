#!/usr/bin/env python
# coding: utf-8

# In[2]:


from google.colab import files
uploaded = files.upload()


# In[ ]:


import numpy as np # linear algebra
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import statsmodels.api as sm
from sklearn import datasets, linear_model
from sklearn.linear_model import lasso_path
import matplotlib.pyplot as plt

np.set_printoptions(suppress = True, precision=3); #Options for NumPy



# In[4]:


import io
df3 = pd.read_csv(io.BytesIO(uploaded['Data.csv']))


# In[ ]:


df3.describe().T


# In[ ]:


def dddraw(X_reduced,name):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # To getter a better understanding of interaction of the dimensions
    # plot the first three PCA dimensions
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y,cmap=plt.cm.Paired)
    titel="First three directions of "+name 
    ax.set_title(titel)
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    plt.show()


# In[ ]:


from sklearn.linear_model import OrthogonalMatchingPursuit,RANSACRegressor,LogisticRegression,ElasticNetCV,HuberRegressor, Ridge, Lasso,LassoCV,Lars,BayesianRidge,SGDClassifier,LogisticRegressionCV,RidgeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}

    
# import some data to play with
X = df3
Y=df3['RPDE']
def rmsle(y_predicted, y_real):
    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))
def procenterror(y_predicted, y_real):
     return np.round( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ,1)

X = df3
Y=np.round(df3['RPDE']*100) //preprocessing
X=X.replace([np.inf, -np.inf], np.nan).fillna(value=0)
#print(X) #nasty NaN
scaler = MinMaxScaler()
scaler.fit(X)
X=scaler.transform(X)
poly = PolynomialFeatures(2)
X=poly.fit_transform(X)


names = [
         #'ElasticNet',
         'SVC',
         'kSVC',
         'KNN',
         'DecisionTree',
         'RandomForestClassifier',
         #'GridSearchCV',
         'HuberRegressor',
         'Ridge',
         'Lasso',
         'LassoCV',
         'Lars',
         #'BayesianRidge',
         'SGDClassifier',
         'RidgeClassifier',
         'LogisticRegression',
         'OrthogonalMatchingPursuit',
         #'RANSACRegressor',
         ]

classifiers = [
    #ElasticNetCV(cv=10, random_state=0),
    SVC(),
    SVC(kernel = 'rbf', random_state = 0),
    KNeighborsClassifier(n_neighbors = 1),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators = 200),
    #GridSearchCV(SVC(),param_grid, refit = True, verbose = 1),
    HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100,epsilon=2.95),
    Ridge(fit_intercept=True, alpha=0.0, random_state=0, normalize=True),
    Lasso(alpha=0.05),
    LassoCV(),
    Lars(n_nonzero_coefs=10),
    #BayesianRidge(),
    SGDClassifier(),
    RidgeClassifier(),
    LogisticRegression(),
    OrthogonalMatchingPursuit(),
    #RANSACRegressor(),
]
correction= [0,0,0,0,0,0,0,0,0,0,0,0]

temp=zip(names,classifiers,correction)
print(temp)

for name, clf,correct in temp:
    regr=clf.fit(X,Y)
    #print( name,'% errors', abs(regr.predict(X)+correct-Y).sum()/(Y.sum())*100)
    print(name,'%error',procenterror(regr.predict(X),Y),'rmsle',rmsle(regr.predict(X),Y))
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score, precision_score, recall_score

    # Confusion Matrix
    print(name,'Confusion Matrix')
    print(confusion_matrix(Y, np.round(regr.predict(X) ) ) )
    print('--'*40)

    # Classification Report
    print('Classification Report')
    print(classification_report(Y,np.round( regr.predict(X) ) ))

    # Accuracy
    print('--'*40)
    logreg_accuracy = round(accuracy_score(Y, np.round( regr.predict(X) ) ) * 100,2)
    print('Accuracy', logreg_accuracy,'%')


# In[ ]:





# In[ ]:


from sklearn.decomposition import PCA, FastICA,SparsePCA,NMF, LatentDirichletAllocation,FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection,SparseRandomProjection
from sklearn.cluster import KMeans,Birch
import statsmodels.formula.api as sm
from scipy import linalg
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
import matplotlib.pyplot as plt

def rmsle(y_predicted, y_real):
    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))
def procenterror(y_predicted, y_real):
     return ( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ).round()

n_col=3
X = df3
Y=df3['RPDE']
def rmsle(y_predicted, y_real):
    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))
def procenterror(y_predicted, y_real):
     return np.round( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ,1)

names = [
         'PCA',
         'FastICA',
         'Gauss',
         'KMeans',
          #'SparsePCA',
         'SparseRP',
         'Birch',
         #'NMF',    
         #'LatentDietrich',    
        ]

classifiers = [
    PCA(n_components=n_col),
    FastICA(n_components=n_col),
    GaussianRandomProjection(n_components=3),
    KMeans(n_clusters=3),
  
    SparseRandomProjection(n_components=n_col, dense_output=True),
    Birch(branching_factor=10, n_clusters=2,threshold=0.5),
     
  
    ]
correction= [1,1,0,0,0,0]

temp=zip(names,classifiers,correction)
print(temp)

for name, clf,correct in temp:
    Xr=clf.fit_transform(X,Y)
    dddraw(Xr,name)


# Analyzing Variables to Find the Dependent Variable
# 
# As it can be seen, the table doesn't tell you what the dependent variable of this data set is. So we must discover it. To do so, we find the correlation matrix and see which column has the highest average correlation constant. This is obviously for demonstration purposes and it cannot be used as a way of determining a "dependent variable". #
# 
# Qualitative Look at Correlation Matrix

# In[ ]:


corrMat = df3[::].corr(); 
sns.heatmap(corrMat, vmax=0.8, square = True);


# Quantitative Look at Correlation Matrix

# In[ ]:


print(corrMat);


# After some simple calculations, one can see that Jitter(Percent) seems to generally have the highest percent of correlation with all of the other variables. Thus, let us use that as the dependent variable and the rest as the independent variables

# In[ ]:


X = df3.values[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]] # First 19 columns of data
y = df3.values[:, 19]; # Last col of data
X = np.array(X); 
X = sm.add_constant(X); 
results = sm.OLS(endog=y, exog = X).fit(); 
print(results.summary())


# As it can be, only some of the above variables are truly contributing to the regression. In particular, the following do not seem to contribute to the model as much as the other variables because they are less than 2 standard deviations from the mean: x1, x2, x6, x8, x11, x14 and x18. In other words, these variables are accepted by the null hypothesis. Thus, let us see how things change if we ignore these variables

# In[ ]:


X = df3.values[:,[2,3,4,6,8,9,11,12,14,15,16,18]] # First 19 columns of data
y = df3.values[:, 19]; # Last col of data
X = np.array(X); 
X = sm.add_constant(X); 
results1 = sm.OLS(endog=y, exog = X).fit(); 

print(results1.summary())


# In[ ]:


print("\n\n\nβ =", results1.params) #new beta vector for linear regression


# Some of the t-values have changed and the R-squared variable has decreased, but the new model seems to predict the Jitter percentage as well as the old model. Thus, ignoring those certain variables with the low t-scores did not do much damage.
# 
# Let us See how Ridge Regression Works on this Data se

# In[ ]:



X = df3.values[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]] # First 19 columns of data
y = df3.values[:, 19]; # Last col of data

n_alphas = 200

alphas = np.logspace(-10, -2, n_alphas)
regr = linear_model.Ridge(fit_intercept = False)

coefs = []; 
for a in alphas:
    regr.set_params(alpha = a)
    regr.fit(X,y)
    coefs.append(regr.coef_)
    
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()


# In[ ]:


from itertools import cycle

X = df3.values[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]] # First 19 columns of data
y = df3.values[:, 19]; # Last col of data

eps = 5e-20
alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps, fit_intercept = False)

plt.figure(1)
ax = plt.gca()
colors = cycle(['b', 'r', 'g', 'c', 'k']);
neg_log_alphas_lasso = -np.log10(alphas_lasso)
for coef_l, c in zip(coefs_lasso, colors):
    l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c);
   
plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Lasso Paths')
plt.axis('tight')




#lassocoefs = []; 
#for a in alphas:
#    regr.set_params(alpha = a)
#    regr.fit(X,y)
#    lassocoefs.append(regr.coef_)

#plt.xlabel('-Log(alpha)');
#plt.ylabel('coefficients');
#plt.title('Lasso Paths');
#plt.axis('tight');


# In[ ]:


print("Ridge   ", "Lasso    ", "OLS")

A  = [coefs[0], coefs_lasso.T[::-1][0], results.params[1:]]

print(np.matrix(A).T)

