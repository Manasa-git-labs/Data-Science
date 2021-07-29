#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

df = pd.read_csv('diabetes.csv')

print(df)


# In[21]:


X = df.drop('Outcome', axis = 1)
y = df['Outcome']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape)
print(X_test.shape)


# In[22]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
mannequin = lr.match(X_train,y_train)
y_pred = lr.predict(X_test)


# In[ ]:


print(y_pred)


# In[ ]:


print(accuracy_score(y_pred,y_test))


# In[ ]:


import pickle    

Model = pickle.dumps(mannequin)  


# In[ ]:


import tkinter as tk

from tkinter import ttk

win = tk.Tk()

win.title('Diabetes Predictions')


# In[ ]:


#Column 1 
Preg=ttk.Label(win,textual content="Preg")
Preg.grid(row=0,column=0,sticky=tk.W)
Preg_var=tk.StringVar()
Preg_entrybox=ttk.Entry(win,width=16,textvariable=Preg_var)
Preg_entrybox.grid(row=0,column=1)
#Column 2
Plas=ttk.Label(win,textual content="Plas")
Plas.grid(row=1,column=0,sticky=tk.W)
Plas_var=tk.StringVar()
Plas_entrybox=ttk.Entry(win,width=16,textvariable=Plas_var)
Plas_entrybox.grid(row=1,column=1)
#Column 3
Pres=ttk.Label(win,textual content="Pres")
Pres.grid(row=2,column=0,sticky=tk.W)
Pres_var=tk.StringVar()
Pres_entrybox=ttk.Entry(win,width=16,textvariable=Pres_var)
Pres_entrybox.grid(row=2,column=1)
#Column 4
pores and skin=ttk.Label(win,textual content="skin")
pores and skin.grid(row=3,column=0,sticky=tk.W)
skin_var=tk.StringVar()
skin_entrybox=ttk.Entry(win,width=16,textvariable=skin_var)
skin_entrybox.grid(row=3,column=1)
#Column 5
take a look at=ttk.Label(win,textual content="test")
take a look at.grid(row=4,column=0,sticky=tk.W)
test_var=tk.StringVar()
test_entrybox=ttk.Entry(win,width=16,textvariable=test_var)
test_entrybox.grid(row=4,column=1)
#Column 6
mass=ttk.Label(win,textual content="mass")
mass.grid(row=5,column=0,sticky=tk.W)
mass_var=tk.StringVar()
mass_entrybox=ttk.Entry(win,width=16,textvariable=mass_var)
mass_entrybox.grid(row=5,column=1)
#Column 7
pedi=ttk.Label(win,textual content="pedi")
pedi.grid(row=6,column=0,sticky=tk.W)
pedi_var=tk.StringVar()
pedi_entrybox=ttk.Entry(win,width=16,textvariable=pedi_var)
pedi_entrybox.grid(row=6,column=1)
#Column 8
age=ttk.Label(win,textual content="age")
age.grid(row=7,column=0,sticky=tk.W)
age_var=tk.StringVar()
age_entrybox=ttk.Entry(win,width=16,textvariable=age_var)
age_entrybox.grid(row=7,column=1)


# In[ ]:


import pandas as pd
DF = pd.DataBody()
def motion():
    international DB
    import pandas as pd
    DF = pd.DataBody(columns=['Preg','Plas','Pres','skin','test','mass','pedi','age'])
    PREG=Preg_var.get()
    DF.loc[0,'Preg']=PREG
    PLAS=Plas_var.get()
    DF.loc[0,'Plas']=PLAS
    PRES=Pres_var.get()
    DF.loc[0,'Pres']=PRES
    SKIN=skin_var.get()
    DF.loc[0,'skin']=SKIN
    TEST=test_var.get()
    DF.loc[0,'test']=TEST
    MASS=mass_var.get()
    DF.loc[0,'mass']=MASS
    PEDI=pedi_var.get()
    DF.loc[0,'pedi']=PEDI
    AGE=age_var.get()
    DF.loc[0,'age']=AGE
print(DF.form)
DB=DF


# In[ ]:


def Output():
    DB["Preg"] = pd.to_numeric(DB["Preg"])
    DB["Plas"] = pd.to_numeric(DB["Plas"])
    DB["Pres"] = pd.to_numeric(DB["Pres"])
    DB["skin"] = pd.to_numeric(DB["skin"])
    DB["test"] = pd.to_numeric(DB["test"])
    DB["mass"] = pd.to_numeric(DB["mass"])
    DB["pedi"] = pd.to_numeric(DB["pedi"])
    DB["age"] = pd.to_numeric(DB["age"])


# In[ ]:


output=mannequin.predict(DB)
    if output==1:
        consequence="Diabetic"
    elif output==0:
        consequence="Non-Diabetic"


# In[ ]:


Predict_entrybox=ttk.Entry(win,width=16)
    Predict_entrybox.grid(row=20,column=1)
    Predict_entrybox.insert(1,str(consequence))
Predict_button=ttk.Button(win,textual content="Predict",command=Output)
Predict_button.grid(row=20,column=0)
win.mainloop()

