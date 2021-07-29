#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##Accident Analysis


# In[ ]:


'''A model trained to predict number of accidents that might happen in a region. 
This model was trained using seasonal ARIMA, Time series analysis and checking accuracy by MSE.'''


# In[68]:


import pandas as pd
import matplotlib.pyplot as plt
import re
pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings("ignore")


# # I Import Data and Data Exploration

# In[69]:


df=pd.read_csv("database.csv")


# In[70]:


df.shape


# In[71]:


df.head()


# # II Preprocessing and dealing with timestamp

# In[72]:


df.info()


# In[73]:


df=df.fillna(0)


# In[74]:


import re
from datetime import datetime


# In[75]:


#Convert time to standard time object
def time_clean(time):
    time=re.split('[/:\s]',time) #split with delimiters comma, semicolon and space 
    for i in [0,1,3,4]:
        if len(time[i])<2:
            time[i]='0'+time[i]
    time=(' ').join(time)
    
    datetime_object = datetime.strptime(time, '%m %d %Y %I %M %p')
    return datetime_object


# In[76]:


#Extracting datetime values to make groupby easier
df['datetime']=df['Accident Date/Time'].apply(time_clean)
df['month']=[i.month for i in df['datetime']]
df['hour']=[i.hour for i in df['datetime']]


# # II Answer Interesting questions
# 
# ### 1) How are incidents distributed by year, month, and hour of the day?

# In[77]:


numacc_year=df.groupby('Accident Year').agg('count')[['Report Number']]
numacc_month=df.groupby('month').agg('count')[['Report Number']]
numacc_hour=df.groupby('hour').agg('count')[['Report Number']]


# In[78]:


plt.figure(figsize=(14,3))
plt.bar(numacc_year.index, numacc_year['Report Number'])
plt.title('Cases per Year ')
plt.ylabel('Cases')
plt.show()


# In[79]:


numacc_year.head(6)


# In[80]:


plt.figure(figsize=(14,3))
plt.plot(numacc_month.index, numacc_month['Report Number'])
plt.title('Cases per Month ')
plt.ylabel('Cases')
plt.show()


# In[81]:


numacc_month.head(12)


# In[82]:


plt.figure(figsize=(14,3))
plt.plot(numacc_hour.index, numacc_hour['Report Number'])
plt.title('Cases per Hour ')
plt.ylabel('Cases')
plt.show()


# ### 2) Where are these accidents located?

# In[83]:


import plotly.express as px


# In[84]:


df_scatter=df[['Accident Date/Time','Accident City','Pipeline/Facility Name','Pipeline Type',
    'Accident Latitude', 'Accident Longitude','Accident State' ]]
df_scatter.head()


# In[85]:


fig = px.scatter_mapbox(df_scatter, lat="Accident Latitude", lon="Accident Longitude", hover_name="Pipeline/Facility Name", 
                        hover_data=["Accident Date/Time", "Accident City",'Pipeline Type'],
                        color_discrete_sequence=["fuchsia"], zoom=3, height=500)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# ### 3) What is the net loss of each state?

# In[86]:


df_bystate=df
df_bystate=df_scatter.groupby('Accident State').agg('count')['Pipeline Type']
df_bystate=df_bystate.reset_index()
df_bystate=df_bystate.rename(columns={"Pipeline Type": "Number of Accidents"})


# In[87]:


columns=['Net Loss (Barrels)','All Costs']
df_bystate=df.groupby('Accident State').agg(['sum','count'])
df_bystate=df_bystate[columns]
df_bystate['Number of Accident']=df_bystate['Net Loss (Barrels)']['count']
df_bystate['Num Barrels Lost']=df_bystate['Net Loss (Barrels)']['sum'].apply(lambda x: round(x,2))
df_bystate['Loss (million USD)']=(df_bystate['All Costs']['sum']/(1000000)).apply(lambda x: round(x,2))
df_bystate=df_bystate.drop(columns=columns,axis=1)
df_bystate=df_bystate.reset_index()


# In[88]:


df_bystate.head()


# In[89]:


k=df_bystate.sort_values(by='Loss (million USD)',ascending=False)
k.head()


# In[90]:


for col in df_bystate.columns:
    df_bystate[col] = df_bystate[col].astype(str)
    
df_bystate['text']=df_bystate['Accident State'] + '<br>' +                     'Number of Accident: ' + df_bystate['Number of Accident']+ '<br>' +                     'Num Barrels Lost: ' + df_bystate['Num Barrels Lost']


# In[91]:


import plotly.graph_objects as go


fig = go.Figure(data=go.Choropleth(
    locations=df_bystate['Accident State'], # Spatial coordinates
    z = df_bystate['Loss (million USD)'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    text=df_bystate['text'],
    colorbar_title = "Net Loss (million of USD)"
    
))

fig.update_layout(
    title_text = 'Overall Loss (million of USD) by State from 2010-2017',
    geo_scope='usa' # limite map scope to USA
)

fig.show()


# ### 4) So … What really happened in Michigan?

# In[92]:


df_temp=df[df['Accident State']=='MI']
df_temp=df_temp.sort_values('datetime')
plt.figure(figsize=(14,5))
plt.plot(df_temp['datetime'], df_temp['All Costs']/1000000)
plt.ylabel('Net loss (million of USD)')
plt.show()


# In[93]:


df_temp.shape


# In[94]:


df_temp['All Costs'].sum()


# In[95]:


df_temp=df_temp.sort_values('All Costs', ascending=False)
df_temp[['Accident City','datetime','Cause Category','Unintentional Release (Barrels)',
         'Intentional Release (Barrels)', 'Liquid Recovery (Barrels)',
         'Net Loss (Barrels)','All Costs' ]].reset_index().head(10)


# In[96]:


#Analyze the major leakage
a=df_temp.iloc[0]['All Costs']/df_temp['All Costs'].sum()
b=df_temp.iloc[0]['All Costs']/df['All Costs'].sum()
print(a,b)


# ### 5) What is the culprit of oil leakages?

# In[97]:


df_temp=df.groupby('Cause Category').agg(['count','sum'])[['All Costs','All Fatalities']]
df_temp['Number of cases']=df_temp['All Costs']['count']
df_temp['Loss (million USD)']=(df_temp['All Costs']['sum']/(1000000)).apply(lambda x: round(x,2))
df_temp['Fatalities']=df_temp['All Fatalities']['sum']
df_temp=df_temp.drop(['All Costs','All Fatalities'],axis=1).reset_index()
df_temp


# In[98]:


#Analyze their percentage
df_temp_2=df_temp.copy()
df_temp_2['Number of cases']=df_temp['Number of cases']/df_temp['Number of cases'].sum()
df_temp_2['Loss (million USD)']=df_temp['Loss (million USD)']/df_temp['Loss (million USD)'].sum()
df_temp_2['Fatalities']=df_temp['Fatalities']/df_temp['Fatalities'].sum()
df_temp_2


# # III References
# 
# https://plot.ly/python/choropleth-maps/#united-states-choropleth-map
