#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 
import os
from glob import glob
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import keras
#from skimage.util.import montage2d
from skimage.io import imread
from scipy.io import loadmat # for loading mat files
from tqdm import tqdm_notebook as tqdm
#from tqdm import tqdm_notebook
in_path="/content/drive/My Drive/Colab Notebooks/Normalized"


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import os
os.chdir("/content/drive/My Drive/Colab Notebooks/Normalized")


# In[ ]:


def parse_mat(in_path):
    in_dat = loadmat(in_path, squeeze_me = True, struct_as_record = True)
    vec1_load, img_load,vec2_load = in_dat['data'].tolist()[1].tolist()
    return vec1_load, img_load, vec2_load
def mat_to_df(in_path):
    vec1_load, img_load, vec2_load = parse_mat(in_path)
    c_df = pd.DataFrame(dict(img=[x for x in img_load], 
                             vec1=[x for x in vec1_load],
                            vec2=[x for x in vec2_load]))
    c_df['group'] = os.path.basename(os.path.dirname(in_path))
    c_df['day'] = os.path.splitext(os.path.basename(in_path))[0]
    return c_df
def safe_mat_to_df(in_path):
    try:
        return mat_to_df(in_path)
    except ValueError as e:
        print('ValueError', e, in_path)
        return None 
mat_files = glob("/content/drive/My Drive/Colab Notebooks/Normalized/*/*.mat")
print(mat_files)
print(len(mat_files), 'normalized files found')


# In[ ]:


all_norm_df = pd.concat([safe_mat_to_df(in_path) for in_path in tqdm(mat_files)], ignore_index=True)
all_norm_df.sample(2)


# In[ ]:


all_norm_df.head(2)


# In[ ]:



#all_norm_df.tail(20)
print(all_norm_df.shape[0], 'images loaded')
#group_view = all_norm_df.groupby('group').apply(lambda x: x.sample(20)).reset_index(drop = True)
#print(group_view)
fig, m_axs = plt.subplots(5,5, figsize = (20, 20))
#for (_, c_row), c_ax in zip(group_view.iterrows(), m_axs.flatten()):
for (_, c_row), c_ax in zip(all_norm_df.iterrows(), m_axs.flatten()):
    c_ax.imshow((c_row['img']),cmap = 'gray')
   # cv2.imwrite('images\\'+str(t)+'.jpg',c_row)
    #c_ax.
    #c_ax.legend()
    #c_ax.set_title('{group}'.format(**c_row))
    #c_ax.imwrite(c_row[img],'images_train.jpg')
    


# In[ ]:


all_norm_df.shape


# In[ ]:


for v in ['vec1', 'vec2']:
    for i, x_dim in enumerate('xyz'):
        all_norm_df['{}_{}'.format(v, x_dim)] = all_norm_df[v].map(lambda x: x[i])
all_norm_df.sample(3)


# In[ ]:


fig, m_axs = plt.subplots(6, 4, figsize = (20, 20))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
from itertools import product
for (ax_dist, ax_min, ax_mean, ax_max), (v, (i, x)) in zip(m_axs, product(['vec1', 'vec2'], enumerate('xyz'))):
    # use random sampling to get a better feeling
    c_vec = all_norm_df.sample(10000)['{}_{}'.format(v, x)]
    ax_dist.hist(c_vec.values, 30)
    ax_dist.axis('on')
    j = c_vec.idxmin()
    ax_min.imshow(all_norm_df.iloc[j]['img'], cmap = 'bone')
    ax_min.set_title('min {}_{}: {:2.2f}'.format(v, x, all_norm_df.iloc[j]['{}_{}'.format(v, x)]))
    
    k = c_vec.idxmax()
    ax_max.imshow(all_norm_df.iloc[k]['img'], cmap = 'bone')
    ax_max.set_title('max {}_{}: {:2.2f}'.format(v, x, all_norm_df.iloc[k]['{}_{}'.format(v, x)]))
    
    p = np.abs(c_vec-np.mean(c_vec)).idxmin()
    ax_mean.imshow(all_norm_df.iloc[p]['img'], cmap = 'bone')
    ax_mean.set_title('mean: {}_{}: {:2.2f}'.format(v, x, all_norm_df.iloc[p]['{}_{}'.format(v, x)]))


# In[ ]:


#find normalised data in all_norm_df
df1= all_norm_df.copy()


# In[ ]:


df1.values


# In[ ]:


df1.describe


# In[ ]:


df1['img'][0].shape


# In[ ]:



df1['theta'] = -(df1['vec2_y'])


train= df1[:1000]


# In[ ]:


sample_train.head(1)


# In[ ]:


#import math as m
#m.asin(-0.0370)


# #### convulational neural network

# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Concatenate
from keras import layers


# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[ ]:


initial_model = keras.Sequential(
    [
        keras.Input(shape=(36, 60, 1)),
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        layers.MaxPooling2D(3),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(3),
        layers.Flatten(),
        layers.Dense(50),
        
        
    ]
)

feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=initial_model.output
)
initial_model.compile(loss='mse',optimizer='adam',metrics=['mse','mae','accuracy'])


# In[ ]:


feature_vector=pd.DataFrame()
for i in range(0,1000):
  feature_vector['{}'.format(i)]=i
j=0
for v in train['img']:
  x=np.array( train['img'][j])
  x = x.reshape((1, 36, 60,1))
  features = feature_extractor(x)
  proto_tensor=tf.make_tensor_proto(features)
  final=tf.make_ndarray(proto_tensor)
  final=final.reshape(50,)
  feature_vector['{}'.format(j)]=final
  j=j+1


# df7=pd.DataFrame()
# for i in range(0,1000):
#   df7['{}'.format(i)]=i
# j=0
# for v in train['img']:
#   x=np.array( train['img'][j])
#   x = x.reshape((1, 36, 60,1))
#   features = feature_extractor(x)
#   proto_tensor=tf.make_tensor_proto(features)
#   final=tf.make_ndarray(proto_tensor)
#   final=final.reshape(50,)
#   df7['{}'.format(j)]=final
#   j=j+1

# In[ ]:


feature_vector


# In[ ]:


feature_vector=feature_vector.transpose()


# In[ ]:


feature_vector


# In[ ]:


train.head(5)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




