#!/usr/bin/env python
# coding: utf-8

# # TODO
# - ⬜ add unblacend class
# - ⬜ batch norm
# - ⬜ regularizer
# - ⬜ train/dev/test
# - ⬜ lr tuner
# - ✅ add normalizing 
# 
# 
# 
# 

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf 
import matplotlib.pyplot as plt
import sklearn as sk


# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[3]:



import seaborn as sns
import warnings
warnings.simplefilter('ignore')
sns.set(rc={'figure.figsize' : (10, 5)})
sns.set_style("darkgrid", {'axes.grid' : True})


# Reading csv files and showing first and last 5 records. 

# In[4]:


diabetes = pd.read_csv('diabetes.csv')
diabetes


# In[5]:


diabetes.describe()


# In[6]:


corrMatrix = diabetes.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()


# In[7]:


from sklearn.dummy import DummyClassifier
x = diabetes.drop(columns = 'Outcome')
y = diabetes['Outcome']


# In[8]:


x.head()


# In[9]:


y.head()


# In[10]:


dummy = DummyClassifier('most_frequent') #returining most frequent class in this case 1/
results = dummy.fit(x,y)
results.score(x,y)


# In[24]:


from tensorflow import keras
import tensorflow.keras.backend as K

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


# In[50]:


def max_metric (history):
    max_acc = max(history.history['accuracy'])
    max_f1 = max(history.history['get_f1'])
    min_loss = min(history.history['loss'])
    max_val_acc = max(history.history['val_accuracy'])
    max_val_f1 = max(history.history['val_get_f1'])
    min_val_loss = min(history.history['val_loss'])
    print(f"Maximum Accuracy: {max_acc} \nMaximum F1 Score: {max_f1} \nMinimum Binary CrossEntropy Loss: {min_loss} \nMaximum Validation Accuracy: {max_val_acc} \nMaximum Validation F1 Score: {max_val_f1} \nMaximum Validation Binary CrossEntropy Loss: {min_val_loss} \n")


# In[67]:


def plot_this(history):
    # summarize history for accuracy
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # summarize history for f1
    plt.plot(history.history['val_get_f1'])
    plt.plot(history.history['get_f1'])
    plt.title('model f1')
    plt.ylabel('f1')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # summarize history for loss
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# In[51]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(16, input_dim=x.shape[1], activation = 'relu' ))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()


# In[52]:


model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy', get_f1])


# In[53]:


history = model.fit(x,y, validation_split=0.33, batch_size=128, epochs=200, workers=4, verbose=3)


# In[68]:


max_metric(history)
plot_this(history)


# ## with Normalization

# In[56]:


diabetes.columns


# In[57]:


# normalize the data
# we do not want to modify our label column Exited
cols_to_norm = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
       'DiabetesPedigreeFunction', 'Age']

# copy churn dataframe to churn_norm to do not affect the original data
dia_norm = diabetes.copy()

# normalize churn_norm dataframe 
dia_norm[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min())/ (x.max() - x.min()) )


# In[58]:


dia_norm


# In[59]:


dia_norm.describe()


# In[70]:


model2 = tf.keras.Sequential()
model2.add(tf.keras.layers.Dense(16, input_dim=x.shape[1], activation = 'relu' ))
model2.add(tf.keras.layers.Dense(16, activation='relu'))
model2.add(tf.keras.layers.Dense(16, activation='relu'))
model2.add(tf.keras.layers.Dense(16, activation='relu'))
model2.add(tf.keras.layers.Dense(1, activation='sigmoid'))


# In[71]:


model2.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy', get_f1])


# In[72]:


x = dia_norm.drop(columns = 'Outcome')
y = dia_norm['Outcome']


# In[73]:


history2 = model.fit(x,y, validation_split=0.33, batch_size=128, workers=4, epochs=200, verbose=3)


# In[74]:


max_metric(history2)
plot_this(history2)


# In[76]:


print("without normalization")
max_metric(history)

print("#################################################")

print("\nwith normalization")
max_metric(history2)


# In[ ]:




