#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM


# In[2]:


data = pd.read_csv('../input_data/GoogleStocks.csv', thousands=',')


# In[3]:


print data.head()
data = data.drop(['close','open','date'],axis=1)


# In[4]:


print data.head()


# In[5]:


data['avg'] = (data.high + data.low)/2.0


# In[6]:


data = data.drop(['high','low'],axis=1)
data = (data - data.mean())/data.std()


# In[7]:


data.head()


# In[8]:


def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix() #pd.DataFrame(stock)
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1][:,-1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:,-1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  

    return [x_train, y_train, x_test, y_test]


# In[9]:


def build_model2(cell,layers):
    d = 0.2
    model = Sequential()
    model.add(LSTM(cell, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(cell, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
#     model.add(Dense(16,init='uniform',activation='relu'))        
    model.add(Dense(1,init='uniform',activation='relu'))
    model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
    return model


# In[10]:


def build_model3(cell,layers):
    d = 0.2
    model = Sequential()
    model.add(LSTM(cell, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
    
    model.add(LSTM(cell, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
    
    model.add(LSTM(cell, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
#     model.add(Dense(16,init='uniform',activation='relu'))        
    model.add(Dense(1,init='uniform',activation='relu'))
    model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
    return model


# In[11]:


# time_step = 5
# X_train, Y_train, X_test, Y_test = load_data(data[::-1],time_step)


# In[12]:


# print X_train.shape


# In[13]:


# model = build_model3(30,[2,time_step,1])


# In[14]:


# model.fit(X_train, 
#           Y_train, 
#           batch_size=512, 
#           epochs=500, 
#           validation_split=0.1, 
#           verbose=0)


# In[15]:


# trainScore = model.evaluate(X_train, Y_train, verbose=0)
# print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

# testScore = model.evaluate(X_test, Y_test, verbose=0)

# print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))


# In[19]:


import matplotlib.pyplot as plt2

def plot(model,X_test,Y_test):
#     diff=[]
#     ratio=[]
    p = model.predict(X_test)
#     for u in range(len(Y_test)):
#         pr = p[u][0]
#         ratio.append((Y_test[u]/pr)-1)
#         diff.append(abs(Y_test[u]- pr))

    plt2.plot(p, label='prediction')
    plt2.plot(Y_test, label='y_test')
    plt2.legend()
    plt2.show()


# In[ ]:





# In[20]:



def rnn(l,c,ts):
    print "for ",l," layers ",c," cells ",ts," time-steps"
    X_train, Y_train, X_test, Y_test = load_data(data[::-1],ts)
    if l==2:
        model = build_model2(c,[2,ts,1])
    elif l==3:
        model = build_model3(c,[2,ts,1])

    model.fit(X_train, 
          Y_train, 
          batch_size=512, 
          epochs=500, 
          validation_split=0.1, 
          verbose=0)
    
    trainScore = model.evaluate(X_train, Y_train, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

    testScore = model.evaluate(X_test, Y_test, verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

    plot(model,X_test,Y_test)


# In[21]:


rnn(2,30,20)


# In[28]:


nhl = [2,3]
noc = [30,50,80]
nots = [20,50,75]

for l in nhl:
    for c in noc:
        for ts in nots:
            rnn(l,c,ts)


# In[ ]:




