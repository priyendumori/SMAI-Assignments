#!/usr/bin/env python
# coding: utf-8

# In[65]:


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
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split


# In[66]:


data = pd.read_csv('../input_data/GoogleStocks.csv', thousands=',')
data = data.iloc[::-1].reset_index(drop=True)


# In[67]:


print data.head()
data = data.drop(['close','open','date'],axis=1)


# In[68]:


print data.head()


# In[69]:


data['avg'] = (data.high + data.low)/2.0


# In[70]:


data = data.drop(['high','low'],axis=1)
data = (data - data.mean())/data.std()


# In[71]:


data.head()


# In[72]:


def load_data(stock):
    print "stock ",stock.shape
    amount_of_features = len(stock.columns)
    data = stock.as_matrix() #pd.DataFrame(stock)
    print data.shape
#     sequence_length = seq_len + 1
#     result = []
#     for index in range(len(data) - sequence_length):
#         print index, data[index: index + sequence_length].shape
#         result.append(data[index: index + sequence_length])
    
#     result = np.array(result)
#     print "result ",result.shape
    row = round(0.9 * data.shape[0])
    train = data[:int(row), :]
    print "train ",train.shape
    x_train = train[:, :-1]
    y_train = train[:, -1][:,-1]
    print "x train ",x_train.shape
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:,-1]
    print "x train ",x_train.shape
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    print "x train ",x_train.shape
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  

    return [x_train, y_train, x_test, y_test]

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
    return modeldef build_model3(cell,layers):
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
# In[73]:


# time_step = 5
# X_train, Y_train, X_test, Y_test = load_data(data[::-1],time_step)


# In[74]:


# print X_train.shape


# In[75]:


# model = build_model3(30,[2,time_step,1])


# In[76]:


# model.fit(X_train, 
#           Y_train, 
#           batch_size=512, 
#           epochs=500, 
#           validation_split=0.1, 
#           verbose=0)


# In[77]:


# trainScore = model.evaluate(X_train, Y_train, verbose=0)
# print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

# testScore = model.evaluate(X_test, Y_test, verbose=0)

# print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))


# In[90]:


import matplotlib.pyplot as plt2

def plot(model,X_test,Y_test):
    samples, _ = model.sample(len(X_test))

    volume = samples[:,0]
    avg = samples[:,1]

    plt2.figure()
    plt2.plot(np.arange(len(X_test)),avg,label='predicted avg')
    plt2.plot(np.arange(len(X_test)),X_test,label='actual avg')
    plt2.show()


# In[ ]:





# In[94]:



def hmm(hs,ts):
    
#     X_train, Y_train, X_test, Y_test = load_data(data[::-1])
    X = data.volume
    Y = data.avg
    row = round(0.9 * X.shape[0])

    X_train = X[:int(row)]
    X_test = X[int(row):]
    
    Y_train = Y[:int(row)]
    Y_test = Y[int(row):]
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
#     print type(X_train)
    hmm = GaussianHMM(n_components=hs,covariance_type="diag", n_iter=ts)
    train = np.column_stack([X_train,Y_train])
    hmm.fit(train)
    
    
#     samples, _ = hmm.sample(len(X_test))
#     avg = samples[:,1]
    
#     plt2.figure()
#     plt2.plot(np.arange(len(avg)),avg,label='predicted avg')
#     plt2.plot(np.arange(len(Y_test)),Y_test,label='actual avg')
#     logprob, de = hmm.decode(train)
    
    
    plot(hmm,X_test,Y_test)


# In[95]:


# hmm(4,20)


# In[96]:


hs = [4,8,12]
nots = [20,50,75]

for c in hs:
    for ts in nots:
        print "for ",c," hidden states ",ts," time-steps"
        hmm(c,ts)


# In[ ]:





# In[ ]:




