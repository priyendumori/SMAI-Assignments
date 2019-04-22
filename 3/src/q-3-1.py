#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score,confusion_matrix,classification_report,accuracy_score


# In[2]:


df=pd.read_csv("../input_data/wine-quality/data.csv", sep=';')


# In[3]:


learning_rate = 0.01
iterations = 1000
threshold = 0.5


# In[4]:


columns = df.columns


# In[5]:


X = df.drop(['quality'],axis=1)
Y = df['quality']


# In[6]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)


# In[7]:


tr_mean = X_train.mean()
tr_std = X_train.std()
te_mean = X_test.mean()
te_std = X_test.std()
X_train = (X_train - tr_mean)/tr_std
X_test = (X_test - te_mean)/te_std


# In[8]:


X_train = pd.concat([X_train,Y_train],axis=1)
ones = np.ones([X_train.shape[0],1])
Y_train = X_train.iloc[:,11:12].values
X_train = X_train.iloc[:,0:11]
X_train = np.concatenate((ones,X_train),axis=1)


# In[9]:


theta = np.zeros([15,12])


# In[10]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# In[11]:


def gradient_decent(X_train,Y_train,theta1,learning_rate,iterations):
    
    for i in range(iterations):
        h = sigmoid(np.dot(X_train,theta1.T))
        theta1 = theta1 - (learning_rate/len(X_train)) * np.sum(X_train * (h - Y_train), axis=0)
    
    return theta1


# In[12]:


x=0
for i in range(4,9):
    for j in range(i+1,10):
        theta1 = np.zeros([1,12])
        
        Xcopy = []
        Ycopy = []
#         print(Y_train.shape)
        W = np.array(Y_train)
#         print(W.shape)
        for k in range(len(W)):
            if W[k] == j or W[k]==i:
                Xcopy.append(X_train[k])
                if W[k] == i:
                    Ycopy.append(1)
                else:
                    Ycopy.append(0)
        
        Xcopy = np.array(Xcopy)
        Ycopy = np.array(Ycopy)
        Ycopy = Ycopy.reshape((len(Ycopy),1))

        theta[x]=gradient_decent(Xcopy,Ycopy,theta1,learning_rate,iterations)
        x = x + 1


# In[13]:


y_pred = []
for index,rows in X_test.iterrows():

    rows = list(rows)
    counts = {}
    label = 0
    for i in range(4,10):
        counts[i]=0
    max_h = 0
    c = 0
    for a in range(4,9):
        for b in range(a+1,10):
            y = 0
            for i in range(len(rows)):
                y = y + rows[i]*theta[c][i+1]
            y = y + theta[c][0]
            y = sigmoid(y)
            c = c + 1
            if y >= threshold:
                counts[a]=counts[a]+1
            else:
                counts[b]=counts[b]+1
    for i in range(4,10):
        if(counts[i]>=max_h):
            max_h=counts[i]
            label=i
    y_pred.append(label)


# In[14]:


print(confusion_matrix(Y_test,y_pred))
print((y_pred == Y_test).mean()*100)
print(accuracy_score(Y_test, y_pred)*100)


# In[ ]:




