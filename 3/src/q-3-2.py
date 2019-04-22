#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score,confusion_matrix,classification_report,accuracy_score


# In[3]:


df=pd.read_csv("../input_data/wine-quality/data.csv",sep=';')


# In[4]:


learning_rate = 0.01
iterations = 1000


# In[5]:


columns = df.columns


# In[6]:


X = df.drop(['quality'],axis=1)
Y = df['quality']


# In[7]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)


# In[8]:


tr_mean = X_train.mean()
tr_std = X_train.std()
te_mean = X_test.mean()
te_std = X_test.std()
X_train = (X_train - tr_mean)/tr_std
X_test = (X_test - te_mean)/te_std


# In[9]:


lr = LogisticRegression(solver='lbfgs',multi_class='multinomial',max_iter=iterations)
lr.fit(X_train, Y_train) 
y_pred = lr.predict(X_test)
score = lr.score(X_test,Y_test)
print(score)
print((Y_test == y_pred).mean())
print(confusion_matrix(Y_test,y_pred))
# print(classification_report(Y_test,y_pred))
print(accuracy_score(Y_test, y_pred)*100)


# In[10]:


X_train = pd.concat([X_train,Y_train],axis=1)
ones = np.ones([X_train.shape[0],1])
Y_train = X_train.iloc[:,11:12].values
X_train = X_train.iloc[:,0:11]
X_train = np.concatenate((ones,X_train),axis=1)


# In[11]:


theta = np.zeros([11,12])


# In[12]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# In[13]:


def gradient_decent(X_train,Y_train,theta1,learning_rate,iterations):
    
    for i in range(iterations):
        h = sigmoid(np.dot(X_train,theta1.T))
        theta1 = theta1 - (learning_rate/len(X_train)) * np.sum(X_train * (h - Y_train), axis=0)
    
    return theta1


# In[14]:


for i in range(0,11):
    theta1 = np.zeros([1,12])
    W = np.array(Y_train)
    for j in range(len(W)):
        if W[j] == i:
            W[j] = 1
        else:
            W[j] = 0
    
    theta[i]=gradient_decent(X_train,W,theta1,learning_rate,iterations)


# In[15]:


y_pred = []
for index,rows in X_test.iterrows():

    rows = list(rows)
    max_h=0
    for a in range(0,11):
        y = 0
        for i in range(len(rows)):
            y = y + rows[i]*theta[a][i+1]
        y = y + theta[a][0]
        y = sigmoid(y)
        if y >= max_h:
            label = a
            max_h = y
    y_pred.append(label)


# In[16]:


print(confusion_matrix(Y_test,y_pred))
print((y_pred == Y_test).mean()*100)
print(accuracy_score(Y_test, y_pred)*100)

