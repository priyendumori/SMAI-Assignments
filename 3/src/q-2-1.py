#!/usr/bin/env python
# coding: utf-8

# ## q-2-1

# In[1]:


import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import numpy as np
import sklearn as sk
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[2]:


df = pd.read_csv("../input_data/AdmissionDataset/data.csv")
threshold = 0.5
df.loc[df['Chance of Admit ']<threshold,'Chance of Admit '] = 0
df.loc[df['Chance of Admit ']>=threshold,'Chance of Admit '] = 1
X = df.drop(['Serial No.','Chance of Admit '],axis=1)
Y = df['Chance of Admit ']

col_names = [i for i in X]
X = pd.DataFrame(preprocessing.scale(X), columns = col_names)


# In[3]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

regressor=LinearRegression()
regressor.fit(X_train,Y_train)
pred = regressor.predict(X_test) 
print(regressor.coef_)
print(regressor.intercept_)
r2_score(Y_test,pred)
# In[4]:


logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, Y_train)
Z = logreg.predict(X_test)
print confusion_matrix(Y_test,Z)
print classification_report(Y_test,Z)
print accuracy_score(Y_test,Z)


# In[5]:


X_train1 = X_train.reset_index(drop=True)
Y_train1 = Y_train.reset_index(drop=True)


# In[6]:


ones = pd.DataFrame(1,index=np.arange(X_train.shape[0]),columns=["ones"])
X_train1 = pd.concat([ones, X_train1],axis=1)
X_train1 = np.array(X_train1)
Y_train1 = np.array(Y_train1).reshape(X_train1.shape[0],1)


# In[7]:


theta = np.zeros([1,8])
alpha = 0.01
iterations = 1000


# In[8]:


def h(X):
    X=-X
    return 1/(1+np.exp(X))


# In[9]:


def gradientDescent(X,Y,theta,it,alpha):
    for i in range(it):
        theta = theta - (alpha) * np.sum(X * (h(np.matmul(X, theta.T)) - Y), axis=0)
    return theta

g = gradientDescent(X_train1,Y_train1,theta,iterations,alpha)
theta_list = g[0]


# In[10]:


def predict(X_test):
    Y_pred=[]
    for index,row in X_test.iterrows():
        row=list(row)
        y1=0
        for i in range(1,8):
            y1=y1+theta_list[i]*row[i-1]
        y1=y1+theta_list[0]
        Y_pred.append(0 if y1<0.5 else 1)
    return Y_pred
pred = predict(X_test)


# In[11]:


# print r2_score(list(Y_test),pred)
# print theta_list

print confusion_matrix(Y_test,pred)
print classification_report(Y_test,pred)
print accuracy_score(Y_test,pred)

