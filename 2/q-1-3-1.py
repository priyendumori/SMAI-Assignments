#!/usr/bin/env python
# coding: utf-8

# ## q-1-3-1
# #### Implement a model using linear regression to predict the probablity of getting the admit.

# In[1]:


import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import numpy as np
import sklearn as sk
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# ###### loading dataset and preprocessing

# In[2]:


df = pd.read_csv("input_data/AdmissionDataset/data.csv")
X = df.drop(['Serial No.','Chance of Admit '],axis=1)
Y = df['Chance of Admit ']
col_names = [i for i in X]
X = pd.DataFrame(preprocessing.scale(X), columns = col_names)


# ###### splitting data in training and validation

# In[3]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# ###### using inbuilt linear regression model

# In[4]:


regressor=LinearRegression()
regressor.fit(X_train,Y_train)
pred = regressor.predict(X_test)
inbuilt_coeff = []
inbuilt_coeff.append(regressor.intercept_)
inbuilt_coeff.append(list(regressor.coef_))
# print(regressor.coef_)
# print(regressor.intercept_)
print inbuilt_coeff
r2_score(Y_test,pred)
print pred


# ###### appending a column of ones at the beginning

# In[5]:


X_train1 = X_train.reset_index(drop=True)
Y_train1 = Y_train.reset_index(drop=True)

ones = pd.DataFrame(1,index=np.arange(X_train.shape[0]),columns=["ones"])
X_train1 = pd.concat([ones, X_train1],axis=1)
X_train1 = np.array(X_train1)
Y_train1 = np.array(Y_train1).reshape(X_train1.shape[0],1)


# ###### initializing parameters for gradient descent

# In[6]:


theta = np.zeros([1,8])
alpha = 0.01
iterations = 1000


# ###### method to calculate values of theta using gradient descent

# In[7]:


def gradientDescent(X,Y,theta,it,alpha):
    for i in range(it):
        theta = theta - (alpha/len(X)) * np.sum(X * (np.matmul(X, theta.T) - Y), axis=0)
    return theta

g = gradientDescent(X_train1,Y_train1,theta,iterations,alpha)
theta_list = g[0]


# ###### method to predict values for test_data

# In[8]:


def predict(X_test):
    Y_pred=[]
    for index,row in X_test.iterrows():
        row=list(row)
        y1=0
        for i in range(1,8):
            y1=y1+theta_list[i]*row[i-1]
        y1=y1+theta_list[0]
        Y_pred.append(y1)
    return Y_pred
pred = predict(X_test)


# In[9]:


print theta_list
print r2_score(list(Y_test),pred)
print pred


# ###### testing from file

# In[10]:


def test_function():
    test_file = sys.argv[1]
    df = pd.read_csv(test_file)
    return predict(df)

print test_function()

