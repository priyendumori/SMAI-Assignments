#!/usr/bin/env python
# coding: utf-8

# ## q-1-3-2
# #### Compare  the  performance  of  Mean  square  error  loss  function  vs  Mean  Absolute error function vs Mean absolute percentage error function and explain the reasons for the observed behaviour.

# In[1]:


import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import numpy as np
import sklearn as sk
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv("input_data/AdmissionDataset/data.csv")
X = df.drop(['Serial No.','Chance of Admit '],axis=1)
Y = df['Chance of Admit ']
col_names = [i for i in X]
X = pd.DataFrame(preprocessing.scale(X), columns = col_names)


# In[3]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# In[4]:


regressor=LinearRegression()
regressor.fit(X_train,Y_train)
sys_pred = regressor.predict(X_test) 
inbuilt_coeff = []
inbuilt_coeff.append(regressor.intercept_)
inbuilt_coeff.append(list(regressor.coef_))
# print(regressor.coef_)
# print(regressor.intercept_)
r2_score(Y_test,sys_pred)


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


def gradientDescent(X,Y,theta,it,alpha):
    for i in range(it):
        theta = theta - (alpha/len(X)) * np.sum(X * (np.matmul(X, theta.T) - Y), axis=0)
    return theta

g = gradientDescent(X_train1,Y_train1,theta,iterations,alpha)
theta_list = g[0]


# In[ ]:





# In[9]:


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


# In[10]:


print r2_score(list(Y_test),pred)
print theta_list


# ###### methods to calculate all types of errors

# In[11]:


def mean_percentage_error(Y_test,pred):
    return np.mean(np.abs( Y_test - pred ) / Y_test) * 100.0

def mean_asolute_error(Y_test,pred):
    return np.mean(np.abs( Y_test - pred ))

def mean_squared_error(Y_test,pred):
    return np.mean( ( Y_test - pred )**2 )


# In[14]:


print "My model"
print mean_percentage_error(Y_test,pred)
print mean_asolute_error(Y_test,pred)
print mean_squared_error(Y_test,pred)

print "\nSystem's model"
print mean_percentage_error(Y_test,sys_pred)
print mean_asolute_error(Y_test,sys_pred)
print mean_squared_error(Y_test,sys_pred)


# ###### Observation
# * MAE gives less weight to outliers, which is not sensitive to outliers.
# 
# * MSE has the benefit of penalizing large errors more so can be more appropriate in some cases, for example, if being off by 10 is more than twice as bad as being off by 5. But if being off by 10 is just twice as bad as being off by 5, then MAE is more appropriate.
# 
# * Similar to MAE, but normalized by true observation. Downside is when true obs is zero, this metric will be problematic.
# 
