#!/usr/bin/env python
# coding: utf-8

# ## q-1
# Implement Lasso regression also known as L1 regularisation and plot graph between
# regularisation coefficient λ and error

# In[15]:


import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import sklearn as sk
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from pylab import *
import matplotlib
import matplotlib.pyplot as plt


# ###### loading dataset and preprocessing

# In[2]:


df = pd.read_csv("../input_data/AdmissionDataset/data.csv")
X = df.drop(['Serial No.','Chance of Admit '],axis=1)
Y = df['Chance of Admit ']
col_names = [i for i in X]
X = pd.DataFrame(preprocessing.scale(X), columns = col_names)

# scaler = StandardScaler()
# X = scaler.fit_transform(X)


# ###### splitting data in training and validation

# In[3]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# ###### using inbuilt linear regression model
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
pred = regressor.predict(X_test)
inbuilt_coeff = []
inbuilt_coeff.append(regressor.intercept_)
inbuilt_coeff.append(list(regressor.coef_))
# print(regressor.coef_)
# print(regressor.intercept_)
print inbuilt_coeff
# In[4]:


from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.01)
clf.fit(X_train, Y_train)


# ###### appending a column of ones at the beginning

# In[5]:


X_train1 = X_train.reset_index(drop=True)
Y_train1 = Y_train.reset_index(drop=True)

ones = pd.DataFrame(1,index=np.arange(X_train.shape[0]),columns=["ones"])
X_train1 = pd.concat([ones, X_train1],axis=1)
X_train1 = np.array(X_train1)
Y_train1 = np.array(Y_train1).reshape(X_train1.shape[0],1)
print X_train1[0]


# ###### initializing parameters for gradient descent

# In[6]:


theta = np.zeros([1,8])
alpha = 0.01
iterations = 1000


# ###### method to calculate values of theta using gradient descent

# In[7]:


costs = []
def gradientDescent(X,Y,theta,it,alpha,lamb):
    for i in range(it):
        theta = theta - (alpha/len(X)) * np.sum(X * (np.matmul(X, theta.T) - Y) + lamb*np.sign(theta), axis=0) 
    return theta

lamb = 0.01
g = gradientDescent(X_train1,Y_train1,theta,iterations,alpha,lamb)
theta_list = g[0]


# In[8]:


print theta_list

print(clf.intercept_)  
print(clf.coef_)


# ###### method to predict values for test_data

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


# print theta_list
# print r2_score(list(Y_test),pred)


# In[11]:


lamb = 0.01
lamb_list = []
train_error = []
test_error = []
sys_train_error = []
sys_test_error = []
flag=1
while lamb < 1:
    print lamb
    theta = np.zeros([1,8])
    g = gradientDescent(X_train1,Y_train1,theta,iterations,alpha,lamb)
    theta_list = g[0]
    
    clf = linear_model.Lasso(alpha=lamb)
    clf.fit(X_train, Y_train)
    
    pred = clf.predict(X_test)
    sys_test_err = mean_squared_error(Y_test, pred)
    sys_test_error.append(sys_test_err)
    
    pred = clf.predict(X_train)
    sys_train_err = mean_squared_error(Y_train, pred)
    sys_train_error.append(sys_train_err)
    
    pred = predict(X_test)
    test_err = mean_squared_error(Y_test, pred)
    test_error.append(test_err)
    
    pred = predict(X_train)
    train_err = mean_squared_error(Y_train, pred)
    train_error.append(train_err)
    
    lamb_list.append(lamb)
    lamb+=0.01
#     if lamb > -5 and flag==1:
#         lamb=0.01
#         flag=0
#     if lamb>=0:
#         lamb*=2
#     else:
#         lamb/=2
    
    
print len(lamb_list)
print len(test_error)
print len(train_error)


# In[12]:


# fig,ax = plt.subplots()
plt.plot(lamb_list, sys_train_error, label="train error")
plt.plot(lamb_list, sys_test_error, label="test error")
plt.legend()
print "system : lambda vs error"
plt.show()


# In[13]:


# fig,ax = plt.subplots()
plt.plot(lamb_list, train_error, label="train error")
plt.plot(lamb_list, test_error, label="test error")
plt.legend()
print "my model : lambda vs error"
plt.show()


# ###### testing from file
def test_function():
    test_file = sys.argv[1]
    df = pd.read_csv(test_file)
    return predict(df)

print test_function()
# In[ ]:




