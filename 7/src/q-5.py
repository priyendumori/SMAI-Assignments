#!/usr/bin/env python
# coding: utf-8

# ## q-5
# 
# In this part implement regression with k-fold cross validation. Analyse how behav-
# ior changes with different values of k. Also implement a variant of this which is the
# leave-one-out cross validation.

# In[1]:


import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import sklearn as sk
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import linear_model

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
X_copy = X.copy(deep=True)
Y_copy = Y.copy(deep=True)
X = X.values
Y = Y.values
n = df.shape[0]
df.shape


# In[3]:


theta = np.zeros([1,8])
alpha = 0.01
iterations = 1000


# In[4]:


def gradientDescent(X,Y,theta,it,alpha,lamb):
    for i in range(it):
        theta = theta - (alpha/len(X)) * np.sum(X * (np.matmul(X, theta.T) - Y) + lamb*np.sign(theta), axis=0) 
    return theta

lamb = 0.01
# g = gradientDescent(X_train1,Y_train1,theta,iterations,alpha,lamb)
# theta_list = g[0]


# In[5]:


def predict(X_test,theta_list):
    Y_pred=[]
    for index,row in X_test.iterrows():
        row=list(row)
        y1=0
        for i in range(1,8):
            y1=y1+theta_list[i]*row[i-1]
        y1=y1+theta_list[0]
        Y_pred.append(y1)
    return Y_pred
# pred = predict(X_test)


# In[6]:


def getFolds(k,X):
    no_of_indices = len(X)/k
#     print no_of_indices
    start_indices = []
    t = 0
    while t<len(X):
        start_indices.append(t)
        t+=no_of_indices
    
    if len(start_indices) > k:
        start_indices = start_indices[:-1]
#     print start_indices
    
    test_fold = 0
    folds = []
    train_array = np.array([],dtype=int32)
    for i in xrange(k):
        train_array = []
        for s in xrange(len(start_indices)-1):
#             print start_indices[s],test_fold
            if s==test_fold:
#                 print "1 test ",start_indices[s], "to ",start_indices[s+1]
                test_array = np.arange(start_indices[s],start_indices[s+1],dtype=int32)
            else:
#                 print "1 train ",start_indices[s], "to ",start_indices[s+1]
                temp = np.arange(start_indices[s],start_indices[s+1],dtype=int32)
#                 print type(temp)
#                 temp_list = list(temp)
#                 print type(temp_list)
#                 print type(train_array)
#                 train_array.append(temp)
#                 train_array = train_array.flatten()
                train_array = np.append(train_array,temp)
                train_array = train_array.astype(np.int32)

                
#         print "adf ",len(start_indices)-1
        if test_fold == len(start_indices)-1:
#             print "2 test ",start_indices[-1], "to ",len(X)
            test_array = np.arange(start_indices[-1],len(X),dtype=int32)
        else:
#             print "2 train ",start_indices[-1], "to ",len(X)
            temp = np.arange(start_indices[-1],len(X),dtype=int32)
#             temp_list = list(temp)
#             train_array.append(temp)
            train_array = np.append(train_array,temp)
#             train_array = train_array.astype(np.int32)

#         train_array = np.array(train_array)
#         train_array = train_array.flatten()
        test_array = test_array.flatten()
        train_array = train_array.astype(np.int32)
        folds.append((train_array,test_array))
        test_fold+=1
    return folds


# In[7]:


temp = np.ones(9)
temp.shape
getFolds(9,temp)


# In[8]:


def KFolds(k):
#     kf = KFold(n_splits=k)
    kf = getFolds(k,X)
    mse_train = []
    mse_test = []
#     for train_index, test_index in kf.split(X):
    for train_index, test_index in kf:
#         print("TRAIN:", train_index, "TEST:", test_index)
#         print type(train_index[0])
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        clf = linear_model.Lasso(alpha=0.01)
        clf.fit(X_train, Y_train)

        z = np.ones((X_train.shape[0],1), dtype=float64)
        X_train = np.append(z,X_train,axis=1)
        Y_train = Y_train.reshape(Y_train.shape[0],1)

        g = gradientDescent(X_train,Y_train,theta,iterations,alpha,lamb)
        theta_list = g[0]
#         print theta_list
#         print 
#         print(clf.intercept_)  
#         print(clf.coef_)
#         print 
#         print

        X_test = pd.DataFrame(X_test)
        pred = predict(X_test,theta_list)
        mse_test.append(mean_squared_error(pred,Y_test))
        
        X_train = pd.DataFrame(X_train)
        pred = predict(X_train,theta_list)
        mse_train.append(mean_squared_error(pred,Y_train))
    return sum(mse_test) / len(mse_test), sum(mse_train) / len(mse_train) 


# In[9]:


for i in xrange(2,16):
    mse_test, mse_train = KFolds(i)
    print "for k=",i," mse on test = ",mse_test," mse on train = ",mse_train


# In[10]:


print "leave one out : "
mse_test, mse_train = KFolds(n)
print "for k=",n," mse on test = ",mse_test," mse on train = ",mse_train


# ###### splitting data in training and validation

# In[11]:


# X_train_c, X_test_c, Y_train_c, Y_test_c = train_test_split(X_copy, Y_copy, test_size=0.2)


# ###### using inbuilt linear regression model
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
pred = regressor.predict(X_test)
inbuilt_coeff = []
inbuilt_coeff.append(regressor.intercept_)
inbuilt_coeff.append(list(regressor.coef_))
# print(regressor.coef_)
# print(regressor.intercept_)
print inbuilt_coeff# clf = linear_model.Lasso(alpha=0.01)
# clf.fit(X_train_c, Y_train_c)
# pred = clf.predict(X_test_c)

X_train_c = X_train_c.values
Y_train_c = Y_train_c.values
z = np.ones((X_train_c.shape[0],1), dtype=float64)
X_train_c = np.append(z,X_train_c,axis=1)
Y_train_c = Y_train_c.reshape(Y_train_c.shape[0],1)


theta = np.zeros([1,8])
alpha = 0.01
iterations = 1000

g = gradientDescent(X_train_c,Y_train_c,theta,iterations,alpha,lamb)
theta_list = g[0]
pred = predict(X_test_c,theta_list)

print r2_score(pred,Y_test_c)
# ###### appending a column of ones at the beginning
X_train1 = X_train.reset_index(drop=True)
Y_train1 = Y_train.reset_index(drop=True)

ones = pd.DataFrame(1,index=np.arange(X_train.shape[0]),columns=["ones"])
X_train1 = pd.concat([ones, X_train1],axis=1)
X_train1 = np.array(X_train1)
Y_train1 = np.array(Y_train1).reshape(X_train1.shape[0],1)
# ###### initializing parameters for gradient descent

# ###### method to calculate values of theta using gradient descent
print theta_list

print(clf.intercept_)  
print(clf.coef_)
# ###### method to predict values for test_data

# In[12]:


# print theta_list
# print r2_score(list(Y_test),pred)

lamb = 0.01
lamb_list = []
train_error = []
test_error = []
sys_train_error = []
sys_test_error = []
flag=1
while lamb < 10000:
    print lamb
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
    lamb*=2
#     if lamb > -5 and flag==1:
#         lamb=0.01
#         flag=0
#     if lamb>=0:
#         lamb*=2
#     else:
#         lamb/=2
    
    
print len(lamb_list)
print len(test_error)
print len(train_error)fig,ax = plt.subplots()
ax.plot(lamb_list, sys_train_error, label="train error")
ax.plot(lamb_list, sys_test_error, label="test error")
ax.legend()
ax.set_xlabel("lambda")
ax.set_ylabel("error")
ax.set_title("system : lambda vs error")
show()fig,ax = plt.subplots()
ax.plot(lamb_list, train_error, label="train error")
ax.plot(lamb_list, test_error, label="test error")
ax.legend()
ax.set_xlabel("lambda")
ax.set_ylabel("error")
ax.set_title("my model : lambda vs error")
show()
# ###### testing from file
def test_function():
    test_file = sys.argv[1]
    df = pd.read_csv(test_file)
    return predict(df)

print test_function()
# In[ ]:




