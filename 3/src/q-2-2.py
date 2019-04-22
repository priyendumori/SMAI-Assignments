#!/usr/bin/env python
# coding: utf-8

# ## q-2-2

# In[1]:


import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import numpy as np
import sklearn as sk
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
np.seterr(divide='ignore', invalid='ignore')
import operator

from pylab import *
import matplotlib
import matplotlib.pyplot as plt


# In[2]:


df1 = pd.read_csv("../input_data/AdmissionDataset/data.csv")
# print df1.describe()
df = df1.copy(deep=True)
threshold = 0.5
df.loc[df['Chance of Admit ']<threshold,'Chance of Admit '] = 0
df.loc[df['Chance of Admit ']>=threshold,'Chance of Admit '] = 1
X = df.drop(['Serial No.','Chance of Admit '],axis=1)
Y = df['Chance of Admit ']
labels = Y.unique()
col_names = [i for i in X]
X = pd.DataFrame(preprocessing.scale(X), columns = col_names)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# In[3]:


def knn(X_train, Y_train,k):
    df1 = pd.concat([X_train, Y_train],axis=1).reset_index(drop=True)
    
    def euclidean_distance(x, y):   
        return np.sqrt(np.sum((x - y) ** 2))

    def predict(X_test,k):
        Y_predict = []
        for index, row in X_test.iterrows():
            dist = {}
            labeldict = {i:0 for i in labels}
            for index1, row1 in df1.iterrows():
                dist[index1] = euclidean_distance(row,row1)

            od = sorted(dist.items(), key=operator.itemgetter(1))
            count = k
            for i,j in od:
                count-=1
                labeldict[df1.iloc[i]['Chance of Admit ']]+=1
                if count==0:
                    break

            ans_label=0
            ans_count=-1
            for i,j in labeldict.iteritems():
                if j>=ans_count:
                    ans_label=i
                    ans_count=j
            Y_predict.append(ans_label)
        return Y_predict

    p = predict(X_test,k)
    print confusion_matrix(Y_test,p)
    print classification_report(Y_test,p)
    print accuracy_score(Y_test,p)


# In[4]:


def logistic(X_train,Y_train):
    

    X_train1 = X_train.reset_index(drop=True)
    Y_train1 = Y_train.reset_index(drop=True)

    ones = pd.DataFrame(1,index=np.arange(X_train.shape[0]),columns=["ones"])
    X_train1 = pd.concat([ones, X_train1],axis=1)
    X_train1 = np.array(X_train1)
    Y_train1 = np.array(Y_train1).reshape(X_train1.shape[0],1)

    theta = np.zeros([1,8])
    alpha = 0.01
    iterations = 1000

    def h(X):
        X=-X
        return 1/(1+np.exp(X))

    def gradientDescent(X,Y,theta,it,alpha):
        for i in range(it):
            theta = theta - (alpha) * np.sum(X * (h(np.matmul(X, theta.T)) - Y), axis=0)
        return theta

    g = gradientDescent(X_train1,Y_train1,theta,iterations,alpha)
    theta_list = g[0]

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

    # print r2_score(list(Y_test),pred)
    # print theta_list

    
#     average_precision = average_precision_score(Y_test, pred)
#     recall_value = recall_score(Y_test, pred)
    print confusion_matrix(Y_test,pred)
    print classification_report(Y_test,pred)
    print accuracy_score(Y_test,pred)


# In[5]:


knn(X_train.copy(deep=True),Y_train.copy(deep=True),5)
print "#########################################################"
logistic(X_train.copy(deep=True),Y_train.copy(deep=True))

