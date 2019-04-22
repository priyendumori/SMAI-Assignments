#!/usr/bin/env python
# coding: utf-8

# ## q-1-2
# #### A bank is implementing a system to identify potential customers who have higher probablity of availing loans to increase its profit.  Implement Naive Bayes classifier on this dataset to help bank achieve its goal. 

# In[1]:


import numpy as np
from numpy import log2 as log
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import operator
import sys
from sklearn.naive_bayes import GaussianNB

from pylab import *
import matplotlib
import matplotlib.pyplot as plt


# ###### loading dataset

# In[2]:


df = pd.read_csv("input_data/LoanDataset/data.csv", names = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "Y", "k", "l", "m", "n"])
df = df.drop([0])


# ###### separating class label

# In[3]:


Y = df.Y
labels = Y.unique()
X = df.drop(['Y'], axis=1)


# ###### splitting data in training and validation

# In[4]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2)
df1 = pd.concat([X_train, Y_train],axis=1).reset_index(drop=True)


# ###### using inbuilt naive bayes

# In[5]:


gnb = GaussianNB()
gnb.fit(X_train, Y_train)
y_pred = gnb.predict(X_test)

print confusion_matrix(Y_test,y_pred)
print classification_report(Y_test,y_pred)
print accuracy_score(Y_test,y_pred)


# ###### splitting data according to class label (0/1)
# ###### storing their summary (i.e. mean, std, etc.)

# In[6]:


df_z = df1[df1.Y==0].reset_index(drop=True)
df_o = df1[df1.Y==1].reset_index(drop=True)
df_z_summary = df_z.describe().drop(['Y'],axis=1)
df_o_summary = df_o.describe().drop(['Y'],axis=1)
# print df_z_summary


# ###### method to calculate probability of a data point using gaussian distribution 

# In[7]:


def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


# ###### returns probability of inputVector being in either classes

# In[8]:


def calculateClassProbabilities(sum0, sum1, inputVector):
    probabilities = {0:1, 1:1}
    counter=0
    for col in sum0:
        x = inputVector[counter]
        counter+=1
        probabilities[0] *= calculateProbability(x, sum0[col]['mean'], sum0[col]['std'])
        
    counter=0
    for col in sum1:
        x = inputVector[counter]
        counter+=1
        probabilities[1] *= calculateProbability(x, sum1[col]['mean'], sum1[col]['std'])
        
    return probabilities


# ###### method uses above two methods and predicts label which is highest probable for one row

# In[9]:


def predict(sum0, sum1, inputVector):
    probabilities = calculateClassProbabilities(sum0, sum1, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


# ###### method calls predict method in a loop for all rows in test data

# In[10]:


def getPredictions(sum0, sum1, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(sum0, sum1, testSet.iloc[i])
        predictions.append(result)
    return predictions

p = getPredictions(df_z_summary, df_o_summary, X_test)
print confusion_matrix(Y_test,p)
print classification_report(Y_test,p)
print accuracy_score(Y_test,p)


# ###### Observation
# * Very simple, easy to implement and fast.
# * Can be used for both binary and mult-iclass classification problems.
# * Can make probabilistic predictions.
# * If the NB conditional independence assumption holds, then it will converge quicker than discriminative models like logistic regression.
# * Even if the NB assumption doesn’t hold, it works great in practice.
# * Need less training data.
# * It can’t learn interactions between features

# ###### testing from file

# In[11]:


def test_function():
    test_file = sys.argv[1]
    df = pd.read_csv(test_file, names = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n"])
    pred = getPredictions(df_z_summary, df_o_summary, df)
    return pred

print test_function()

