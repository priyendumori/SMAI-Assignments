#!/usr/bin/env python
# coding: utf-8

# ## q-1-1-1
# #### Implement a KNN classifier.

# In[73]:


import numpy as np
from numpy import log2 as log
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
import operator
import sys


# ###### loading dataset

# In[74]:


dataset = sys.argv[1]
# testfile = sys.argv[2]
# dataset = "iris"
def loadfile(dataset):
    if dataset=="iris":
        filename = 'input_data/Iris.csv'
        df = pd.read_csv(filename, names = ["a", "b", "c", "d", "Y"])
    elif dataset=="robot1":
        filename = 'input_data/Robot1'
        df = pd.read_csv(filename, delim_whitespace=True, names = ["Y", "a", "b", "c", "d", "e", "f", "g"])
        df = df.drop(['g'],axis=1)
    else:
        filename = 'input_data/Robot2'
        df = pd.read_csv(filename, delim_whitespace=True, names = ["Y", "a", "b", "c", "d", "e", "f", "g"])
        df = df.drop(['g'],axis=1)
    return df

df = loadfile(dataset)


# ###### seperating class label and preprocessing data if required

# In[75]:


Y = df.Y
labels = Y.unique()
X = df.drop(['Y'], axis=1)
if dataset != "iris":
    X = pd.DataFrame(preprocessing.normalize(X), columns = ["a","b", "c", "d", "e", "f"])
    


# ###### splitting data into training and validation

# In[76]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2)
df1 = pd.concat([X_train, Y_train],axis=1).reset_index(drop=True)


# ###### using inbuilt knn classifier

# In[77]:


neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, Y_train)
p = neigh.predict(X_test)
print confusion_matrix(Y_test,p)
print classification_report(Y_test,p)
print accuracy_score(Y_test,p)


# ###### method to calculate euclidean distance

# In[78]:


def euclidean_distance(x, y):   
    return np.sqrt(np.sum((x - y) ** 2))


# ###### predict method calculates distances of test data point to all the points available
# ###### store them in ascending order
# ###### pick first k points from the sorted list
# ###### count occurences of class labels
# ###### assign highest occurring class lable from these k points to our test data point

# In[79]:


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
            labeldict[df1.iloc[i].Y]+=1
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

p = predict(X_test,5)
print confusion_matrix(Y_test,p)
print classification_report(Y_test,p)
print accuracy_score(Y_test,p)


# In[80]:


def test_function(dataset):
    testfile = sys.argv[2]
    if dataset=="iris":
        df = pd.read_csv(testfile, names = ["a", "b", "c", "d"])
    elif dataset=="robot1":
        df = pd.read_csv(testfile, delim_whitespace=True, names = ["a", "b", "c", "d", "e", "f", "g"])
        df = df.drop(['g'],axis=1)
    else:
        df = pd.read_csv(testfile, delim_whitespace=True, names = ["a", "b", "c", "d", "e", "f", "g"])
        df = df.drop(['g'],axis=1)
    return predict(df,5)

print test_function(dataset)

