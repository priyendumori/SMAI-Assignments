#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random as rd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import math


# In[2]:


df = pd.read_csv("../input_data/intrusion_detection/data.csv")
Y = df.xAttack
X = df.drop(['xAttack'],axis=1)
X = (X - X.mean())/X.std()


# In[3]:


cov_x = np.cov(X.T)

U,S,V = np.linalg.svd(cov_x)
S_total = float(np.sum(S))

sum_i = 0
num_of_comp = 0
for i in xrange(len(S)):
    sum_i += S[i]
    if sum_i  / S_total  >= 0.90:
        num_of_comp = i+1
        break

U_red = U[:,:num_of_comp]

Z = np.matmul(U_red.T, X.T)
Z = Z.T
Z_new = pd.DataFrame( Z,columns=[ "pc"+str(i) for i in xrange(Z.shape[1]) ] )


# In[ ]:





# In[4]:


kmeans = KMeans(n_clusters=5, random_state=0).fit(Z_new)
cluster_scikit = kmeans.labels_


# In[5]:


def euclidean_distance(x,y):
    return np.sum((x - y)**2)


# In[6]:


def kmeans(K,df):

    d = df.shape[1] 
    n = df.shape[0]
    Max_Iterations = 100
    i = 0
    
    cluster = [0] * n
    prev_cluster = [-1] * n
    
    cluster_centers = [rd.choice(df) for i in xrange(K) ]    
    force_recalculation = False
    
    while (cluster != prev_cluster) or (i > Max_Iterations) or (force_recalculation) :
        prev_cluster = list(cluster)
        force_recalculation = False
        i += 1
    
        for p in xrange(n):
            min_dist = float("inf")
            for c in xrange(K):
                dist = euclidean_distance(df[p],cluster_centers[c])
                if (dist < min_dist):
                    min_dist = dist  
                    cluster[p] = c
        
        for k in xrange(K):
            new_center = [0] * d
            members = 0
            for p in xrange(n):
                if (cluster[p] == k):
                    for j in xrange(d):
                        new_center[j] += df[p][j]
                    members += 1
            
            for j in xrange(d):
                if members != 0:
                    new_center[j] = new_center[j] / float(members) 
                else: 
                    new_center = rd.choice(df)
                    force_recalculation = True                    
            
            cluster_centers[k] = new_center
            
    return cluster


# In[7]:


cluster = kmeans(5, Z)


# In[8]:


Z_new = pd.DataFrame( Z,columns=[ "pc"+str(i) for i in xrange(Z.shape[1]) ] )
Z_new = pd.concat([Z_new, Y], axis=1)

pred_Y = pd.DataFrame( cluster,columns=[ 'pred_Y' ] )
pred_Y_scikit = pd.DataFrame( cluster_scikit,columns=[ 'pred_Y' ] )

Z_mymodel = pd.concat([Z_new, pred_Y],axis=1)
Z_scikit  = pd.concat([Z_new, pred_Y_scikit],axis=1)


# In[9]:


def purity(df,K):
    purity_dict = {}
    for i in xrange(K):
        sub_table = df[ df['pred_Y'] == i ]
        label, count = np.unique(sub_table['xAttack'],return_counts=True)
        mx_ind = np.argmax(count)
        print i , label[mx_ind]
        purity_dict[i] = count[mx_ind] / float(len(sub_table))
        print label
        print count
    return purity_dict


# In[10]:


purity_dict = purity(Z_mymodel,5)
print purity_dict


# In[11]:


purity_scikit = purity(Z_scikit,5)
print purity_scikit


# In[ ]:




