#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


# In[1]:


digits = load_digits()
X = digits.data
print X.shape
X = pd.DataFrame( X,columns=[ str(i) for i in xrange(X.shape[1]) ] )

X = StandardScaler().fit_transform(X)
X = pd.DataFrame( X,columns=[ str(i) for i in xrange(X.shape[1]) ] )

cov_x = np.cov(X.T)
cov_x.shape

U,S,V = np.linalg.svd(cov_x)


# In[2]:


def reduce_dim(no_of_components, U, X):
    U_red = U[:,:no_of_components]
    X=np.array(X)
    Z = np.matmul(U_red.T, X.T)
    Z = Z.T
    Z_new = pd.DataFrame( Z,columns=[ "pc"+str(i) for i in xrange(Z.shape[1]) ] )
    return Z_new


# In[3]:


U_copy = U.copy()
X_copy = X.copy()
Z_1 = reduce_dim(17, U_copy, X_copy)

U_copy = U.copy()
X_copy = X.copy()
Z_2 = reduce_dim(26, U_copy, X_copy)

U_copy = U.copy()
X_copy = X.copy()
Z_3 = reduce_dim(38, U_copy, X_copy)


# In[4]:


print Z_1.shape
print Z_2.shape
print Z_3.shape


# In[5]:


def print_score(data):
    params = {'bandwidth': np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params, cv=5)
    grid.fit(data)

    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

    # use the best estimator to compute the kernel density estimate
    kde = grid.best_estimator_

    score = kde.score_samples(data)
    print score.shape
    for i in score:
        print i


# In[6]:


print_score(Z_1)


# In[7]:


print_score(Z_2)


# In[8]:


print_score(Z_3)


# In[ ]:




