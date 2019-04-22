#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


# In[2]:


df = pd.read_csv("../input_data/intrusion_detection/data.csv")
Y = df.xAttack
X = df.drop(['xAttack'],axis=1)
X = (X - X.mean())/X.std()
X.head()


# In[3]:


pca = PCA(.90)
principalComponents = pca.fit(X)
X_reduce = pca.transform(X)
X_new = pd.DataFrame( X_reduce,columns=[ "pc"+str(i) for i in xrange(X_reduce.shape[1]) ] )


# In[4]:


cov_mat = np.cov(X.T)


# In[5]:


U,S,V = np.linalg.svd(cov_mat)
S_total = float(np.sum(S))

sum_i = 0
num_of_comp = 0
for i in xrange(len(S)):
    sum_i += S[i]
    if sum_i  / S_total  >= 0.90:
        num_of_comp = i+1
        break
print num_of_comp


# In[6]:


U_reduce = U[:,:num_of_comp]


# In[ ]:





# In[7]:


Z = np.matmul(U_reduce.T, X.T)
Z = Z.T
Z_new = pd.DataFrame( Z,columns=[ "pc"+str(i) for i in xrange(Z.shape[1]) ] )

System's
# In[8]:


X_new.head()

Mine
# In[9]:


Z_new.head()


# In[ ]:




