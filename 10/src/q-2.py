#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split


# In[2]:


sequence = "CTTCATGTGAAAGCAGACGTAAGTCA"
state_path = "EEEEEEEEEEEEEEEEEE5IIIIIII$"


# In[3]:


emission_prob = {
                    ("E","A"):0.25,
                    ('E','C'):0.25,
                    ("E","G"):0.25,
                    ("E","T"):0.25,
    
                    ("5","A"):0.05,
                    ("5","C"):0,
                    ("5","G"):0.95,
                    ("5","T"):0,
                    
                    ("I","A"):0.4,
                    ("I","C"):0.1,
                    ("I","G"):0.1,
                    ("I","T"):0.4
    
                }
transition_prob = {
                    ("^","E"):1,
                    ("E","E"):0.9,
                    ("E","5"):0.1,
                    ("5","I"):1,
                    ("I","I"):0.9,
                    ("I","$"):0.1
                  }


# In[4]:


prob = transition_prob[("^","E")]


# In[5]:



for i in xrange(len(sequence)):
    state1 = state_path[i]
    state2 = state_path[i+1]
    seq = sequence[i]
    
    e = emission_prob[(state1,seq)]
    t = transition_prob[(state1, state2)]
    
    prob *= e*t


# In[6]:


print prob


# In[7]:


print np.log(prob)


# In[ ]:




