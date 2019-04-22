#!/usr/bin/env python
# coding: utf-8

# ## q-1-1
# ###### Decision tree with only categorical data

# ###### STEPS FOLLOWED:
# 1. Read csv file to panda dataframe
# 2. Apply one hot encoding to categorical features
# 3. Split data to 80-20 for training and validation using train_test_split()
# 4. Build Decision Tree using recursive function build_Tree()
# 5. Apply predict method to predict class label and use inbuilt functions to calculate confusion matrix, classification report and accuracy score.
# 6. Calculate the same measures using inbuilt scikitlearn decision tree to compare performance

# In[ ]:





# In[1]:


import numpy as np
from numpy import log2 as log
import pandas as pd
from sklearn import tree, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
eps = np.finfo(float).eps


# #### read data

# In[2]:


df = pd.read_csv('../input_data/train.csv')


# #### get class label into Y and drop it from it from df and assign to X

# In[3]:


Y = df.left
X = df.drop(['left','number_project','last_evaluation','satisfaction_level','average_montly_hours','time_spend_company'], axis=1)


# #### perform one hot encoding 

# In[4]:


Z = pd.concat([X,pd.get_dummies(X['sales'],prefix='sales')],axis=1)
Z = pd.concat([Z,pd.get_dummies(Z['salary'],prefix='salary')],axis=1)
Z = Z.drop(['sales','salary'],axis=1)


# #### split data into training(80%) and testing(20%)

# In[5]:


X_train, X_test, Y_train, Y_test = train_test_split(Z, Y,test_size=0.2)
df1 = pd.concat([X_train, Y_train],axis=1)


# #### Calculate entropy of class label

# In[7]:


def class_entropy(df):
    class_label = df.keys()[-1]
    class_entropy = 0
    values = df[class_label].unique()
    for val in values:
        q = float(df[class_label].value_counts()[val])/len(df[class_label])
        class_entropy += -q*log(q)
    return class_entropy


# #### calculate entropy of a feature

# In[8]:


def feature_entropy(df, feature):
    class_label = df.keys()[-1]
    target_variables = df[class_label].unique()
    variables = df[feature].unique()
    entropy = 0
    for var in variables:
        ent = 0
        for t in target_variables:
            n = len(df[feature][df[feature]==var][df[class_label]==t])
            d = len(df[feature][df[feature]==var])
            q = n/(d+eps)
            ent += -q*log(q+eps)
        q2 = float(d)/len(df)
        entropy += -q2*ent
    return abs(entropy)


# #### calculate IG of all features and select feature with maximum gain

# In[9]:


def feature_to_select(df):
    gain = []
    for key in df.keys()[:-1]:
        gain.append(class_entropy(df)-feature_entropy(df,key))
    return df.keys()[:-1][np.argmax(gain)]


# #### function to split data

# In[10]:


def subtable(df,node,value):
    return df[df[node]==value].reset_index(drop=True)


# #### node of decision tree

# In[11]:


class Node:
    def __init__(self,feature,positive=0,negative=0):
        self.feature=feature
        self.positive=positive
        self.negative=negative
        self.left=None
        self.right=None


# #### function that generates the tree

# In[12]:


def build_Tree(df):
    if len(df.columns)==1:
        return None
    
    split_node = feature_to_select(df)
    root = Node(split_node)
    root.positive=len(df[df['left']==1]['left'])
    root.negative=len(df[df['left']==0]['left'])
    
    subtable_left = subtable(df,split_node,0)
    subtable_right = subtable(df,split_node,1)
    
    subtable_left = subtable_left.drop(split_node,axis=1)
    subtable_right = subtable_right.drop(split_node,axis=1)
    
    clValue_left,counts_left = np.unique(subtable_left['left'],return_counts=True)
    clValue_right,counts_right = np.unique(subtable_right['left'],return_counts=True)
    
    if len(counts_left)>1:
        root.left=build_Tree(subtable_left)
    
    if len(counts_right)>1:
        root.right=build_Tree(subtable_right)
        
    return root


# In[13]:


root=build_Tree(df1)


# #### prediction function

# In[14]:


def rec_predict(df,root,prediction):
    if root==None:
        return None
    
    try:
        if root.right==None and root.left==None:
            prediction.append(1 if root.positive>root.negative else 0)
            return

        if root.right==None and df[root.feature]==1:
            prediction.append(1 if root.positive>root.negative else 0)
            return

        if root.left==None and df[root.feature]==0:
            prediction.append(1 if root.positive>root.negative else 0)
            return

        if df[root.feature]==0:
            rec_predict(df,root.left,prediction)
        else:
            rec_predict(df,root.right,prediction)
    except KeyError:
        if root.left!=Node:
            prediction.append(1 if root.positive>root.negative else 0)
            return
        else:
            rec_predict(df,root.left,prediction)
            
def predict(df,root,prediction):
    for col,row in df.iterrows():
        rec_predict(row,root,prediction)


# #### perform prediction with our model

# In[15]:


prediction = []
predict(X_test,root,prediction)

print confusion_matrix(Y_test,prediction)
print classification_report(Y_test,prediction)
print accuracy_score(Y_test,prediction)


# #### perform prediciton with inbuilt model

# In[16]:


model = tree.DecisionTreeClassifier()

model.fit(X_train, Y_train)
Y_predict = model.predict(X_test)
print confusion_matrix(Y_test,Y_predict)
print classification_report(Y_test,Y_predict)
print accuracy_score(Y_test,Y_predict)


# #### Testing from file

# In[17]:


ot = []
test_rows=pd.read_csv('../input_data/sample_test.csv')
test_rows1 = pd.concat([test_rows,pd.get_dummies(test_rows['sales'],prefix='sales')],axis=1)
test_rows1 = pd.concat([test_rows1,pd.get_dummies(test_rows1['salary'],prefix='salary')],axis=1)
test_rows1 = test_rows1.drop(['sales','salary'],axis=1)
# preprocess(test_rows1,split_values)
predict(test_rows1,root,ot)
print ot


# #### Observation
# To handle categorical features in a binary decision tree, they have to be encoded using one hot encoding or binary encoding or using other methods
# 
# Using only some features of the data, the result is poor

# In[ ]:




