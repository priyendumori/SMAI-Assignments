#!/usr/bin/env python
# coding: utf-8

# # q-1-3
# ###### Contrast  the effectiveness of Misclassification  rate,  Gini,  Entropy as impurity measures

# In[1]:


import numpy as np
from numpy import log2 as log
import pandas as pd
from sklearn import tree, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
eps = np.finfo(float).eps


# #### load data

# In[2]:


df = pd.read_csv('../input_data/train.csv')


# #### get class label into Y and drop it from it from df and assign to X

# In[3]:


Y = df.left
X = df.drop(['left'], axis=1)


# #### perform one hot encoding 

# In[4]:


Z = pd.concat([X,pd.get_dummies(X['sales'],prefix='sales')],axis=1)
Z = pd.concat([Z,pd.get_dummies(Z['salary'],prefix='salary')],axis=1)
Z = Z.drop(['sales','salary'],axis=1)


# #### split data into training(80%) and testing(20%)

# In[5]:


X_train1, X_test1, Y_train1, Y_test1 = train_test_split(Z, Y,test_size=0.2)
df1 = pd.concat([X_train1, Y_train1],axis=1)
df2 = pd.concat([X_train1, Y_train1],axis=1)
df3 = pd.concat([X_train1, Y_train1],axis=1)


# #### Function takes dataframe and flag as argument.
# ```
# Flag: 1 - entropy
#       2 - gini
#       3 - misclassification rate
# Function takes argument and as per flag calculate measures for different type of impurities
# ```

# In[6]:


def func(df1,flag):
    
    def class_imp(df):
        if flag==1:
            class_label = df.keys()[-1]
            class_entropy = 0
            values = df[class_label].unique()
            for val in values:
                q = float(df[class_label].value_counts()[val])/len(df[class_label])
                class_entropy += -q*log(q)
            return class_entropy  
        elif flag==2:
            class_label = df.keys()[-1]
            class_gini = 1
            values = df[class_label].unique()
            for val in values:
                q = float(df[class_label].value_counts()[val])/len(df[class_label])
                class_gini *= q
            return class_gini*2
        elif flag==3:
            class_label = df.keys()[-1]
            class_mcr = 1
            values = df[class_label].unique()
            for val in values:
                q = float(df[class_label].value_counts()[val])/len(df[class_label])
                class_mcr = min(q,1-q)
            return class_mcr
    
    def feature_imp(df, feature):
        if flag==1:
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
        elif flag==2:
            class_label = df.keys()[-1]
            target_variables = df[class_label].unique()
            variables = df[feature].unique()
            gini = 0
            for var in variables:
                ent = 2
                for t in target_variables:
                    n = len(df[feature][df[feature]==var][df[class_label]==t])
                    d = len(df[feature][df[feature]==var])
                    q = n/(d+eps)
                    ent *= q
                q2 = float(d)/len(df)
                gini += q2*ent
            return abs(gini)
        elif flag==3:
            class_label = df.keys()[-1]
            target_variables = df[class_label].unique()
            variables = df[feature].unique()
            mcr = 0
            for var in variables:
                ent = 1
                for t in target_variables:
                    n = len(df[feature][df[feature]==var][df[class_label]==t])
                    d = len(df[feature][df[feature]==var])
                    q = n/(d+eps)
                    ent = min(q,1-q)
                q2 = float(d)/len(df)
                mcr += q2*ent
            return abs(mcr)

    def split_num_feature(data,cl_label,feature):
        max_ig = 0
        max_split = None
        pair = pd.concat([data,cl_label],axis=1)
        pair = pair.sort_values(by=feature).reset_index()
        found = set()
        for i in xrange(len(data)-1):
            if pair['left'][i]!=pair['left'][i+1] and (float(pair[feature][i] + pair[feature][i+1])/2) not in found:
                found.add(float(pair[feature][i] + pair[feature][i+1])/2)
                ig = compute_IG(pair,float(pair[feature][i] + pair[feature][i+1])/2, feature)
                if ig > max_ig:
                    max_ig = ig
                    max_split = float(pair[feature][i] + pair[feature][i+1])/2
        return max_split

    numerical_attributes=['number_project','last_evaluation', 'satisfaction_level','average_montly_hours','time_spend_company']
    split_values={}

    def feature_to_select(df):
        num_attr = [i for i in df.columns if i in numerical_attributes]
        for at in num_attr:
            split = split_num_feature(df[at],df['left'],at)
            split_values[at]=split

        entropy_attr = []
        gain = []
        for key in df.keys()[:-1]:
            gain.append(class_imp(df)-feature_imp(df,key))
        return df.keys()[:-1][np.argmax(gain)]

    def subtable(df,node,value):
        return df[df[node]==value].reset_index(drop=True)

    def compute_IG(df,val,feature):
        cl_imp = class_imp(df)
        l = df[df[feature]<val].reset_index(drop=True)
        r = df[df[feature]>=val].reset_index(drop=True)
        l_imp = class_imp(l)
        r_imp = class_imp(r)
        return cl_imp - ( (float(len(l))/(len(df)+eps)*l_imp) + (float(len(r))/(len(df)+eps)*r_imp) )  

    
    class Node:
        def __init__(self,feature,positive=0,negative=0):
            self.feature=feature
            self.split_pos=0
            self.positive=positive
            self.negative=negative
            self.left=None
            self.right=None

    def build_Tree(df):
        if len(df.columns)==1:
            return None

        split_node = feature_to_select(df)
        root = Node(split_node)
        if split_node in numerical_attributes:
            split_point = split_values[root.feature]
            root.split_pos = split_point

            root.positive=len(df[df['left']>=split_point]['left'])
            root.negative=len(df[df['left']<split_point]['left'])

            subtable_left = df[df[split_node]<split_point].reset_index(drop=True)
            subtable_right = df[df[split_node]>=split_point].reset_index(drop=True)

        else:
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

    root=build_Tree(df1)

    
    def rec_predict(df,root,prediction):
        if root==None:
            return None

        if root.feature in numerical_attributes:
            try:
                if root.right==None and root.left==None:
                    prediction.append(1 if root.positive>root.negative else 0)
                    return

                if root.right==None and df[root.feature]>=root.split_pos:
                    prediction.append(1 if root.positive>root.negative else 0)
                    return

                if root.left==None and df[root.feature]<root.split_pos:
                    prediction.append(1 if root.positive>root.negative else 0)
                    return

                if df[root.feature]<root.split_pos:
                    rec_predict(df,root.left,prediction)
                else:
                    rec_predict(df,root.right,prediction)
            except KeyError:
                if root.left==None:
                    prediction.append(1 if root.positive>root.negative else 0)
                    return
                rec_predict(df,root.left,prediction)
        else:
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
                if root.left==None:
                    prediction.append(1 if root.positive>root.negative else 0)
                    return
                rec_predict(df,root.left,prediction)

    def predict(df,root,prediction):
        for col,row in df.iterrows():
            rec_predict(row,root,prediction)

            
    pd.options.mode.chained_assignment = None
    X1_test = X_test1.copy(deep=True)

    prediction = []
    predict(X1_test,root,prediction)
    print confusion_matrix(Y_test1,prediction)
    print classification_report(Y_test1,prediction)
    print accuracy_score(Y_test1,prediction)


# #### Call function with entropy as impurity measure

# In[7]:


func(df1,1)


# #### Call function with gini as impurity measure

# In[8]:


func(df2,2)


# #### Call function with misclassification rate as impurity measure

# In[9]:


func(df3,3)


# #### Observations
# Impurity measures impacts the performance a lot
# 
# Performance of gini and entropy is same for most cases. 
# Gini is easier to compute compared to entropy.
# 
# Misclassification rate is the worst performer among the three
