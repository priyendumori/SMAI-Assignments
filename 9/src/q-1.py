# -*- coding: utf-8 -*-
"""q-1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AEbsHC9nEzw0bj_Vqi_R32VL6Vd382Hu

#### Q1 - Train and validate an n-layer Neural Network on apparel dataset to predict the class label of a given apparel.
"""

import pandas as pd
import numpy as np

from scipy.spatial import distance
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances

from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import io
eps = np.finfo(float).eps

from google.colab import drive
drive.mount('/content/gdrive')

"""#### read data"""

data = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/data.csv')

"""#### helper functions"""

def act_fun(function_name, delta,x):
    if function_name == "sigmoid":
        if delta==True:
            return delta_sigmoid(x)
        else:
            return sigmoid(x)
        
    elif function_name == "linear":
        if delta==True:
            return delta_linear(x)
        else:
            return linear(x)
        
def linear(x):
    return 0.1*x
  
def delta_linear(x):
#     print x.shape
    a = np.empty(x.shape)
    a.fill(0.1)
#     return np.ones(x.shape)
    return a

def sigmoid(x):
    x=-x
    return 1.0 / (1.0 + np.exp(x))
  
def delta_sigmoid(x):
    sig = sigmoid(x)
    del_sig = sig * (1.0 - sig)
    return del_sig
    
def delta_mean_square_error(y1, y):
    return (y1 - y)
  
def cross_entropy(predictions, targets, epsilon = 1e-12):
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions + 1e-9))/N
    return ce

def model_acc(model):
    model.feed_forward(data)
    predicted_y = model.layer_output[-1]
    print np.all(np.isfinite(predicted_y)), np.all(np.isfinite(data)) 
    return np.linalg.norm(data - predicted_y)
#     return distance.euclidean(data, predicted_y)

Y = data.xAttack
data = data.drop('xAttack',axis=1)
data = StandardScaler().fit_transform(data)
# Y = data_train.iloc[:,0]

# unique_labels = np.unique(Y).tolist() 
# Y = one_hot_encode(Y, unique_labels)

"""#### Neural Network class"""

class NeuralNetwork:
    def __init__(self,layers, neurons, features, batch_size, epochs, function_name, alpha=0.00001):
        
        self.weights = np.empty(layers + 1, dtype = object)
        for i in range(len(self.weights)):
            if i == layers:
                self.weights[i] = self.get_random_weights(neurons[i-1],features)
            elif i == 0:
                self.weights[i] = self.get_random_weights(features,neurons[i])
            else:
                self.weights[i] = self.get_random_weights(neurons[i-1],neurons[i])
        
        
        self.alpha = alpha
        self.layers = layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.function_name = function_name
        
    
    def get_random_weights(self,a,b):
        return np.random.uniform(-np.sqrt(2.0 / a),np.sqrt(2.0 / a),(a, b))
        
    def update_weights(self,X,dw):    
        for i in range(layers + 1):
            if i == 0:
                dot = np.dot(X.T, dw[i])
                decrement = self.alpha * dot
                self.weights[i] -= decrement
            else:
                dot = np.dot(self.layer_output[i-1].T, dw[i])
                decrement = self.alpha * dot
                self.weights[i] -= decrement
    
    def back_prop(self,X, y):
        delta_weights = np.empty(layers + 1, dtype = object)
        for i in range(len(self.weights) - 1, -1, -1): 
            if i == self.layers:
                term1 = delta_mean_square_error(self.layer_output[-1], y)
                term2 = act_fun(self.function_name, True, self.layer_input[-1])
                delta_weights[i] = np.multiply(term1, term2)
            else: 
                term1 = np.dot(delta_weights[i+1], self.weights[i+1].T)
                term2 = act_fun(self.function_name,True,self.layer_input[i])
                delta_weights[i] = np.multiply(term1, term2)
                
        self.update_weights(X,delta_weights)    
        
    
    def feed_forward_encode(self,X):
        k = (self.layers+2)/2
        self.layer_input = np.empty(layers + 1, dtype = object)
        self.layer_output = np.empty(layers + 1, dtype = object)
        for i in range(k+1):
            if i == 0:
                self.layer_input[i] = np.dot(X, self.weights[i])
                self.layer_output[i] = act_fun(self.function_name, False, self.layer_input[i])
            else:
                self.layer_input[i] = np.dot(self.layer_output[i - 1], self.weights[i])
                self.layer_output[i] = act_fun(self.function_name, False, self.layer_input[i])
    
    def feed_forward(self,X):
        self.layer_input = np.empty(layers + 1, dtype = object)
        self.layer_output = np.empty(layers + 1, dtype = object)
        for i in range(len(self.weights)):
            if i == self.layers:
                self.layer_input[i] = np.dot(self.layer_output[i - 1], self.weights[i])
                self.layer_output[i] = act_fun(self.function_name,False,self.layer_input[i])
            elif i == 0:
                self.layer_input[i] = np.dot(X, self.weights[i])
                self.layer_output[i] = act_fun(self.function_name, False, self.layer_input[i])
            else:
                self.layer_input[i] = np.dot(self.layer_output[i - 1], self.weights[i])
                self.layer_output[i] = act_fun(self.function_name, False, self.layer_input[i])
        
    def batch_process(self,index,X,Y):
        start = index
        end = index + self.batch_size
        X_batch, Y_batch = X[start:end], Y[start:end]
        self.feed_forward(X_batch)
        self.back_prop(X_batch, Y_batch)
        ce = cross_entropy(self.layer_output[-1], Y_batch)
        return ce
    
    def train(self,X, y):
        for i in range(self.epochs):
            cost = 0
            for j in range(0, X.shape[0], self.batch_size):
                cost += self.batch_process(j,X,y)
                
            if i%50 == 0:    
                print str(i)+"   "+str(cost / self.batch_size)
            
    def test(self,test_x):
        k = (self.layers+2)/2
        self.feed_forward_encode(test_x)
        y_ = self.layer_output[k-1]
        return y_

"""#### calling neural network with all 3 activation functions"""

neurons = [14]
layers = len(neurons) 
batch_size = 500
epochs = 500
features = 29

# running for linear
model_l3 = NeuralNetwork(layers, neurons, features, batch_size, epochs, "linear", 0.001)
model_l3.train(data, data)
print model_acc(model_l3)

data_linear_3 = model_l3.test(data)
data_linear_3 = pd.DataFrame( data_linear_3, columns=[ "new_dim"+str(i) for i in xrange(data_linear_3.shape[1]) ] )
print data_linear_3.head()

neurons = [14]
layers = len(neurons) 
batch_size = 500
epochs = 500
features = 29

# running for sigmoid
model_nl3 = NeuralNetwork(layers, neurons, features, batch_size, epochs, "sigmoid", 0.001)
model_nl3.train(data, data)
print model_acc(model_nl3)

data_non_linear_3 = model_nl3.test(data)
data_non_linear_3 = pd.DataFrame( data_non_linear_3, columns=[ "new_dim"+str(i) for i in xrange(data_non_linear_3.shape[1]) ] )
print data_non_linear_3.head()

neurons = [20,14,20]
layers = len(neurons) 
batch_size = 500
epochs = 500
features = 29

# running for sigmoid and more layers
model_nl5 = NeuralNetwork(layers, neurons, features, batch_size, epochs, "sigmoid", 0.5)
model_nl5.train(data, data)
print model_acc(model_nl5)

data_non_linear_5 = model_nl5.test(data)
print data_non_linear_5.shape
data_non_linear_5 = pd.DataFrame( data_non_linear_5, columns=[ "new_dim"+str(i) for i in xrange(data_non_linear_5.shape[1]) ] )
print data_non_linear_5.head()



def plot_func(purity_dict):
    val = []
    label = []
    for k,v in purity_dict.items():
        val.append(v)
        label.append(k)

    print val,
    print label

    plt.pie( val, labels = label)
    plt.show()

def plot_all():
    print "K means : "
    plot_func(purity_dict_k_means)
    print "GMM : "
    plot_func(purity_dict_gmm)
    print "AGM : "
    plot_func(purity_dict_agm)

def purity(df,pred_Y,K):
    pred_Y = pd.DataFrame( pred_Y,columns=[ 'pred_Y' ] )
    df = pd.concat([df, pred_Y],axis=1)
    purity_dict = {}
    for i in xrange(K):
        sub_table = df[ df['pred_Y'] == i ]
        label, count = np.unique(sub_table['xAttack'],return_counts=True)
        mx_ind = np.argmax(count)
        purity_dict[i] = count[mx_ind] / float(len(sub_table))
    return purity_dict

"""using compressed data from 3 layer linear activation autoencoder"""

kmeans = KMeans(n_clusters=5, random_state=0).fit(data_linear_3)
cluster_k_means = kmeans.labels_

gmm = GaussianMixture(n_components=5).fit(data_linear_3)
cluster_gmm =  gmm.predict(data_linear_3)

agm = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='single').fit(data_linear_3)
cluster_agm = agm.labels_

data_linear_3 = pd.concat([data_linear_3, Y], axis=1)

purity_dict_k_means = purity(data_linear_3,cluster_k_means,5)
purity_dict_gmm = purity(data_linear_3, cluster_gmm ,5)
purity_dict_agm = purity(data_linear_3, cluster_agm ,5)

print "Dimensionality reduced by autoencoder with linear activation and 3 layers"
print
print "k means : ",purity_dict_k_means
print "gmm : ",purity_dict_gmm
print "agm : ",purity_dict_agm

print "linear activation and 1 layer"
plot_all()



"""using compressed data from 3 layer non linear activation autoencoder"""

kmeans = KMeans(n_clusters=5, random_state=0).fit(data_non_linear_3)
cluster_k_means = kmeans.labels_

gmm = GaussianMixture(n_components=5, n_init=10 ).fit(data_non_linear_3)
cluster_gmm =  gmm.predict(data_non_linear_3)

agm = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='single').fit(data_non_linear_3)
cluster_agm = agm.labels_

data_non_linear_3 = pd.concat([data_non_linear_3, Y], axis=1)

purity_dict_k_means = purity(data_non_linear_3,cluster_k_means,5)
purity_dict_gmm = purity(data_non_linear_3, cluster_gmm ,5)
purity_dict_agm = purity(data_non_linear_3, cluster_agm ,5)

print "Dimensionality reduced by autoencoder with non linear activation and 3 layers"
print
print "k means : ",purity_dict_k_means
print "gmm : ",purity_dict_gmm
print "agm : ",purity_dict_agm

print "non linear activation and 1 layer"
plot_all()



"""using compressed data from 5 layer non linear activation autoencoder"""

kmeans = KMeans(n_clusters=5, random_state=0).fit(data_non_linear_5)
cluster_k_means = kmeans.labels_

gmm = GaussianMixture(n_components=5, n_init=10 ).fit(data_non_linear_5)
cluster_gmm =  gmm.predict(data_non_linear_5)

agm = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='single').fit(data_non_linear_5)
cluster_agm = agm.labels_

data_non_linear_5 = pd.concat([data_non_linear_5, Y], axis=1)

purity_dict_k_means = purity(data_non_linear_5,cluster_k_means,5)
purity_dict_gmm = purity(data_non_linear_5, cluster_gmm ,5)
purity_dict_agm = purity(data_non_linear_5, cluster_agm ,5)

print "Dimensionality reduced by autoencoder with non linear activation and 3 layers"
print
print "k means : ",purity_dict_k_means
print "gmm : ",purity_dict_gmm
print "agm : ",purity_dict_agm

print "non linear activation and 3 layer"
plot_all()

