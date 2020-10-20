
# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
"""
t = time.time()
########################### PARAMETERS
# This dataset was extracted from python main library datasets.load_iris()
File_name   = 'C:/Users/Lenovo/Documents/01. University of Leicester/02. PhD/06 Python_Templates/data.csv'
Output_cols = ['Outcome']
numerics    = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
por_valid   = 0.4

########################### DATA PREPARATION 
df_data    = pd.read_csv(File_name, header = 0)
Y          = np.array(df_data[Output_cols]).reshape(len(df_data),len(Output_cols))
X          = np.array(df_data.drop(Output_cols,axis=1).select_dtypes(include=numerics))

########################### CONSTRUCTION OF RAIN AND VALIDATION SETS
X_train = X[:]; Y_train = Y[:]; X_valid=[]; Y_valid = []
np.random.seed(2)
for i in range(int(len(X) * por_valid)):
    idx = np.random.randint(0,len(X_train))
    X_valid.append(X_train[idx]); X_train = np.delete(X_train,idx,0)
    Y_valid.append(Y_train[idx]); Y_train = np.delete(Y_train,idx,0)
X_valid = np.asarray(X_valid)
Y_valid = np.asarray(Y_valid)
"""
def initialize_weights(n_weights):
  a = np.random.randn(n_weights, 1)
  b = np.random.randn()
  return a, b

def sigmoid(x):
  sig = 1/(1+ np.exp(x))
  return sig

def predictions(a,b,X):
  z = sigmoid(-(np.dot(X,a) + b))
  return z

def find_cost(y_hat,y):
  m = y.shape[0]
  total_cost = (1/m) * np.sum((y_hat-y)**2)
  return total_cost

def  Update_weights(a,b,X,y,y_hat,lr):
  m = y.shape[0]
  n = a.shape[0]
  a_new = np.zeros(a.shape) 
  b_new = b - lr * (1/m) * np.sum((y_hat - y))
  for i in range(n):
    a_new[i] = a[i] - lr * (1/m) * np.dot((y_hat- y).T, X[:,i])
  
  return a_new, b_new

def logistic_regression(X,y,lr, epochs):
  error_list = []
  n_weights = X.shape[1]  
  a, b      = initialize_weights(n_weights)

  for i in range(epochs):
    y_hat = predictions(a, b, X)
    cost  = find_cost(y_hat,y)
    error_list.append(cost)
    a, b  = Update_weights(a,b,X,y,y_hat,lr)
    if i % (epochs / 10) == 0:
      print(cost)
  return a,b,error_list
########################### MODEL SET, TRAINING AND VALIDATION
def train_logReg(X_train, Y_train):
     ####### TRAINING 
    # Create an instance of Logistic Regression Classifier and fit the data.
    # Parameters for the model are describes in:
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.fit# 
    logreg     = linear_model.LogisticRegression(C=1e5, max_iter = 1000)
    logreg_clf = logreg.fit(X_train, Y_train.ravel()) #classsifier
    #print('This are the classes of the model');print(logreg_clf.classes_)
    #print('');print('This are the coefficients of the model');print(logreg_clf.coef_)
    return logreg_clf
 
 ####### PREDICTION AND VALIDATION 
def predict_logReg(logreg_clf, X):
    Y_pred = logreg_clf.predict(X)
    return Y_pred

def metrics_logReg(Y_real, Y_pred, logreg_clf):
    smry = metrics.classification_report(Y_real, Y_pred)
    cm   = metrics.confusion_matrix(Y_real, Y_pred, labels=range(len(logreg_clf.classes_)))
    return smry, cm
    
    #print(time.time() - t)
