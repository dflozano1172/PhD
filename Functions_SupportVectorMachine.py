# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:59:54 2020

@author: Lenovo
"""
import numpy as np
import pandas as pd
import time

from sklearn import svm
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

########################### CONSTRUCTION OF TRAIN AND VALIDATION SETS
X_train = X[:]; Y_train = Y[:]; X_valid=[]; Y_valid = []
np.random.seed(2)
for i in range(int(len(X) * por_valid)):
    idx = np.random.randint(0,len(X_train))
    X_valid.append(X_train[idx]); X_train = np.delete(X_train,idx,0)
    Y_valid.append(Y_train[idx]); Y_train = np.delete(Y_train,idx,0)
X_valid = np.asarray(X_valid)
Y_valid = np.asarray(Y_valid)
"""
########################### MODEL SET, TRAINING AND VALIDATION

def train_SVM(X_train, Y_train):
     ####### TRAINING 
    # Create an instance of Support Vector Machine Classifier and fit the data.
    # Parameters for the model are describes in:
    # https://scikit-learn.org/stable/modules/svm.html
    SVM_mod = svm.SVC(C=1e5, max_iter = 1000)
    SVM_clf = SVM_mod.fit(X_train, Y_train.ravel()) #classsifier
    print('This are the classes of the model');print(SVM_clf.classes_)
    #print('');print('This are the coefficients of the model');print(SVM_clf.coef_)
    return SVM_clf
 ####### PREDICTION AND VALIDATION 
print("")
def predict_SVM(SVM_clf, X):
    Y_pred = SVM_clf.predict(X)
    return Y_pred
"""
print(metrics.classification_report(Y_valid, Y_pred))
print('Confussion Matrix')
print(metrics.confusion_matrix(Y_valid, Y_pred, labels=range(len(SVM_clf.classes_))))
print(time.time() - t)
"""