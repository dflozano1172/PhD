import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

import Functions_LogisticRegression   as LR_fns 
import Functions_SupportVectorMachine as SVM_fns
import Functions_NeuralNetwork        as NN_fns

numerics    = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
Prp_valid   = 0.4 # Proportion of data set to be validation set
########################### SET X AND y 
data = pd.read_csv('FakeData.txt', sep=",", header=0)
data['gender'] = np.array([1 if g == "F" else 0 for g in data['gender']])

X_cols = ['gender', 'x_ray', 'confusion', 'comorbidity',
       'septic_shock', 'Temperature', 'resp_rate', 'Systolic_BP',
       'Diastolic_BP', 'BUN', 'Heart_Rate']
Y_col  = 'death'

X  = np.array(data[X_cols])
Y  = np.array(data[Y_col]).reshape(np.array(data[Y_col]).shape[0],1)


########################### CONSTRUCTION OF TRAIN AND VALIDATION SETS
def create_trainValid_Set(X,y,Prp_valid):
    X_train = X[:]; y_train = y[:]; X_valid=[]; y_valid = []
    np.random.seed(2)
    for i in range(int(len(X) * Prp_valid)):
        idx = np.random.randint(0,len(X_train))
        X_valid.append(X_train[idx]); X_train = np.delete(X_train,idx, axis = 0)
        y_valid.append(y_train[idx]); y_train = np.delete(y_train,idx, axis = 0)
    X_valid = np.asarray(X_valid)
    y_valid = np.asarray(y_valid)
    return X_train, y_train, X_valid, y_valid

X_train, Y_train, X_valid, Y_valid = create_trainValid_Set(X,Y,Prp_valid)

########################### LOGISTIC REGRESSION
Clsfr_lgR  = LR_fns.train_logReg(X_train, Y_train)
Y_Pred_lgR = LR_fns.predict_logReg(Clsfr_lgR, X_valid)


########################### SUPPORT VECTOR MACHINE - LINEAR KERNEL
Clsfr_SVM  = SVM_fns.train_SVM(X_train, Y_train)
Y_Pred_SVM = SVM_fns.predict_SVM(Clsfr_SVM, X_valid)

########################### NEURAL NETWORK

lr        = 0.015
epochs    = 2000
w,b,error = NN_fns.neural_network(X_train,Y_train,12, lr, epochs)
plt.plot(error)
plt.show()


a_l_valid = NN_fns.prediction_FeedForward(X_valid, w, b)
Y_Pred_NN = (a_l_valid[-1] > 0.5).astype(int)

########################### metrics
def metrics_Classifier(Y_real, Y_pred, labels):
    smry = metrics.classification_report(Y_real, Y_pred)
    cm   = metrics.confusion_matrix(Y_real, Y_pred, labels=range(len(labels)))
    return smry, cm

smry_lgR, cm_lgR = metrics_Classifier(Y_valid, Y_Pred_lgR, Clsfr_lgR.classes_)
smry_SVM, cm_SVM = metrics_Classifier(Y_valid, Y_Pred_lgR, Clsfr_SVM.classes_)
smry_NN , cm_NN  = metrics_Classifier(Y_valid, Y_Pred_NN, list(set([n[0] for n in Y.tolist()])))

print("confussion Matrix")
print(cm_lgR, cm_SVM, cm_NN)
print("Summary of validation set")
print(smry_lgR, smry_SVM, smry_NN)


