%reset -f
# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import time, os, sqlite3, pickle
from matplotlib import pyplot as plt
# Oteher ML libraries
from sklearn.model_selection import StratifiedKFold
# Machine Learning Liraries
from sklearn import linear_model
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# Deep Learning Libraries
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
##################### LOAD THE DATA ############################################
z_save_list = dir()

URI_SQLITE_DB = "test.db"
path = os.getcwd()
data_imputed = pickle.load(open(path + "\data_imputed.pickle","rb"))
conn = sqlite3.connect(path + "\\test1.db")  #Mandatory line
c = conn.cursor()
c.execute("DROP TABLE IF EXISTS test1")
conn.commit()
data_imputed.to_sql('test1', con = conn)
conn.commit()
c.execute("SELECT * FROM test1")
    
df = pd.read_sql("SELECT * FROM test1", con=conn).drop(columns = ['index'])
conn.close()

##################DELETE ALL VARIABLES BUT DATA################################
z_save_list.append('df')
for z_name in dir():
    if z_name not in z_save_list:
        del globals()[z_name]
        
############ SET DATA IN KERAS FORMAT #########################################
t = time.time()

patient_ID_col = 'PATIENT_VISIT_IDENTIFIER'
time_ID_col    = 'Time_Val'
target_col     = 'ICU'
time_stamps    = df[time_ID_col].unique().tolist()

patients = df[patient_ID_col].unique().tolist()
time_stamp = df[time_ID_col].unique().tolist()

#df_copy = df.drop(columns = [target_col, time_ID_col])

Y = np.array(df[(df[time_ID_col] == np.max(time_stamps))][target_col])
X = np.zeros((len(patients),len(time_stamps),len(df.drop(columns = [target_col, time_ID_col,patient_ID_col]).columns.tolist())))
for i, pat in enumerate(patients): X[i,:,:] = df[df[patient_ID_col] == pat].drop(columns = [target_col, time_ID_col,patient_ID_col])


##################### 1  Split the data set in Train_validatyion and test sets
test_portion = 0.2
seed = np.random.seed(0)
train_rows_n = int(np.around( (1 - test_portion) * len(X), decimals = 0))
X_shuffle   = np.random.shuffle(X)
x_train_val = np.array(X)[:train_rows_n,:,:]
x_test      = np.array(X)[train_rows_n:,:,:]
y_train_val = Y[:train_rows_n]
y_test      = Y[train_rows_n:]
################ DENSE NEURAL NETWORK #########################################

model = Sequential()

model.add(layers.Dense(16, activation = 'relu', input_shape = (40,)))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])


kfold = StratifiedKFold(n_splits = 4, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(x_train_val, y_train_val):
    trainX, valX = x_train_val[train], x_train_val[test]
    trainY, valY = y_train_val[train], y_train_val[test]
    
    
    history = model.fit(trainX.reshape(trainX.shape[0],40), trainY, 
              epochs=20, 
              verbose = 'silent'
              )
    results = model.evaluate(valX.reshape(valX.shape[0],40), valY)
    cvscores.append(history)
    




