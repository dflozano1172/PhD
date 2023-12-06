# ============================================================================
##############################################################################
##############################################################################
# IMPORT ALL TIME PACKAGES
##############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pickle, time, random, sys, warnings
# sys.path is a list of absolute path strings
sys.path.append('/home/d/dlr10/Documents/02_Statitics_modelling/0_FunctionsScripts')
import Loading_Data_Functions as load_fn
import FineTuning_Functions as FineTuning
import Learning_Curves_Functions as LearningCurves
##############################################################################
# IMPORT DEEP LEARNING PACKAGES
##############################################################################

from sklearn.model_selection import cross_val_score, StratifiedKFold

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten, LeakyReLU
from tensorflow.keras import layers

from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from tensorflow.keras.utils import to_categorical

from sklearn.metrics import make_scorer, accuracy_score
LeakyReLU = LeakyReLU(alpha=0.1)

from bayes_opt import BayesianOptimization

import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)

print('***********************************************************************')
print('***********************************************************************')
print('***********************************************************************')
print('******************** STARTING GRU EWS BALANCED ***********************')
print('***********************************************************************')
print('***********************************************************************')
print('***********************************************************************')
##############################################################################
# ENCODING FUNCTIONS
##############################################################################

# ENCODING CATEGORICAL VARIABLES WITH TARGET ENCODER OR NORMALISE CONTINUOUS VARIABLES
# https://towardsdatascience.com/target-encoding-and-bayesian-target-encoding-5c6a6c58ae8c
def target_encoder_Binary(column, target, X_data):
    df = []
    for cat in X_data[column].unique():
        row = []
        row.append(len(X_data[(X_data[target]==0) & (X_data[column] == cat)]))
        row.append(len(X_data[(X_data[target]==1) & (X_data[column] == cat)]))
        df.append(row)
    df = pd.DataFrame(df, index = X_data[column].unique(), columns = ['0','1'])
    df['prob'] = df['1']/(df['1']+df['0'])
    col_encod = X_data[column].map(dict(zip(df.index, df['prob'])))
    return col_encod, df
############################################################
def Normalise_n_encode_train_set(X_data, feat_list, data_types):
    encoder = []
    X_data_norm_2 = pd.DataFrame()
    for feat in feat_list:
        if data_types[feat] == 'Continuous':
            X_data_feat = X_data[feat]
            mean        = X_data_feat.mean()
            std         = X_data_feat.std()
            X_data_norm_2[feat] = (X_data_feat - mean)/std
            encoder.append([feat, data_types[feat], [mean, std]])
        elif data_types[feat] == 'Categorical':
            X_data_norm_2[feat],df = target_encoder_Binary(feat, 'Mortality', X_data)
            encoder.append([feat, data_types[feat], df])
        elif data_types[feat] == 'Binary':
            X_data_norm_2[feat] = X_data[feat].copy()
            encoder.append([feat, data_types[feat], ""])
    encoder =pd.DataFrame(encoder, columns = ['feature','type','parameters'])
    return X_data_norm_2,encoder
############################################################
def Normalise_n_encode_val_set(val_data, norm_encoder, feat_list, data_types):
    X_data_norm_2 = pd.DataFrame()
    for feat in feat_list:
        if data_types[feat] == 'Continuous':
            X_data_feat = val_data[feat]
            mean, std   = norm_encoder[norm_encoder['feature'] == feat].iloc[0]['parameters']
            X_data_norm_2[feat] = (X_data_feat - mean)/std

        elif data_types[feat] == 'Categorical':
            df = norm_encoder[norm_encoder['feature'] == feat].iloc[0]['parameters']
            X_data_norm_2[feat] = val_data[feat].map(dict(zip(df.index, df['prob'])))
            X_data_norm_2[feat] = X_data_norm_2[feat].fillna(0)
        elif data_types[feat] == 'Binary':
            X_data_norm_2[feat] = val_data[feat].copy()
    return X_data_norm_2
##############################################################################
# PLOTTING FUNCTIONS 
##############################################################################
def history_plots_(history, SAVE = 0, name = ''):
    auc_keys = [x for x in history.history.keys() if 'auc' in x]
    loss = history.history["loss"];     val_loss = history.history["val_loss"]
    acc  = history.history["accuracy"]; val_acc  = history.history["val_accuracy"]
    auc  = history.history[auc_keys[0]];      val_auc = history.history[auc_keys[1]]
    epochs  = range(1, len(acc) + 1)
    f, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize = (8,4))
    ax0.plot(epochs, loss, "bo", label="Training Loss")
    ax0.plot(epochs, val_loss, "b", label="Validation Loss")
    ax0.set_title("Train-val Loss")
    ax1.plot(epochs, acc, "bo", label="Training Accuracy")
    ax1.plot(epochs, val_acc, "b", label="Validation Accuracy")
    ax1.set_title("Train-val Accuracy")
    ax2.plot(epochs, auc, "bo", label="Training AUC")
    ax2.plot(epochs, val_auc, "b", label="Validation AUC")
    ax2.set_title("Train-val AUC")
    plt.legend()
    if SAVE == 1: plt.savefig(name, transparent = True, bbox_inches = "tight")
    plt.show()
# ============================================================================

t_tot = time.time()
# ##############
# 1. LOAD DATA ---------------------------------------------------------------
# ============================================================================
t = time.time()
path = r'/home/d/dlr10/Documents/02_Statitics_modelling/DataSets/'
df_patients_16_18, df_admissions_16_18, _ = load_fn.Load_data()
df_patients_19_20, df_admissions_19_20, _ = load_fn.Load_data('2019_2020')

X_data_16_18 = pickle.load(open(path + 'df_ts_2016_18_los_1d.pickle','rb'))
X_data_19_20 = pickle.load(open(path + 'df_ts_2019_20_los_1d.pickle','rb'))

# List of features ------------------------
feat_list = X_data_16_18[0].columns.tolist()
feat_list = feat_list[1:-1]

# Dictionary of features and types --------
data_types = pd.read_csv('/home/d/dlr10/Documents/02_Statitics_modelling/2_Statistics/csv_Type_variables.csv')
data_types = data_types.set_index('Variable').to_dict()['Type']

# Resulst vaiables ------------------------
df_results_train = pd.DataFrame()
df_results_valid = pd.DataFrame()
dict_fpr_tpr = {}

print("Elapsed time loading data:", time.time()-t)
# ============================================================================


# ##############
# 2. PROCESSING DATA ---------------------------------------------------------
# ============================================================================
t = time.time()

### SPLIT DATA #######################################
train_set = X_data_16_18[0].reset_index().copy()
valid_set = X_data_19_20[0].reset_index().copy()

### ENCODING DATA ####################################
train_set_norm, encoder = Normalise_n_encode_train_set(train_set, feat_list, data_types)
valid_set_norm          = Normalise_n_encode_val_set(valid_set, encoder, feat_list, data_types)

### SET DATA AS ARRAYs ###############################
num_samp_train = len(train_set['admission_id'].unique().tolist())
num_samp_valid = len(valid_set['admission_id'].unique().tolist())
num_features   = len(train_set_norm.columns.tolist())
num_time_samp  = len(train_set[train_set['admission_id']==train_set['admission_id'].tolist()[0]])
X_train    = np.array(train_set_norm).reshape((num_samp_train, num_time_samp, num_features ))
X_valid    = np.array(valid_set_norm).reshape((num_samp_valid, num_time_samp, num_features ))
df         = train_set[['admission_id', 'Mortality']].groupby(by= 'admission_id').mean()

y_train = list(list(zip(*X_data_16_18[1]))[1])
#dict_classes = {'3d':0, '4d':1,'5d':2,'6d':3, '1w':4,'2w':5,'3w':6, '4w':7}
dict_classes = {'3-4d':0, '5-6d':1, '7d':2, '8-9d':3,'10-13d':4, '2w':5, '3w':6, '4w':7}
los_codes    = lambda x: dict_classes[x] 
y_train      = np.array([los_codes(x) for x in y_train])

y_valid  = list(list(zip(*X_data_19_20[1]))[1])
y_valid  = np.array([los_codes(x) for x in y_valid])

### ONE-HOT ENCODING Y ###############################
y_train_1 = to_categorical(y_train, num_classes = 8)
y_valid_1 = to_categorical(y_valid, num_classes = 8)


print("Elapsed time processing data:", time.time()-t)
# ============================================================================
print("Shape of X_train",X_train.shape)
print("Shape of y_train",y_train_1.shape)

# ##############
# 3. BAYESIAN OPTIMIZER TO FINE TUNE THE NEURAL NETWORK
# ============================================================================

def nn_cl_bo2(cellsIni, cells1, cells2, cells3, activation1, activation2, activation3, 
              optimizer, learning_rate, batch_size, epochs, layers1, layers2, 
              dropout1, dropout_rate1, dropout2, dropout_rate2):
    optimizerD = {'Adam':Adam(lr=learning_rate), 'SGD':SGD(lr=learning_rate),
                 'RMSprop':RMSprop(lr=learning_rate), 'Adadelta':Adadelta(lr=learning_rate),
                 'Adagrad':Adagrad(lr=learning_rate), 'Adamax':Adamax(lr=learning_rate),
                 'Nadam':Nadam(lr=learning_rate), 'Ftrl':Ftrl(lr=learning_rate)}
    activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
                   'elu', 'exponential', LeakyReLU]
    
    cellsIni = round(cellsIni)
    cells1 = round(cells1)
    cells2 = round(cells2)
    cells3 = round(cells3)
    activation1 = activationL[round(activation1)]
    activation2 = activationL[round(activation2)]
    activation3 = activationL[round(activation3)]
    optimizer  = list(optimizerD.keys())[round(optimizer)]
    batch_size = round(batch_size)
    epochs     = round(epochs)
    layers1    = round(layers1)
    layers2    = round(layers2)
    
    score_acc = make_scorer(accuracy_score)
    
    def nn_cl_fun():      
        nn = Sequential()
        nn.add(layers.LSTM(cellsIni, activation=activation1, return_sequences = True, input_shape=(48,32)))
        
        for i in range(layers1):
            nn.add(layers.LSTM(cells1, recurrent_activation=activation1, return_sequences = True))
            if dropout1 > 0.5:
                nn.add(layers.SpatialDropout1D(dropout_rate1, seed=123))
        
        for i in range(layers2):
            nn.add(layers.LSTM(cells2, recurrent_activation=activation2, return_sequences = True))
            if dropout2 > 0.5:
                nn.add(layers.SpatialDropout1D(dropout_rate2, seed=123))       
        
        nn.add(layers.LSTM(cells3, activation=activation3))
        nn.add(layers.Dense(10, activation='relu'))
        nn.add(layers.Dense(8, activation='softmax'))
        nn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        # model = Sequential()

        # model.add(layers.LSTM(25, activation=activation1, input_shape=(48, 32),return_sequences = True))
        # model.add(layers.LSTM(15, activation=activation1))
        # model.add(layers.Dense(20, activation='softmax'))
        # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        return nn #model
    
    es    = EarlyStopping(monitor='accuracy', mode='max', verbose=0, patience=20)
    nn    = KerasClassifier(build_fn=nn_cl_fun, epochs=epochs, batch_size=batch_size, verbose=0)
    #kfold = StratifiedKFold(n_splits=8, shuffle=True, random_state=123)
    score = cross_val_score(nn, X_train, y_train_1, fit_params={'callbacks':[es]}).mean()
    return score


params_nn2 ={'cellsIni':(10,40), 'cells1':(0,40)     , 'cells2':(0, 40)         , 'cells3':(0,40)  ,
             'activation1':(0, 8), 'activation2':(0, 8)     , 'activation3':(0, 8),
             'dropout1':(0,1)    , 'dropout_rate1':(0.1,0.5),
             'dropout2':(0,1)    , 'dropout_rate2':(0.1,0.5),
             'layers1': (1,2)    , 'layers2': (1,3),
             'optimizer':(0,7),
             'learning_rate':(0.01, 0.3),
             'batch_size':(200, 1000),
             'epochs':(20, 100)}

# Run Bayesian Optimization
t = time.time()
nn_bo = BayesianOptimization(nn_cl_bo2, params_nn2, random_state=111)
nn_bo.maximize(init_points=20, n_iter=20)
print("elapsed Fine tuning the NN", time.time()-t)
# ============================================================================


# ##############
# 4. EXTRACTING RESULTS FROM FINE-TUNING
# ============================================================================
params_nn_ = nn_bo.max['params']
    
activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu','elu', 'exponential',
               LeakyReLU,'relu']
params_nn_['layers1']    = round(params_nn_['layers1'])
params_nn_['layers2']    = round(params_nn_['layers2'])

params_nn_['cellsIni']   = round(params_nn_['cellsIni'])
params_nn_['cells1']     = round(params_nn_['cells1'])
params_nn_['cells2']     = round(params_nn_['cells2'])
params_nn_['cells3']     = round(params_nn_['cells3'])

params_nn_['activation1'] = activationL[round(params_nn_['activation1'])]
params_nn_['activation2'] = activationL[round(params_nn_['activation2'])]
params_nn_['activation3'] = activationL[round(params_nn_['activation3'])]
params_nn_['batch_size']  = round(params_nn_['batch_size'])
params_nn_['epochs']      = round(params_nn_['epochs'])
params_nn_['dropout1']    = round(params_nn_['dropout1'])
params_nn_['dropout2']    = round(params_nn_['dropout2'])

params_nn_['optimizer_no']= round(params_nn_['optimizer'])
# ============================================================================


# ##############
# 5. FIT CLASSIFIER WITH FINE TUNE GHYPERPARAMETERS
# ============================================================================

layers1       = params_nn_['layers1']
layers2       = params_nn_['layers2']
activation1   = params_nn_['activation1']
activation2   = params_nn_['activation2']
activation3   = params_nn_['activation3']
cellsIni      = params_nn_['cellsIni']
cells1        = params_nn_['cells1']
cells2        = params_nn_['cells2']
cells3        = params_nn_['cells3']
batch_size    = params_nn_['batch_size']
epochs        = params_nn_['epochs']
optimizer_no  = params_nn_['optimizer_no']
dropout_rate1 = params_nn_['dropout_rate1']
dropout_rate2 = params_nn_['dropout_rate2']
dropout1      = params_nn_['dropout1']
dropout2      = params_nn_['dropout2']
learning_rate = params_nn_['learning_rate']

def ft_1dcnn_cl_fun(cellsIni,  cells1, cells2, cells3, activation1, activation2, activation3, 
                    optimizer_no, learning_rate, batch_size, epochs, layers1, layers2, 
                    dropout1, dropout_rate1, dropout2, dropout_rate2):      
    optimizerD = {'Adam':Adam(lr=learning_rate), 'SGD':SGD(lr=learning_rate),
                     'RMSprop':RMSprop(lr=learning_rate), 'Adadelta':Adadelta(lr=learning_rate),
                     'Adagrad':Adagrad(lr=learning_rate), 'Adamax':Adamax(lr=learning_rate),
                     'Nadam':Nadam(lr=learning_rate), 'Ftrl':Ftrl(lr=learning_rate)}
    optimizer = list(optimizerD.keys())[optimizer_no]
    nn = Sequential()
    nn.add(layers.LSTM(cellsIni, activation=activation1, return_sequences = True, input_shape=(48,32)))
    
    for i in range(layers1):
        nn.add(layers.LSTM(cells1, recurrent_activation=activation1, return_sequences = True))
        if dropout1 > 0.5:
            nn.add(layers.SpatialDropout1D(dropout_rate1, seed=123))
    
    for i in range(layers2):
        nn.add(layers.LSTM(cells2, recurrent_activation=activation2, return_sequences = True))
        if dropout2 > 0.5:
            nn.add(layers.SpatialDropout1D(dropout_rate2, seed=123))        
    
    nn.add(layers.LSTM(cells3, activation=activation3))
    nn.add(layers.Dense(10, activation='relu'))
    nn.add(layers.Dense(8, activation='softmax'))
    nn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return nn

t = time.time()
clf_model = ft_1dcnn_cl_fun(cellsIni, cells1, cells2, cells3, activation1, activation2, activation3, 
                    optimizer_no, learning_rate, batch_size, epochs, layers1, layers2, 
                    dropout1, dropout_rate1, dropout2, dropout_rate2)

history = clf_model.fit(X_train, y_train_1, epochs=epochs, batch_size = batch_size, validation_split=0.3)
print("elapsed fitting the fine tuned NN", time.time()-t)
# ============================================================================

# ##############
# 6. SAVE MODEL'S RESULTS
# ============================================================================
name = 'los_ts_LSTM'
clf_model.save(name + '.h5')
pickle.dump([params_nn_, history.history], open(name + '_history.pickle', 'wb'))
# ============================================================================



print("")
print("elapsed total", time.time()-t_tot)
print('***********************************************************************')
print('***********************************************************************')
print('***********************************************************************')
print('*******************************FINISHED********************************')
print('***********************************************************************')
print('***********************************************************************')
print('***********************************************************************')
print('')
print('')
print('')
print('')