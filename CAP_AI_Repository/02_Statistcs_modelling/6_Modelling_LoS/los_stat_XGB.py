# ============================================================================
##############################################################################
##############################################################################
# IMPORT ALL TIME PACKAGES
##############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

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
#df_patients_16_18, df_admissions_16_18, _ = load_fn.Load_data()
#df_patients_19_20, df_admissions_19_20, _ = load_fn.Load_data('2019_2020')

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
train_set = X_data_16_18[0].copy()
adm_ts    = train_set['admission_id'].unique().tolist()
valid_set = X_data_19_20[0].copy()

### ENCODING DATA ####################################
train_set_norm, encoder = Normalise_n_encode_train_set(train_set, feat_list, data_types)
valid_set_norm          = Normalise_n_encode_val_set(valid_set, encoder, feat_list, data_types)

### SET DATA AS ARRAYs ###############################
num_samp_train = len(train_set['admission_id'].unique().tolist())
num_samp_valid = len(valid_set['admission_id'].unique().tolist())
num_features   = len(train_set_norm.columns.tolist())
num_time_samp  = len(train_set[train_set['admission_id']==adm_ts[0]])
X_train    = np.array(train_set_norm).reshape((num_samp_train, num_time_samp, num_features ))
X_train    = X_train[:,0,:] # convert en 2D matrix, and use only static variables
X_valid    = np.array(valid_set_norm).reshape((num_samp_valid, num_time_samp, num_features ))
X_valid    = X_valid[:,0,:] # convert en 2D matrix, and use only static variables
df         = train_set[['admission_id', 'Mortality']].groupby(by= 'admission_id').mean()
mort_dict  = dict(zip(df.index.tolist(),df['Mortality']))

y_train = list(list(zip(*X_data_16_18[1]))[1])
#dict_classes = {'3d':0, '4d':1,'5d':2,'6d':3, '1w':4,'2w':5,'3w':6, '4w':7}
dict_classes = {'3-4d':0, '5-6d':1, '7d':2, '8-9d':3,'10-13d':4, '2w':5, '3w':6, '4w':7}
los_codes    = lambda x: dict_classes[x] 
y_train      = np.array([los_codes(x) for x in y_train])
y_valid  = list(list(zip(*X_data_19_20[1]))[1])
y_valid  = np.array([los_codes(x) for x in y_valid])

print("Elapsed time processing data:", time.time()-t)
# ============================================================================

## ##############
# 3. BAYESIAN OPTIMIZER TO FINE TUNE SVM
# ============================================================================
t = time.time()
### CONSTRUCT THE CLASSIFIER ###############################
def nn_cl_bo2(max_depth, booster, learning_rate, gamma, tree_method, num_parallel_tree, n_estimators):
    booster_val  = ['gbtree', 'gblinear', 'dart']
    tree_met_val = ['exact', 'approx', 'hist']
    
    boos_fn      = booster_val[round(booster)]
    tree_met_fn  = tree_met_val[round(tree_method)]
    n_estim_fn   = round(n_estimators)
    n_parll_tree = round(num_parallel_tree)
    max_depth_fn = round(max_depth)
    gamma_fn     = round(gamma)
    
    if round(booster) == 1:
        clf  = XGBClassifier(booster=boos_fn, learning_rate=learning_rate, n_estimators=n_estim_fn,
                         random_state=0, objective='multi:softmax', num_class = 8, eval_metric='mlogloss')
   
    else: 
        clf  = XGBClassifier(booster=boos_fn, max_depth= max_depth_fn, learning_rate=learning_rate, gamma=gamma_fn,
                         tree_method=tree_met_fn, num_parallel_tree=n_parll_tree, n_estimators=n_estim_fn,
                         random_state=0, objective='multi:softmax', num_class = 8, eval_metric='mlogloss')
                         
    
    kfold = StratifiedKFold(n_splits=8, shuffle=True, random_state=123)
    score = cross_val_score(clf, X_train, y_train, cv=kfold).mean()
    return score

params = {'max_depth': (5,15), 'booster': (0,2), 'learning_rate':(0.1, 0.5), 'gamma':(0,10), 
          'tree_method': (0,2.3), 'num_parallel_tree':(1,100), 'n_estimators':(50,100)}
          



### Run Bayesian Optimization ###############################
t = time.time()
nn_bo = BayesianOptimization(nn_cl_bo2, params, random_state=111)
nn_bo.maximize(init_points=20, n_iter=20)
print("Elapsed running the bayesian opt:", time.time()-t)
# ============================================================================

# ##############
# 4. EXTRACTING RESULTS FROM FINE-TUNING
# ============================================================================
booster_val  = ['gbtree', 'gblinear', 'dart']
tree_met_val = ['exact', 'grow_local_histmaker', 'approx', 'hist']

params_nn_ = nn_bo.max['params']
    
#params_nn_['booster'] = booster_val[round(params_nn_['booster'])]
#params_nn_['tree_method'] = tree_met_val[round(params_nn_['tree_method'])]
#params_nn_['n_estimators'] = round(params_nn_['n_estimators'])
#params_nn_['num_parallel_tree'] = round(params_nn_['num_parallel_tree'])
#params_nn_['max_depth'] = round(params_nn_['max_depth'])
#params_nn_['gamma'] = round(params_nn_['gamma'])

booster = params_nn_['booster'] 
tree_method = params_nn_['tree_method'] 
n_estimators = params_nn_['n_estimators'] 
num_parallel_tree = params_nn_['num_parallel_tree'] 
max_depth = params_nn_['max_depth']
gamma = params_nn_['gamma']
learning_rate = params_nn_['learning_rate']


# ============================================================================

# ##############
# 5. FIT CLASSIFIER WITH FINE TUNE GHYPERPARAMETERS
# ============================================================================
t = time.time()
### Build classifier ###############################
def cl_bo2(max_depth, booster, learning_rate, gamma, tree_method, num_parallel_tree, n_estimators):
    booster_val  = ['gbtree', 'gblinear', 'dart']
    tree_met_val = ['exact', 'approx', 'hist']
    
    boos_fn      = booster_val[round(booster)]
    tree_met_fn  = tree_met_val[round(tree_method)]
    n_estim_fn   = round(n_estimators)
    n_parll_tree = round(num_parallel_tree)
    max_depth_fn = round(max_depth)
    gamma_fn     = round(gamma)
    
    if round(booster) == 1:
        clf  = XGBClassifier(booster=boos_fn, learning_rate=learning_rate, n_estimators=n_estim_fn,
                         random_state=0, objective='multi:softmax', num_class = 8, eval_metric='mlogloss')
   
    else: 
        clf  = XGBClassifier(booster=boos_fn, max_depth= max_depth_fn, learning_rate=learning_rate, gamma=gamma_fn,
                         tree_method=tree_met_fn, num_parallel_tree=n_parll_tree, n_estimators=n_estim_fn,
                         random_state=0, objective='multi:softmax', num_class = 8, eval_metric='mlogloss')
                         
    return clf


clf_model = cl_bo2(max_depth, booster, learning_rate, gamma, tree_method, num_parallel_tree, n_estimators)
clf_hist  = clf_model.fit(X_train, y_train)

print("Elapsed training bay-optimal model:", time.time()-t)
# ============================================================================

# ##############
# 6. SAVE MODEL'S RESULTS
# ============================================================================
name = 'los_stat_XGB'
pickle.dump([params_nn_, clf_hist], open(name + '_h5.pickle', 'wb'))
# ============================================================================

print("")
print("Elapsed TOTAL time  data:", time.time()-t_tot)


print('***********************************************************************')
print('***********************************************************************')
print('***********************************************************************')
print('*******************************FINISHED********************************')
print('***********************************************************************')
print('***********************************************************************')
print('***********************************************************************')









