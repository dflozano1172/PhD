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
from sklearn import svm
from bayes_opt import BayesianOptimization

import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)

print('***********************************************************************')
print('***********************************************************************')
print('***********************************************************************')
print('******************** STARTING LSTM EWS BALANCED ***********************')
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

dict_classes = {'3-4d':0, '5-6d':1, '7d':2, '8-9d':3,'10-13d':4, '2w':5, '3w':6, '4w':7}

y_16_18 = [dict_classes[x] for a,x in  X_data_16_18[1]]
y_19_20 = [dict_classes[x] for a,x in  X_data_19_20[1]]

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

EWS_bands = lambda x: 0 if x == 0 else 1 if x == 1 else 2 if x == 2 else 3 if x == 3 else 4
# TRAINING SET 
adms_total = X_data_16_18[0]['admission_id'].unique().tolist()
dict_adms = dict(zip(adms_total, y_16_18))

# TRAINING SET 

adms = X_data_16_18[0][X_data_16_18[0]['age_at_admin']>=65]['admission_id'].unique().tolist()

adms_ews0 = [adms[i] for i in range(len(adms)) if (dict_adms[adms[i]] == 0)]
adms_ews1 = [adms[i] for i in range(len(adms)) if (dict_adms[adms[i]] == 1)]
adms_ews2 = [adms[i] for i in range(len(adms)) if (dict_adms[adms[i]] == 2)]
adms_ews3 = [adms[i] for i in range(len(adms)) if (dict_adms[adms[i]] == 3)]
adms_ews4 = [adms[i] for i in range(len(adms)) if (dict_adms[adms[i]] == 4)]
adms_ews5 = [adms[i] for i in range(len(adms)) if (dict_adms[adms[i]] == 5)]
adms_ews6 = [adms[i] for i in range(len(adms)) if (dict_adms[adms[i]] == 6)]
adms_ews7 = [adms[i] for i in range(len(adms)) if (dict_adms[adms[i]] == 7)]
print('admins  los 0:  ', len(adms_ews0))
print('admins  los 1:  ', len(adms_ews1))
print('admins  los 2:  ', len(adms_ews2))
print('admins  los 3:  ', len(adms_ews3))
print('admins  los 4:  ', len(adms_ews4))
print('admins  los 5:  ', len(adms_ews5))
print('admins  los 6:  ', len(adms_ews6))
print('admins  los 7:  ', len(adms_ews7))

# TRAINING SET 
print("")
adms_total_val = X_data_19_20[0]['admission_id'].unique().tolist()
dict_adms_val  = dict(zip(adms_total_val, y_19_20))
adms_val = X_data_19_20[0][X_data_19_20[0]['age_at_admin']>=65]['admission_id'].unique().tolist()

adms_val_ews0 = [adms_val[i] for i in range(len(adms_val)) if (dict_adms_val[adms_val[i]] == 0)]
adms_val_ews1 = [adms_val[i] for i in range(len(adms_val)) if (dict_adms_val[adms_val[i]] == 1)]
adms_val_ews2 = [adms_val[i] for i in range(len(adms_val)) if (dict_adms_val[adms_val[i]] == 2)]
adms_val_ews3 = [adms_val[i] for i in range(len(adms_val)) if (dict_adms_val[adms_val[i]] == 3)]
adms_val_ews4 = [adms_val[i] for i in range(len(adms_val)) if (dict_adms_val[adms_val[i]] == 4)]
adms_val_ews5 = [adms_val[i] for i in range(len(adms_val)) if (dict_adms_val[adms_val[i]] == 5)]
adms_val_ews6 = [adms_val[i] for i in range(len(adms_val)) if (dict_adms_val[adms_val[i]] == 6)]
adms_val_ews7 = [adms_val[i] for i in range(len(adms_val)) if (dict_adms_val[adms_val[i]] == 7)]
print('admins VAL los 0:  ', len(adms_val_ews0))
print('admins VAL los 1:  ', len(adms_val_ews1))
print('admins VAL los 2:  ', len(adms_val_ews2))
print('admins VAL los 3:  ', len(adms_val_ews3))
print('admins VAL los 4:  ', len(adms_val_ews4))
print('admins VAL los 5:  ', len(adms_val_ews5))
print('admins VAL los 6:  ', len(adms_val_ews6))
print('admins VAL los 7:  ', len(adms_val_ews7))

# =========================================
# Sampling Train Set
# =========================================
no_samples = 430 #Number of samples per class

samp_los_0 = random.sample(adms_ews0, no_samples) 
samp_los_1 = random.sample(adms_ews1, no_samples)
samp_los_2 = random.sample(adms_ews2, no_samples)
samp_los_3 = random.sample(adms_ews3, no_samples)
samp_los_4 = random.sample(adms_ews4, no_samples)
samp_los_5 = random.sample(adms_ews5, no_samples)
samp_los_6 = random.sample(adms_ews6, no_samples)
samp_los_7 = random.sample(adms_ews7, no_samples)
lst_admns  = samp_los_0 + samp_los_1 + samp_los_2 + samp_los_3 + samp_los_4 + samp_los_5 + samp_los_6 + samp_los_7

df_X_train  = X_data_16_18[0][X_data_16_18[0]['admission_id'].isin(lst_admns)].reset_index(drop=True)
los_Y_train = [dict_adms[adm] for adm in  df_X_train['admission_id'].unique().tolist()]

samp_ews_0_t = random.sample(adms_ews0, no_samples) 
samp_ews_1_t = random.sample(adms_ews1, no_samples) 
samp_ews_2_t = random.sample(adms_ews2, no_samples) 
samp_ews_3_t = random.sample(adms_ews3, no_samples) 
samp_ews_4_t = random.sample(adms_ews4, no_samples) 
samp_ews_5_t = random.sample(adms_ews5, no_samples) 
samp_ews_6_t = random.sample(adms_ews6, no_samples) 
samp_ews_7_t = random.sample(adms_ews7, no_samples) 

lst_admns_t  = samp_ews_0_t + samp_ews_1_t + samp_ews_2_t + samp_ews_3_t + samp_ews_4_t + samp_ews_5_t + samp_ews_6_t + samp_ews_7_t

df_X_valid  = X_data_16_18[0][X_data_16_18[0]['admission_id'].isin(lst_admns_t)].reset_index(drop=True)
los_Y_valid = [dict_adms[adm] for adm in  df_X_valid['admission_id'].unique().tolist()]

# =================================================
# VALIDATION SET
# =================================================
#Target Variable
df_X_test = X_data_19_20[0][X_data_19_20[0]['age_at_admin']>=65]
los_Y_test = [dict_adms_val[x] for x in df_X_test['admission_id'].unique().tolist()]

print(len(df_X_test))
feat_list = df_X_valid.columns.tolist()
feat_list = feat_list[1:-1]

print("================= NORMALISED TRAINING SET =================")
train_set_norm, encoder = Normalise_n_encode_train_set(df_X_train.reset_index().copy(), feat_list, data_types)
valid_set_norm          = Normalise_n_encode_val_set(df_X_valid.reset_index(), encoder, feat_list, data_types)
test_set_norm          = Normalise_n_encode_val_set(df_X_test.reset_index(), encoder, feat_list, data_types)

print("================= NORMALISED VALIDATION SET =================")

####### SET DATA AS ARRAYs ######
num_samp_train = len(df_X_train['admission_id'].unique().tolist())
num_samp_valid = len(df_X_valid['admission_id'].unique().tolist())
num_samp_test = len(df_X_test['admission_id'].unique().tolist())
num_features   = len(train_set_norm.columns.tolist())
num_time_samp  = len(df_X_train[df_X_train['admission_id']==df_X_train['admission_id'].tolist()[0]])
X_train    = np.array(train_set_norm).reshape((num_samp_train, num_time_samp, num_features ))
X_train    = X_train[:,0,:] # convert en 2D matrix, and use only static variables
X_valid    = np.array(valid_set_norm).reshape((num_samp_valid, num_time_samp, num_features ))
X_valid    = X_valid[:,0,:] # convert en 2D matrix, and use only static variables
X_test     = np.array(test_set_norm).reshape((num_samp_test, num_time_samp, num_features ))
X_test    = X_test[:,0,:] # convert en 2D matrix, and use only static variables


feat_list = df_X_valid.columns.tolist()
feat_list = feat_list[1:-1]


y_train = np.array(los_Y_train)
y_valid = np.array(los_Y_valid)
y_test = np.array(los_Y_test)




print("Shape of X_train",X_train.shape)
print("Shape of y_train",y_train.shape)

print("Shape of X_valid",X_train.shape)
print("Shape of y_valid",y_train.shape)

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)
X_valid, y_valid = shuffle(X_valid, y_valid)

print("Elapsed time processing data:", time.time()-t)
# ============================================================================

# ##############
# 3. BAYESIAN OPTIMIZER TO FINE TUNE SVM
# ============================================================================
t = time.time()
### CONSTRUCT THE CLASSIFIER ###############################
def nn_cl_bo2(kernel, degree, gamma, c_p):
    gamma_val  = ['scale', 'auto']
    kernel_val = ['linear', 'rbf', 'poly','sigmoid']
    
    ker = kernel_val[round(kernel)]
    gam = gamma_val[round(gamma)]
    deg = round(degree)
    
    clf  = svm.SVC(kernel=ker, C=c_p, decision_function_shape='ovo', gamma = gam, degree=deg)
    
    #nn    = KerasClassifier(build_fn=nn_cl_fun, epochs=epochs, batch_size=batch_size, verbose=0)
    kfold = StratifiedKFold(n_splits=20, shuffle=True, random_state=123)
    score = cross_val_score(clf, X_train, y_train, cv=kfold).mean()
    return score

params = {'kernel' :(0,3), 'degree':(1, 5), 'gamma': (0,1), 'c_p': (1,10)}

### Run Bayesian Optimization ###############################
t = time.time()
nn_bo = BayesianOptimization(nn_cl_bo2, params, random_state=111)
nn_bo.maximize(init_points=20, n_iter=20)
print("Elapsed running the bayesian opt:", time.time()-t)
# ============================================================================

# ##############
# 4. EXTRACTING RESULTS FROM FINE-TUNING
# ============================================================================
params_nn_ = nn_bo.max['params']
    
params_nn_['kernel'] = round(params_nn_['kernel'])
params_nn_['degree'] = round(params_nn_['degree'])
params_nn_['gamma']  = round(params_nn_['gamma'])

kernel = params_nn_['kernel']
degree = params_nn_['degree']
gamma  = params_nn_['gamma']
c_p    = params_nn_['c_p']
# ============================================================================

# ##############
# 5. FIT CLASSIFIER WITH FINE TUNE GHYPERPARAMETERS
# ============================================================================
t = time.time()
### Build classifier ###############################
def cl_bo2(kernel, degree, gamma, c_p):
    gamma_val  = ['scale', 'auto']
    kernel_val = ['linear', 'rbf', 'poly','sigmoid']
    
    ker = kernel_val[round(kernel)]
    gam = gamma_val[round(gamma)]
    deg = round(degree)
    
    clf  = svm.SVC(kernel=ker, C=c_p, decision_function_shape='ovo', gamma = gam, degree=deg)
    
    return clf

clf_model = cl_bo2(kernel, degree, gamma, c_p)
clf_hist  = clf_model.fit(X_train, y_train)

print("Elapsed training bay-optimal model:", time.time()-t)
# ============================================================================


# ##############
# 6. SAVE MODEL'S RESULTS
# ============================================================================
name = 'LOS_sta_SVM'
pickle.dump([X_test, y_test, params_nn_, clf_hist], open(name + '_h5.pickle', 'wb'))
# ============================================================================

print("")
print("Elapsed TOTAL time  data:", time.time()-t_tot)


print('***********************************************************************')
print('***********************************************************************')
print('***********************************************************************')
print('****************************FINISHED SVM*******************************')
print('***********************************************************************')
print('***********************************************************************')
print('***********************************************************************')








