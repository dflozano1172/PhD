#=============================================================================
##############################################################################
##############################################################################
# IMPORT ALL TIME PACKAGES
##############################################################################
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
import matplotlib.pyplot as plt

import seaborn as sns

import pickle, time, random, sys, warnings
# sys.path is a list of absolute path strings
sys.path.append('/home/d/dlr10/Documents/02_Statitics_modelling/0_FunctionsScripts')
import Loading_Data_Functions as load_fn
import FineTuning_Functions as FineTuning
import Learning_Curves_Functions as LearningCurves


from sklearn import svm
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score, StratifiedKFold

import warnings

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


t_tot = time.time()
# ##############
# 1. LOAD DATA ---------------------------------------------------------------
# ============================================================================
t = time.time()
path = r'/home/d/dlr10/Documents/02_Statitics_modelling/DataSets/'
df_patients_16_18, df_admissions_16_18, df_eobs_16_18 = load_fn.Load_data()
df_patients_19_20, df_admissions_19_20, df_eobs_19_20 = load_fn.Load_data('2019_2020')

X_data_16_18 = pickle.load(open(path + 'df_ts_curb_2016_18_1d.pickle','rb'))#.drop(columns = 'no_sample_series')
X_data_19_20 = pickle.load(open(path + 'df_ts_curb_2019_20_1d.pickle','rb'))#.drop(columns = 'no_sample_series')

# Dictionary of features and types --------
data_types = pd.read_csv('/home/d/dlr10/Documents/02_Statitics_modelling/2_Statistics/csv_Type_variables.csv')
data_types = data_types.set_index('Variable').to_dict()['Type']



print("Elapsed time loading data:", time.time()-t)
# ============================================================================

# TRAINING SET 

adms_curb1 = X_data_16_18[0][X_data_16_18[0]['age_at_admin']>=65]['admission_id'].unique().tolist()
adms_curb2 = X_data_16_18[1][X_data_16_18[1]['age_at_admin']>=65]['admission_id'].unique().tolist()
adms_curb3 = X_data_16_18[2][X_data_16_18[2]['age_at_admin']>=65]['admission_id'].unique().tolist()
print('admins  low risk:  ', len(adms_curb1))
print('admins medium risk:', len(adms_curb2))
print('admins  high risk: ', len(adms_curb3))

# VALIDATION SET 
print("")
adms_val_curb1 = X_data_19_20[0][X_data_19_20[0]['age_at_admin']>=65]['admission_id'].unique().tolist()
adms_val_curb2 = X_data_19_20[1][X_data_19_20[1]['age_at_admin']>=65]['admission_id'].unique().tolist()
adms_val_curb3 = X_data_19_20[2][X_data_19_20[2]['age_at_admin']>=65]['admission_id'].unique().tolist()
print('admins VAL low risk:  ', len(adms_val_curb1))
print('admins VAL medium risk:', len(adms_val_curb2))
print('admins VAL high risk: ', len(adms_val_curb3))

# =========================================
# Sampling Train Set
# =========================================
no_samples = 500 #Number of samples per class

samp_crb_0 = random.sample(adms_val_curb1, no_samples) 
samp_crb_1 = random.sample(adms_curb2, no_samples)
samp_crb_2 = random.sample(adms_curb3, no_samples)
lst_admns  = samp_crb_0 + samp_crb_1 + samp_crb_2

df_X_train = pd.concat([X_data_19_20[0][X_data_19_20[0]['admission_id'].isin(samp_crb_0)],
                X_data_16_18[1][X_data_16_18[1]['admission_id'].isin(samp_crb_1)],
                X_data_16_18[2][X_data_16_18[2]['admission_id'].isin(samp_crb_2)]]).reset_index(drop=True)

curb_Y_train = ([0] * len(samp_crb_0)) + ([1] * len(samp_crb_1)) + ([2] * len(samp_crb_2))

samp_crb_0_t = random.sample(adms_val_curb1, no_samples) 
samp_crb_1_t = random.sample(adms_curb2, no_samples)
samp_crb_2_t = random.sample(adms_curb3, no_samples)
lst_admns  = samp_crb_0 + samp_crb_1 + samp_crb_2

df_X_valid = pd.concat([X_data_19_20[0][X_data_19_20[0]['admission_id'].isin(samp_crb_0_t)],
                X_data_16_18[1][X_data_16_18[1]['admission_id'].isin(samp_crb_1_t)],
                X_data_16_18[2][X_data_16_18[2]['admission_id'].isin(samp_crb_2_t)]]).reset_index(drop=True)

curb_Y_valid = ([0] * len(samp_crb_0)) + ([1] * len(samp_crb_1)) + ([2] * len(samp_crb_2))

# =================================================
# VALIDATION SET
# =================================================
#Target Variable
curb_Y_test = ([0] * len(adms_val_curb1)) + ([1] * len(adms_val_curb2)) + ([2] * len(adms_val_curb3))
# Data dataframe 
df_X_test = pd.concat([X_data_19_20[i][X_data_19_20[i]['age_at_admin']>=65] for i in range(len(X_data_19_20))])
#df_X_test = df_X_test.drop(columns = 'no_sample_series')

print(len(df_X_valid))

feat_list = df_X_test.columns.tolist()
feat_list = feat_list[1:-1]



print("================= NORMALISED TRAINING SET =================")
train_set_norm, encoder = Normalise_n_encode_train_set(df_X_train.reset_index().copy(), feat_list, data_types)
print("================= NORMALISED VALIDATION SET =================")
valid_set_norm          = Normalise_n_encode_val_set(df_X_valid.reset_index(), encoder, feat_list, data_types)
print("================= NORMALISED TEST SET =================")
test_set_norm          = Normalise_n_encode_val_set(df_X_test.reset_index(), encoder, feat_list, data_types)


####### SET DATA AS ARRAYs ######
num_samp_train = len(df_X_train['admission_id'].unique().tolist())
num_samp_valid = len(df_X_valid['admission_id'].unique().tolist())
num_samp_test = len(df_X_test['admission_id'].unique().tolist())
num_features   = len(train_set_norm.columns.tolist())
num_time_samp  = len(df_X_train[df_X_train['admission_id']==df_X_train['admission_id'].tolist()[0]])
X_train    = np.array(train_set_norm).reshape((num_samp_train, num_time_samp, num_features ))
X_train    = X_train[:,0,:] # convert en 2D matrix, and use only static variables
X_test     = np.array(test_set_norm).reshape((num_samp_test, num_time_samp, num_features ))
X_valid    = X_test[:,0,:] # convert en 2D matrix, and use only static variables

y_train = np.array(curb_Y_train)
y_valid = np.array(curb_Y_test)


from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

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
name = 'CURB_sta_SVM'
pickle.dump([X_valid, y_valid, params_nn_, clf_hist], open(name + '_h5.pickle', 'wb'))
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









