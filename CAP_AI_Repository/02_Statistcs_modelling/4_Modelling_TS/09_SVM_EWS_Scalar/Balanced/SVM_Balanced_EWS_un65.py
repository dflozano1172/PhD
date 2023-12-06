%reset -f
# ============================================================================
##############################################################################
##############################################################################
# IMPORT ALL TIME PACKAGES
##############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm

import pickle, time, random, sys, warnings
# sys.path is a list of absolute path strings
sys.path.append('/home/d/dlr10/Documents/02_Statitics_modelling/0_FunctionsScripts')
import Loading_Data_Functions as load_fn
from sklearn.model_selection import cross_val_score, StratifiedKFold

from bayes_opt import BayesianOptimization
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################

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

############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################



t_tot = time.time()
# ##############
# 1. LOAD DATA ---------------------------------------------------------------
# ============================================================================
t = time.time()

path = r'/home/d/dlr10/Documents/02_Statitics_modelling/DataSets/'

X_data_16_18 = pickle.load(open(path + 'df_ts_2016_18_ews_1d.pickle','rb'))
X_data_19_20 = pickle.load(open(path + 'df_ts_2019_20_ews_1d.pickle','rb'))

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
### BALANCED DATA ####################################
random.seed(2)
adm_list = X_data_16_18[0]['admission_id'].unique().tolist()
X_0 = X_data_16_18[0][X_data_16_18[0]['Mortality'] == 0]['admission_id'].unique().tolist()
X_1 = X_data_16_18[0][(X_data_16_18[0]['Mortality'] == 1) & (X_data_16_18[0]['age_at_admin'] <= 65) ]

list_X_0   = random.sample(range(len(X_0)),len(X_1['admission_id'].unique().tolist()))
X_0_2_adms = [X_0[pos] for pos in list_X_0]
X_0_2      = X_data_16_18[0][X_data_16_18[0]['admission_id'].isin(X_0_2_adms)]

idx_final = [idx for idx, adm in enumerate(adm_list) if (adm in X_0_2_adms) or (adm in X_1['admission_id'].unique().tolist())]
X_data_bal = pd.concat([X_1,X_0_2])
y_data_bal = [ews for idx, ews in enumerate(X_data_16_18[1]) if idx in idx_final]

### SPLIT DATA #######################################
train_set = X_data_bal.copy()
valid_set = X_data_19_20[0].copy()

### ENCODING DATA ####################################
train_set_norm, encoder = Normalise_n_encode_train_set(train_set, feat_list, data_types)
valid_set_norm          = Normalise_n_encode_val_set(valid_set, encoder, feat_list, data_types)

### SET DATA AS ARRAYs ###############################
num_samp_train = len(train_set['admission_id'].unique().tolist())
num_samp_valid = len(valid_set['admission_id'].unique().tolist())
num_features   = len(train_set_norm.columns.tolist())
num_time_samp  = len(train_set[train_set['admission_id']==train_set['admission_id'].tolist()[0]])
X_train    = np.array(train_set_norm).reshape((num_samp_train, num_time_samp, num_features ))
X_train    = X_train[:,0,:] # convert en 2D matrix, and use only static variables
X_valid    = np.array(valid_set_norm).reshape((num_samp_valid, num_time_samp, num_features ))
X_valid    = X_valid[:,0,:] # convert en 2D matrix, and use only static variables
df         = train_set[['admission_id', 'Mortality']].groupby(by= 'admission_id').mean()
mort_dict  = dict(zip(df.index.tolist(),df['Mortality']))

EWS_bands = lambda x: 0 if x == 0 else 1 if x == 1 else 2 if x == 2 else 3 if x == 3 else 4
y_train = np.array([EWS_bands(x) for x in np.array(y_data_bal.copy())])

df         = valid_set[['admission_id', 'Mortality']].groupby(by= 'admission_id').mean()
mort_dict  = dict(zip(df.index.tolist(),df['Mortality']))
y_valid    = np.array(X_data_19_20[1].copy())
y_valid    = np.array([EWS_bands(x) for x in y_valid])

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

params = {'kernel' :(0,3), 'degree':(1, 5), 'gamma': (0,1), 'c_p': (0,100)}

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
name = 'Balanced_SVM_scal_un65'
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









