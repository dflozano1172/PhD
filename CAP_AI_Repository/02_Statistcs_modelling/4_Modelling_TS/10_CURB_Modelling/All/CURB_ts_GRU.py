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

from sklearn.preprocessing import MinMaxScaler

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

X_data_16_18 = pickle.load(open(path + 'df_ts_curb_2016_18_1d.pickle','rb'))#drop(columns = 'no_sample_series')
X_data_19_20 = pickle.load(open(path + 'df_ts_curb_2019_20_1d.pickle','rb'))#drop(columns = 'no_sample_series')

# Dictionary of features and types --------
data_types = pd.read_csv('/home/d/dlr10/Documents/02_Statitics_modelling/2_Statistics/csv_Type_variables.csv')
data_types = data_types.set_index('Variable').to_dict()['Type']



print("Elapsed time loading data:", time.time()-t)
# ============================================================================

# TRAINING SET 

adms_curb1 = X_data_16_18[0]['admission_id'].unique().tolist()
adms_curb2 = X_data_16_18[1]['admission_id'].unique().tolist()
adms_curb3 = X_data_16_18[2]['admission_id'].unique().tolist()
print('admins  low risk:  ', len(adms_curb1))
print('admins medium risk:', len(adms_curb2))
print('admins  high risk: ', len(adms_curb3))

# VALIDATION SET 
print("")
adms_val_curb1 = X_data_19_20[0]['admission_id'].unique().tolist()
adms_val_curb2 = X_data_19_20[1]['admission_id'].unique().tolist()
adms_val_curb3 = X_data_19_20[2]['admission_id'].unique().tolist()
print('admins VAL low risk:  ', len(adms_val_curb1))
print('admins VAL medium risk:', len(adms_val_curb2))
print('admins VAL high risk: ', len(adms_val_curb3))

# =========================================
# Sampling Train Set
# =========================================
no_samples = 570 #Number of samples per class

samp_crb_0 = random.sample(adms_curb1, no_samples) 
samp_crb_1 = random.sample(adms_curb2, no_samples)
samp_crb_2 = random.sample(adms_curb3, no_samples)
lst_admns  = samp_crb_0 + samp_crb_1 + samp_crb_2

df_X_train = pd.concat([X_data_16_18[0][X_data_16_18[0]['admission_id'].isin(samp_crb_0)],
                X_data_16_18[1][X_data_16_18[1]['admission_id'].isin(samp_crb_1)],
                X_data_16_18[2][X_data_16_18[2]['admission_id'].isin(samp_crb_2)]]).reset_index(drop=True)

curb_Y_train = ([0] * len(samp_crb_0)) + ([1] * len(samp_crb_1)) + ([2] * len(samp_crb_2))

samp_crb_0_t = random.sample(adms_curb1, no_samples) 
samp_crb_1_t = random.sample(adms_curb2, no_samples)
samp_crb_2_t = random.sample(adms_curb3, no_samples)
lst_admns  = samp_crb_0 + samp_crb_1 + samp_crb_2

df_X_valid = pd.concat([X_data_16_18[0][X_data_16_18[0]['admission_id'].isin(samp_crb_0_t)],
                X_data_16_18[1][X_data_16_18[1]['admission_id'].isin(samp_crb_1_t)],
                X_data_16_18[2][X_data_16_18[2]['admission_id'].isin(samp_crb_2_t)]]).reset_index(drop=True)

curb_Y_valid = ([0] * len(samp_crb_0)) + ([1] * len(samp_crb_1)) + ([2] * len(samp_crb_2))

# =================================================
# VALIDATION SET
# =================================================
#Target Variable
curb_Y_test = ([0] * len(adms_val_curb1)) + ([1] * len(adms_val_curb2)) + ([2] * len(adms_val_curb3))
# Data dataframe 
df_X_test = pd.concat(X_data_19_20)


#df_X_test = df_X_test.drop(columns = 'ews')

print(len(df_X_test))

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
X_valid    = np.array(valid_set_norm).reshape((num_samp_valid, num_time_samp, num_features ))
X_test     = np.array(test_set_norm).reshape((num_samp_test, num_time_samp, num_features ))

X_train = X_train[:,:48,:]
X_valid = X_valid[:,:48,:]
X_test  = X_test[:,:48,:]

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(curb_Y_train , num_classes = 3)
y_valid = to_categorical(curb_Y_valid , num_classes = 3)
y_test = to_categorical(curb_Y_test , num_classes = 3)

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)
X_valid, y_valid = shuffle(X_valid, y_valid)

X_train = X_train.astype(float)
X_valid = X_valid.astype(float)
print("Elapsed time processing data:", time.time()-t)
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from tensorflow.keras import regularizers


nn = Sequential()
nn.add(layers.GRU(400, activation='tanh', return_sequences = False, input_shape=(48,32), dropout=0.3,
    recurrent_dropout=0.3, kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.L2(1e-4),
    activity_regularizer=regularizers.L2(1e-5)))

nn.add(layers.Dropout(0.5))
nn.add(layers.Dense(3, activation='softmax'))

learning_rate = 1e-6
optimizerD = {'Adam':Adam(lr=learning_rate), 'SGD':SGD(lr=learning_rate),
                 'RMSprop':RMSprop(lr=learning_rate), 'Adadelta':Adadelta(lr=learning_rate),
                 'Adagrad':Adagrad(lr=learning_rate), 'Adamax':Adamax(lr=learning_rate),
                 'Nadam':Nadam(lr=learning_rate), 'Ftrl':Ftrl(lr=learning_rate)}


nn.compile(loss='categorical_crossentropy', 
           optimizer=optimizerD['RMSprop'],
           metrics=['accuracy'])

history = nn.fit(X_train, y_train, 
                 validation_data=(X_valid, y_valid), epochs=200, batch_size=200, verbose=1)



# list all data in history
print(history.history.keys())

name = 'CURB_ts_GRU'
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig(name +'_accuracy.png', transparent = True, bbox_inches = "tight")
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig(name +'_loss.png', transparent = True, bbox_inches = "tight")
plt.show()


#clf_model.save(name + '.h5')
nn.save(name + '.h5')
pickle.dump([X_test, y_test, history.history], open(name + '_h5.pickle', 'wb'))

print('***********************************************************************')
print('***********************************************************************')
print('***********************************************************************')
print('*****************************FINISHED GRU******************************')
print('***********************************************************************')
print('***********************************************************************')
print('***********************************************************************')
