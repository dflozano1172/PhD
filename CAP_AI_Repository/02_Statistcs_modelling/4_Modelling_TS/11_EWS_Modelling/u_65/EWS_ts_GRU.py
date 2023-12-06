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

X_data_16_18 = pickle.load(open(path + 'df_ts_2016_18_ews_1d.pickle','rb'))
X_data_19_20 = pickle.load(open(path + 'df_ts_2019_20_ews_1d.pickle','rb'))

# List of features ------------------------
feat_list = X_data_16_18[0].drop(columns=['ews','no_sample_series']).columns.tolist()
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
dict_adms = dict(zip(adms_total, [EWS_bands(x) for x in X_data_16_18[1]]))

adms = X_data_16_18[0][X_data_16_18[0]['age_at_admin']<65]['admission_id'].unique().tolist()

adms_ews0 = [adms[i] for i in range(len(adms)) if (dict_adms[adms[i]] == 0)]
adms_ews1 = [adms[i] for i in range(len(adms)) if (dict_adms[adms[i]] == 1)]
adms_ews2 = [adms[i] for i in range(len(adms)) if (dict_adms[adms[i]] == 2)]
adms_ews3 = [adms[i] for i in range(len(adms)) if (dict_adms[adms[i]] == 3)]
adms_ews4 = [adms[i] for i in range(len(adms)) if (dict_adms[adms[i]] >= 4)]
print('admins  EWS 0:  ', len(adms_ews0))
print('admins  EWS 1:  ', len(adms_ews1))
print('admins  EWS 2:  ', len(adms_ews2))
print('admins  EWS 3:  ', len(adms_ews3))
print('admins  EWS 4:  ', len(adms_ews4))

# TRAINING SET 
print("")
adms_total_val = X_data_19_20[0]['admission_id'].unique().tolist()
dict_adms_val  = dict(zip(adms_total_val, [EWS_bands(x) for x in X_data_19_20[1]]))
adms_val = X_data_19_20[0][X_data_19_20[0]['age_at_admin']<65]['admission_id'].unique().tolist()

adms_val_ews0 = [adms_val[i] for i in range(len(adms_val)) if (dict_adms_val[adms_val[i]] == 0)]
adms_val_ews1 = [adms_val[i] for i in range(len(adms_val)) if (dict_adms_val[adms_val[i]] == 1)]
adms_val_ews2 = [adms_val[i] for i in range(len(adms_val)) if (dict_adms_val[adms_val[i]] == 2)]
adms_val_ews3 = [adms_val[i] for i in range(len(adms_val)) if (dict_adms_val[adms_val[i]] == 3)]
adms_val_ews4 = [adms_val[i] for i in range(len(adms_val)) if (dict_adms_val[adms_val[i]] >= 4)]
print('admins VAL EWS 0:  ', len(adms_val_ews0))
print('admins VAL EWS 1:  ', len(adms_val_ews1))
print('admins VAL EWS 2:  ', len(adms_val_ews2))
print('admins VAL EWS 3:  ', len(adms_val_ews3))
print('admins VAL EWS 4:  ', len(adms_val_ews4))

# =========================================
# Sampling Train Set
# =========================================
no_samples = 240 #Number of samples per class

samp_ews_0 = random.sample(adms_ews0, no_samples) 
samp_ews_1 = random.sample(adms_ews1, no_samples)
samp_ews_2 = random.sample(adms_ews2, no_samples)
samp_ews_3 = random.sample(adms_ews3, no_samples)
samp_ews_4 = random.sample(adms_ews4, no_samples)
lst_admns  = samp_ews_0 + samp_ews_1 + samp_ews_2 + samp_ews_3 + samp_ews_4

df_X_train  = X_data_16_18[0][X_data_16_18[0]['admission_id'].isin(lst_admns)].reset_index(drop=True)
ews_Y_train = [dict_adms[adm] for adm in  df_X_train['admission_id'].unique().tolist()]

samp_ews_0_t = random.sample(adms_ews0, no_samples) 
samp_ews_1_t = random.sample(adms_ews1, no_samples) 
samp_ews_2_t = random.sample(adms_ews2, no_samples) 
samp_ews_3_t = random.sample(adms_ews3, no_samples) 
samp_ews_4_t = random.sample(adms_ews4, no_samples) 

lst_admns_t  = samp_ews_0_t + samp_ews_1_t + samp_ews_2_t + samp_ews_3_t + samp_ews_4_t

df_X_valid  = X_data_16_18[0][X_data_16_18[0]['admission_id'].isin(lst_admns_t)].reset_index(drop=True)
ews_Y_valid = [dict_adms[adm] for adm in  df_X_valid['admission_id'].unique().tolist()]


# =================================================
# VALIDATION SET
# =================================================
#Target Variable
# Data dataframe 
df_X_test = X_data_19_20[0][X_data_19_20[0]['age_at_admin']<65]
ews_Y_test = [dict_adms_val[x] for x in df_X_test['admission_id'].unique().tolist()]


print(len(df_X_test))


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
X_valid    = np.array(valid_set_norm).reshape((num_samp_valid, num_time_samp, num_features ))
X_test     = np.array(test_set_norm).reshape((num_samp_test, num_time_samp, num_features ))


X_train = X_train[:,:48,:]
X_valid = X_valid[:,:48,:]
X_test  = X_test[:,:48,:]


y_train = to_categorical(ews_Y_train , num_classes = 5)
y_valid = to_categorical(ews_Y_valid , num_classes = 5)
y_test  = to_categorical(ews_Y_test , num_classes = 5)



print("Shape of X_train",X_train.shape)
print("Shape of y_train",y_train.shape)

print("Shape of X_valid",X_train.shape)
print("Shape of y_valid",y_train.shape)

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)
X_valid, y_valid = shuffle(X_valid, y_valid)

print("Elapsed time processing data:", time.time()-t)
# ============================================================================


from tensorflow.keras import regularizers
nn = Sequential()
nn.add(layers.GRU(250, activation='tanh', return_sequences = False, input_shape=(48,32), 
                  recurrent_dropout=0.3, kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                  bias_regularizer=regularizers.L2(1e-4)))

nn.add(layers.Dropout(0.5))
nn.add(layers.Dense(5, activation='softmax'))

learning_rate = 1e-6
optimizerD = {'Adam':Adam(lr=learning_rate), 'SGD':SGD(lr=learning_rate),
                 'RMSprop':RMSprop(lr=learning_rate), 'Adadelta':Adadelta(lr=learning_rate),
                 'Adagrad':Adagrad(lr=learning_rate), 'Adamax':Adamax(lr=learning_rate),
                 'Nadam':Nadam(lr=learning_rate), 'Ftrl':Ftrl(lr=learning_rate)}


nn.compile(loss='categorical_crossentropy', 
           optimizer=optimizerD['Adam'],
           metrics=['accuracy'])

history = nn.fit(X_train, y_train, 
                 validation_data=(X_valid, y_valid), epochs=200, batch_size=200, verbose=1)



# list all data in history
print(history.history.keys())

name = 'EWS_GRU_ts'
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('Learning_curves'+name +'_accuracy.png', transparent = True, bbox_inches = "tight")
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('Learning_curves'+name +'_loss.png', transparent = True, bbox_inches = "tight")
plt.show()


#clf_model.save(name + '.h5')
nn.save(name + '.h5')
pickle.dump([X_test, y_test, history.history], open(name + '_h5.pickle', 'wb'))


print('***********************************************************************')
print('***********************************************************************')
print('***********************************************************************')
print('*****************************FINISHED GRU*****************************')
print('***********************************************************************')
print('***********************************************************************')
print('***********************************************************************')
