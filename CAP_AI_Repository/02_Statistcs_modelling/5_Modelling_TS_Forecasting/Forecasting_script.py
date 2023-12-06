
import pandas as pd
import numpy as np
import time, pickle, random, argparse

import Functions_forecasting as Forecast_fn


import matplotlib


matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['mathtext.fontset'] = 'cm'



feat_pos = 2

# parser = argparse.ArgumentParser(description='the position of the classifier in list classfiiers inside the script.')
# parser.add_argument("-p", "--print_string", help="Prints the supplied argument.",  nargs='*')
# args = parser.parse_args()
# print(args.print_string)
# feat_pos = int(args.print_string[0])



# ##############
# 1. LOAD DATA ---------------------------------------------------------------
# ============================================================================
t = time.time()
path = r'/home/d/dlr10/Documents/02_Statitics_modelling/DataSets/'
#df_patients_16_18, df_admissions_16_18, _ = load_fn.Load_data()
#df_patients_19_20, df_admissions_19_20, _ = load_fn.Load_data('2019_2020')

X_data_16_18 = pickle.load(open(path + 'df_ts_2016_18.pickle','rb'))
X_data_19_20 = pickle.load(open(path + 'df_ts_2019_20.pickle','rb'))

ls_adms_16_18 = X_data_16_18['admission_id'].unique().tolist()
ls_adms_19_20 = X_data_19_20['admission_id'].unique().tolist()

# List of features ------------------------
feat_list = X_data_16_18.columns.tolist()
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
# ============================================================================
# ============================================================================
dict_sypmt_min_max = {'rr':[5, 40], 'heart_rate':[20,160], 'temperature':[34, 40],'sbp': [60,200], 'dbp':[40,120],
                 'Oxygen_Saturation':[60,100], 'ews':[0,21], 'UREA':[0.5, 30]}
feature_dict = {'rr':'Respiratory Rate', 'UREA':'Urea', 'sbp':'Systolic Blood Pressure',
                'dbp': 'Diastolic blood pressure', 'heart_rate': 'Heart Rate', 'temperature':'Temperature' }

features = ['rr', 'UREA', 'sbp','dbp', 'heart_rate','temperature']
feature   = features[feat_pos]

# ***************************************************
# SCALING DATA --------------------------------------
X_feat_scal_16_18 = Forecast_fn.scaling_feature_ts_(feature, X_data_16_18)
X_feat_scal_19_20 = Forecast_fn.scaling_feature_ts_(feature, X_data_19_20)

# DATA PREPARATION ----------------------------------
time_cols = [x for x in X_feat_scal_16_18.columns if ('time' in x)]# and (int(x.split('_')[1]) < 96)]
X_train = np.asarray(X_feat_scal_16_18[[x for x in time_cols if (int(x.split('_')[1]) < 96)]])
X_train = X_train.reshape((len(X_train),96,1))
y_train = np.asarray(X_feat_scal_16_18[[x for x in time_cols if (int(x.split('_')[1]) > 95)]])
y_train = y_train.reshape((len(y_train),48))
    
# ***************************************************
# MODEL CREATION LSTM--------------------------------

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# https://ai.stackexchange.com/questions/3156/how-to-select-number-of-hidden-layers-and-number-of-memory-cells-in-an-lstm
# https://hagan.okstate.edu/NNDesign.pdf#page=469

N_BLOCKS = 60
N_INPUTS = 96
N_OUTPUTS = 48 
model = Sequential()
model.add(layers.LSTM(N_BLOCKS, return_sequences=True, input_shape=(N_INPUTS, 1)))  
model.add(layers.LSTM(N_BLOCKS, recurrent_dropout=0.5, return_sequences=True))
model.add(layers.LSTM(N_BLOCKS, recurrent_dropout=0.5,))
model.add(layers.Dense(N_OUTPUTS))
model.summary()
    
# ***************************************************
# MODEL TRAINING ------------------------------------
t = time.time()
model.compile(loss="mean_absolute_error", optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size = 550, validation_split=0.3)
print("Elapsed time training the neural network:", time.time()-t)
    
# ***************************************************
# MODEL VALIDATION ----------------------------------    
ls = random.sample(range(6000),1250)
Forecast_fn.Validation_General_ts(ls, model, X_feat_scal_19_20, feature, feature_dict,'_LSTM0_') 
ls = random.sample(range(6000),250)
Forecast_fn.Validation_General_ts(ls, model, X_feat_scal_19_20, feature, feature_dict,'_LSTM1_')     
Forecast_fn.Validation_Samples_ts(ls, model, X_feat_scal_19_20, feature, feature_dict)    
results_feat = Forecast_fn.quantify_results(X_feat_scal_19_20, model, feature)    

#model.save('Results/'+feature + '_LSTM_forecasting_model.h5')
#pickle.dump( [results_feat], open('Results/'+feature+'_LSTM_forecasting_results.pickle', 'wb'))
    
# ***************************************************
# MODEL CREATION GRU --------------------------------   

# https://ai.stackexchange.com/questions/3156/how-to-select-number-of-hidden-layers-and-number-of-memory-cells-in-an-lstm
# https://hagan.okstate.edu/NNDesign.pdf#page=469

N_BLOCKS = 60
N_INPUTS = 96
N_OUTPUTS = 48 
model2 = Sequential()
model2.add(layers.GRU(N_BLOCKS, return_sequences=True, input_shape=(N_INPUTS, 1)))  
model2.add(layers.GRU(N_BLOCKS, recurrent_dropout=0.5, return_sequences=True))
model2.add(layers.GRU(N_BLOCKS, recurrent_dropout=0.5,))
model2.add(layers.Dense(N_OUTPUTS))
#model2.add(layers.Activation('relu'))    
model2.summary()
    
# ***************************************************
# MODEL TRAINING ------------------------------------
t = time.time()
model2.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
model2.fit(X_train, y_train, epochs=100, batch_size = 550, validation_split=0.3)
print("Elapsed time training the neural network:", time.time()-t)
    
# ***************************************************
# MODEL VALIDATION ----------------------------------    
ls = random.sample(range(6000),1250)
Forecast_fn.Validation_General_ts(ls, model2, X_feat_scal_19_20, feature, feature_dict,'_GRU0_') 
ls = random.sample(range(6000),250)
Forecast_fn.Validation_General_ts(ls, model2, X_feat_scal_19_20, feature, feature_dict,'_GRU1_')     
Forecast_fn.Validation_Samples_ts(ls, model2, X_feat_scal_19_20, feature, feature_dict)    
results_feat = Forecast_fn.quantify_results(X_feat_scal_19_20, model2, feature)    

#model2.save('Results/'+feature + '_GRU_forecasting_model.h5')
#pickle.dump( [results_feat], open('Results/'+feature+'_GRU_forecasting_results.pickle', 'wb'))
    
    



    
    
    
    
    
    