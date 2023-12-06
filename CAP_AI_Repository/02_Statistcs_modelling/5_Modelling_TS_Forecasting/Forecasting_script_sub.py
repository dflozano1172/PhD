
import pandas as pd
import numpy as np
import time, pickle, random, argparse

import Functions_forecasting as Forecast_fn

from bayes_opt import BayesianOptimization

import matplotlib


matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['mathtext.fontset'] = 'cm'



feat_pos = 1

parser = argparse.ArgumentParser(description='the position of the classifier in list classfiiers inside the script.')
parser.add_argument("-p", "--print_string", help="Prints the supplied argument.",  nargs='*')
args = parser.parse_args()
print(args.print_string)
feat_pos = int(args.print_string[0])



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
print(feature)

# ***************************************************
# SCALING DATA --------------------------------------
X_feat_scal_16_18 = Forecast_fn.scaling_feature_ts_global(feature, X_data_16_18)
X_feat_scal_19_20 = Forecast_fn.scaling_feature_ts_global(feature, X_data_19_20)

# DATA PREPARATION ----------------------------------
time_cols = [x for x in X_feat_scal_16_18.columns if ('time' in x)]# and (int(x.split('_')[1]) < 96)]
X_train = np.asarray(X_feat_scal_16_18[[x for x in time_cols if (int(x.split('_')[1]) < 96)]])
X_train = X_train.reshape((len(X_train),96,1))
y_train = np.asarray(X_feat_scal_16_18[[x for x in time_cols if (int(x.split('_')[1]) > 95)]])
y_train = y_train.reshape((len(y_train),48))
    
X_train = X_train.astype(float)
y_train = y_train.astype(float)
# ***************************************************
# MODEL CREATION LSTM--------------------------------

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# https://ai.stackexchange.com/questions/3156/how-to-select-number-of-hidden-layers-and-number-of-memory-cells-in-an-lstm
# https://hagan.okstate.edu/NNDesign.pdf#page=469

N_BLOCKS = 3
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
model.fit(X_train, y_train, epochs=100, batch_size = 1550, validation_split=0.3)
print("Elapsed time training the neural network:", time.time()-t)
    
# ***************************************************
# MODEL VALIDATION ----------------------------------    
ls = random.sample(range(6000),1250)
Forecast_fn.Validation_General_ts(ls, model, X_feat_scal_19_20, feature, feature_dict,'_LSTM0_', True) 
ls = random.sample(range(6000),250)
Forecast_fn.Validation_General_ts(ls, model, X_feat_scal_19_20, feature, feature_dict,'_LSTM1_', True)     
Forecast_fn.Validation_Samples_ts(ls, model, X_feat_scal_19_20, feature, feature_dict,'_LSTM_', True)
results_feat = Forecast_fn.quantify_results(X_feat_scal_19_20, model, feature)    

model.save('Results/'+feature + '_LSTM_forecasting_model.h5')
pickle.dump( [results_feat], open('Results/'+feature+'_LSTM_forecasting_results.pickle', 'wb'))
    
# ***************************************************
# MODEL CREATION GRU --------------------------------   

# https://ai.stackexchange.com/questions/3156/how-to-select-number-of-hidden-layers-and-number-of-memory-cells-in-an-lstm
# https://hagan.okstate.edu/NNDesign.pdf#page=469

N_BLOCKS = 3
N_INPUTS = 96
N_OUTPUTS = 48 
model2 = Sequential()
model2.add(layers.GRU(N_BLOCKS, return_sequences=True, input_shape=(N_INPUTS, 1)))  
model2.add(layers.GRU(N_BLOCKS, recurrent_dropout=0.5, return_sequences=True))
model2.add(layers.GRU(N_BLOCKS, recurrent_dropout=0.5,))
model2.add(layers.Dense(N_OUTPUTS)) 
model2.summary()


    
# ***************************************************
# MODEL TRAINING ------------------------------------
t = time.time()
model2.compile(loss="mean_absolute_error", optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size = 1550, validation_split=0.3)
print("Elapsed time training the neural network:", time.time()-t)
    
# ***************************************************
# MODEL VALIDATION ----------------------------------    
ls = random.sample(range(6000),1250)
Forecast_fn.Validation_General_ts(ls, model2, X_feat_scal_19_20, feature, feature_dict,'_GRU0_', True) 
ls = random.sample(range(6000),250)
Forecast_fn.Validation_General_ts(ls, model2, X_feat_scal_19_20, feature, feature_dict,'_GRU1_', True)     
Forecast_fn.Validation_Samples_ts(ls, model2, X_feat_scal_19_20, feature, feature_dict,'_GRU_', True)    
results_feat = Forecast_fn.quantify_results(X_feat_scal_19_20, model2, feature)

model2.save('Results/'+feature + '_GRU_forecasting_model.h5')
pickle.dump( [results_feat], open('Results/'+feature+'_GRU_forecasting_results.pickle', 'wb'))
# ============================================================================    
# ============================================================================
# ============================================================================
# ============================================================================    
# ============================================================================
# ##############
# 3. BAYESIAN OPTIMIZER TO FINE TUNE THE NEURAL NETWORK 
# ============================================================================
from sklearn.model_selection import cross_val_score
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
N_INPUTS = 96
N_OUTPUTS = 48 




# ============================================================================
# GRU
# ============================================================================
print("start BO GRU")
def nn_cl_bo2(cellsIni, cells1, cells2, learning_rate, batch_size, epochs):
    cellsIni = round(cellsIni)
    cells1 = round(cells1)
    cells2 = round(cells2)
    batch_size = round(batch_size)
    epochs     = round(epochs)
    def nn_GRU_fun():      

        nn = Sequential()        
        nn.add(layers.GRU(cellsIni, return_sequences=True, input_shape=(N_INPUTS, 1)))  
        nn.add(layers.GRU(cells1, recurrent_dropout=0.5, return_sequences=True))
        nn.add(layers.GRU(cells2, recurrent_dropout=0.5,))        
        nn.add(layers.Dense(N_OUTPUTS))
        
        nn.compile(optimizer="adam", loss="mean_absolute_error", metrics=["mae"])
        return nn #model
    nn    = KerasRegressor(build_fn=nn_GRU_fun, epochs=epochs, batch_size=batch_size, verbose=0)
    
    #kfold = StratifiedKFold(n_splits=8, shuffle=True, random_state=123)
    score = cross_val_score(nn, X_train, y_train, scoring = 'neg_median_absolute_error').mean()
    return score

params_nn2 ={'cellsIni':(10,40), 'cells1':(0,20), 'cells2':(0, 20),
             'learning_rate':(0.01, 0.3),
             'batch_size':(600, 1500),
             'epochs':(20, 30)}

# Run Bayesian Optimization
t = time.time()
nn_bo = BayesianOptimization(nn_cl_bo2, params_nn2, random_state=111)
#nn_bo.maximize(init_points=20, n_iter=10)
nn_bo.maximize(init_points=5, n_iter=20)
print("elapsed Fine tuning the NN", time.time()-t)

# ##############
# 4. EXTRACTING RESULTS FROM FINE-TUNING

params_nn_ = nn_bo.max['params']
    

params_nn_['cellsIni']   = round(params_nn_['cellsIni'])
params_nn_['cells1']     = round(params_nn_['cells1'])
params_nn_['cells2']     = round(params_nn_['cells2'])
params_nn_['batch_size']  = round(params_nn_['batch_size'])
params_nn_['epochs']      = round(params_nn_['epochs'])
# ============================================================================
   # FIT CLASSIFIER WITH FINE TUNE GHYPERPARAMETERS
# ============================================================================

cellsIni      = params_nn_['cellsIni']
cells1        = params_nn_['cells1']
cells2        = params_nn_['cells2']
batch_size    = params_nn_['batch_size']
epochs        = params_nn_['epochs']
learning_rate = params_nn_['learning_rate']
print("")
print("RESULTS OF MODEL BO GRU")
print(cellsIni,  cells1, cells2, learning_rate, batch_size, epochs)
def ft_cl_fun(cellsIni,  cells1, cells2, learning_rate, batch_size, epochs):    
    nn = Sequential()        
    nn.add(layers.GRU(cellsIni, return_sequences=True, input_shape=(N_INPUTS, 1)))  
    nn.add(layers.GRU(cells1, recurrent_dropout=0.5, return_sequences=True))
    nn.add(layers.GRU(cells2, recurrent_dropout=0.5,))        
    nn.add(layers.Dense(N_OUTPUTS))
    nn.compile(optimizer="adam", loss="mean_absolute_error", metrics=["mae"])
    return nn

t = time.time()
clf_model = ft_cl_fun(cellsIni, cells1, cells2, learning_rate, batch_size, epochs)
history = clf_model.fit(X_train, y_train, epochs=epochs, batch_size = batch_size, validation_split=0.3)


ls = random.sample(range(6000),1250)
Forecast_fn.Validation_General_ts(ls, clf_model, X_feat_scal_19_20, feature, feature_dict,'_BOGRU0_', True) 
ls = random.sample(range(6000),250)
Forecast_fn.Validation_General_ts(ls, clf_model, X_feat_scal_19_20, feature, feature_dict,'_BOGRU1_', True)     
Forecast_fn.Validation_Samples_ts(ls, clf_model, X_feat_scal_19_20, feature, feature_dict,'_BOGRU_', True)    
results_feat = Forecast_fn.quantify_results(X_feat_scal_19_20, clf_model, feature)   

clf_model.save('Results/'+feature + '_BOGRU_forecasting_model.h5')
pickle.dump( [results_feat], open('Results/'+feature+'_BOGRU_forecasting_results.pickle', 'wb'))
print("elapsed fitting the fine tuned BO_GRU", time.time()-t)
# ============================================================================

# ============================================================================
# LSTM
# ============================================================================
print("start BO LSTM")
def nn_cl_bo2(cellsIni, cells1, cells2, learning_rate, batch_size, epochs):
    cellsIni = round(cellsIni)
    cells1 = round(cells1)
    cells2 = round(cells2)
    batch_size = round(batch_size)
    epochs     = round(epochs)
    def nn_LSTM_fun():      
        nn = Sequential()        
        nn.add(layers.LSTM(cellsIni, return_sequences=True, input_shape=(N_INPUTS, 1)))  
        nn.add(layers.LSTM(cells1, recurrent_dropout=0.5, return_sequences=True))
        nn.add(layers.LSTM(cells2, recurrent_dropout=0.5,))        
        nn.add(layers.Dense(N_OUTPUTS))
        
        nn.compile(optimizer="adam", loss="mean_absolute_error", metrics=["mae"])
        return nn #model
    nn    = KerasRegressor(build_fn=nn_LSTM_fun, epochs=epochs, batch_size=batch_size, verbose=0)
    #kfold = StratifiedKFold(n_splits=8, shuffle=True, random_state=123)
    score = cross_val_score(nn, X_train, y_train, scoring = 'neg_median_absolute_error').mean()
    return score

params_nn2 ={'cellsIni':(10,40), 'cells1':(0,20), 'cells2':(0, 20),
             'learning_rate':(0.01, 0.3),
             'batch_size':(600, 1500),
             'epochs':(20, 30)}

# Run Bayesian Optimization
t = time.time()
nn_bo = BayesianOptimization(nn_cl_bo2, params_nn2, random_state=111)
#nn_bo.maximize(init_points=20, n_iter=10)
nn_bo.maximize(init_points=6, n_iter=20)
print("elapsed Fine tuning the NN", time.time()-t)

# ##############
# 4. EXTRACTING RESULTS FROM FINE-TUNING

params_nn_ = nn_bo.max['params']
    

params_nn_['cellsIni']   = round(params_nn_['cellsIni'])
params_nn_['cells1']     = round(params_nn_['cells1'])
params_nn_['cells2']     = round(params_nn_['cells2'])
params_nn_['batch_size']  = round(params_nn_['batch_size'])
params_nn_['epochs']      = round(params_nn_['epochs']) 
# ##############
# FIT CLASSIFIER WITH FINE TUNE GHYPERPARAMETERS
# ============================================================================

cellsIni      = params_nn_['cellsIni']
cells1        = params_nn_['cells1']
cells2        = params_nn_['cells2']
batch_size    = params_nn_['batch_size']
epochs        = params_nn_['epochs']
learning_rate = params_nn_['learning_rate']
print("")
print("RESULTS OF MODEL BO GRU")
print(cellsIni,  cells1, cells2, learning_rate, batch_size, epochs)
def ft_LSTM_fun(cellsIni,  cells1, cells2, learning_rate, batch_size, epochs):    
    nn = Sequential()        
    nn.add(layers.LSTM(cellsIni, return_sequences=True, input_shape=(N_INPUTS, 1)))  
    nn.add(layers.LSTM(cells1, recurrent_dropout=0.5, return_sequences=True))
    nn.add(layers.LSTM(cells2, recurrent_dropout=0.5,))        
    nn.add(layers.Dense(N_OUTPUTS))
    nn.compile(optimizer="adam", loss="mean_absolute_error", metrics=["mae"])
    return nn

t = time.time()
clf_model_LSTM = ft_LSTM_fun(cellsIni, cells1, cells2, learning_rate, batch_size, epochs)
history = clf_model_LSTM.fit(X_train, y_train, epochs=epochs, batch_size = batch_size, validation_split=0.3)


ls = random.sample(range(6000),1250)
Forecast_fn.Validation_General_ts(ls, clf_model_LSTM, X_feat_scal_19_20, feature, feature_dict,'_BOLSTM0_', True) 
ls = random.sample(range(6000),250)
Forecast_fn.Validation_General_ts(ls, clf_model_LSTM, X_feat_scal_19_20, feature, feature_dict,'_BOLSTM1_', True)     
Forecast_fn.Validation_Samples_ts(ls, clf_model_LSTM, X_feat_scal_19_20, feature, feature_dict,'_BOLSTM_', True)       
results_feat = Forecast_fn.quantify_results(X_feat_scal_19_20, clf_model_LSTM, feature)   

clf_model_LSTM.save('Results/'+feature + '_BOLSTM_forecasting_model.h5')
pickle.dump( [results_feat], open('Results/'+feature+'_BOLSTM_forecasting_results.pickle', 'wb'))
print("elapsed fitting the fine tuned BO_LSTM", time.time()-t)
# ============================================================================


    
    
    
    
    

    
    
    
    
    
    