#=============================================================================
##############################################################################
##############################################################################
# IMPORT ALL TIME PACKAGES
##############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import pickle, time,  sys, warnings
# sys.path is a list of absolute path strings
sys.path.append('/home/d/dlr10/Documents/02_Statitics_modelling/0_FunctionsScripts')
import Loading_Data_Functions as load_fn
import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)

##############################################################################
##############################################################################
# CODE STARTS
##############################################################################

t_tot = time.time()
# ##############
# 1. LOAD DATA ---------------------------------------------------------------
# ============================================================================
t = time.time()
path = r'/home/d/dlr10/Documents/02_Statitics_modelling/DataSets/'
df_patients_16_18, df_admissions_16_18, df_eobs_16_18 = load_fn.Load_data()
df_patients_19_20, df_admissions_19_20, df_eobs_19_20 = load_fn.Load_data('2019_2020')

X_data_16_18 = pickle.load(open(path + 'df_ts_2016_18.pickle','rb')).drop(columns = 'no_sample_series')
X_data_19_20 = pickle.load(open(path + 'df_ts_2019_20.pickle','rb')).drop(columns = 'no_sample_series')


# Dictionary of features and types --------
data_types = pd.read_csv('/home/d/dlr10/Documents/02_Statitics_modelling/2_Statistics/csv_Type_variables.csv')
data_types = data_types.set_index('Variable').to_dict()['Type']

# List of features ------------------------
feat_list = X_data_16_18.columns.tolist()
feat_list = feat_list[1:-1]

# Resulst vaiables ------------------------
df_results_train = pd.DataFrame()
df_results_valid = pd.DataFrame()
dict_fpr_tpr = {}

print("Elapsed time loading data:", time.time()-t)
# ============================================================================

##############################################################################
##############################################################################
# SELECT FEATURE
##############################################################################

feature   = 'heart_rate'

df        = X_data_16_18[feature]
lst_admns = X_data_16_18['admission_id'].unique().tolist()
mort_df   = X_data_16_18[['admission_id', 'Mortality']].groupby(by = ['admission_id']).mean()
mort_dict = dict(zip(mort_df.index, mort_df['Mortality']))
lst_mort  = [mort_dict[i] for i in lst_admns]

# Array with the ts of each admission of the defined feature
X_feat_ts = np.array(df).reshape((len(lst_admns), int(len(df) / len(lst_admns))))
print("Shape of X", X_feat_ts.shape)

#X_feat_ts = X_feat_ts[:10,:].copy()
#display(X_feat_ts.shape)

##############################################################################
##############################################################################
# Divide timeseries into windows
##############################################################################
print("============= Window division =================")
time_window_size = 36 # timesteps
time_overlap     = 12 # timesteps

time_step = (time_window_size - time_overlap)

ls_ts_seg = []
idx_ts_sg = []
for i in range(X_feat_ts.shape[1] // time_step):
    lo_bd = i *time_step
    up_bd = lo_bd +  time_window_size if lo_bd +  time_window_size <=  X_feat_ts.shape[1] else  X_feat_ts.shape[1] 
    print(i,  lo_bd, up_bd)
    idx_ts_sg.append([lo_bd, up_bd])
    ls_ts_seg.append(X_feat_ts[:,lo_bd: up_bd])
    
##############################################################################
##############################################################################
# CALCULATE THE SIMILARITY MATRIX
##############################################################################    
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

t = time.time()
dim = len(ls_ts_seg) * X_feat_ts.shape[0]
corr_matrix = np.zeros((dim, dim))
print("============= Dynamic Time Wraping =================")
for window in  range(len(ls_ts_seg)):
    ls_seg = ls_ts_seg[window]
    for idx, ts1 in enumerate(ls_seg):
        for window_2 in  range(len(ls_ts_seg)):
            ls_seg_2 = ls_ts_seg[window_2]
            for idx_2, ts2 in enumerate(ls_seg_2):
                distance, path = fastdtw(ts1, ts2, dist=euclidean)
                
                i = (window * len(ls_seg)) + idx
                j = (window_2 * len(ls_seg_2)) + idx_2
                
                corr_matrix[i][j] = distance
                #print(i,j, distance)
print("time elapsed:", time.time() - t)
##############################################################################
##############################################################################
# SAVE THE CORR MATRIX
##############################################################################   
print("============= Saving the Pickle =================")
pickle.dump(corr_matrix, open(r'my_corr_matrix_dtw.pickle',"wb"))

##############################################################################
##############################################################################
# GENERATE HEAT MAP
##############################################################################   
print("============= Generating Heat Map =================")
plt.imshow(corr_matrix, cmap='hot')
cax = plt.axes([0.85, 0.1, 0.075, 0.8])
plt.colorbar(cax=cax)
plt.savefig('correlation_heatmap.png', transparent = True, bbox_inches = "tight")
plt.show()


























