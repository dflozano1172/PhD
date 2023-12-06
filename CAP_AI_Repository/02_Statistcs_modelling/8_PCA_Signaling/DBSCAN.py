#%reset -f
#=============================================================================
##############################################################################
##############################################################################
# IMPORT ALL TIME PACKAGES
##############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import pickle, time, random, argparse
# sys.path is a list of absolute path strings


from sklearn.preprocessing import MinMaxScaler
# Algorithms
#from minisom import MiniSom
#from tslearn.barycenters import dtw_barycenter_averaging

from tslearn.utils import to_time_series_dataset
#from tslearn.clustering import KShape, TimeSeriesKMeans, KernelKMeans



import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)

import matplotlib
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['mathtext.fontset'] = 'cm'



t_tot = time.time()

feat_pos = 0

# parser = argparse.ArgumentParser(description='the position of the classifier in list classfiiers inside the script.')
# parser.add_argument("-p", "--print_string", help="Prints the supplied argument.",  nargs='*')
# args = parser.parse_args()
# print(args.print_string)
# feat_pos = int(args.print_string[0])

#classifiers = ['LR', 'RF', 'XGB', 'SVM', 'NN', 'LSTM', 'GRU']

features = ['rr','heart_rate','temperature','sbp', 'CREA','UREA','K', 'GFR','HCT','RBC',
               'MCV','TLYMAB','ALB', 'ALP','BILI']

feature   = features[feat_pos]

print(feature)

# ##############
# 1. LOAD DATA ---------------------------------------------------------------
# ============================================================================
t = time.time()
path = r'/home/d/dlr10/Documents/02_Statitics_modelling/DataSets/'

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


df        = X_data_16_18.copy()
lst_admns = df['admission_id'].unique().tolist()


admins_0 = X_data_16_18[X_data_16_18['Mortality'] == 0]['admission_id'].unique().tolist()
admins_1 = X_data_16_18[X_data_16_18['Mortality'] == 1]['admission_id'].unique().tolist()
samp_adm_0 = random.sample(admins_0, 350) 
samp_adm_1 = random.sample(admins_1, 350)
lst_admns  = samp_adm_0 + samp_adm_1

df        = X_data_16_18[X_data_16_18['admission_id'].isin(lst_admns)]


mort_df   = df[['admission_id', 'Mortality']].groupby(by = ['admission_id']).mean()
mort_dict = dict(zip(mort_df.index, mort_df['Mortality']))

# ============================================================================

dict_results = {}
#for feature in dict_labels.keys():
print("")
print('-------------------------------')
print(feature)
print("")
# Array with the ts of each admission of the defined feature
X_feat_ts = np.array(df[feature]).reshape((len(lst_admns), int(len(df) / len(lst_admns))))

t = time.time()
mySeries = []
for i in range(X_feat_ts.shape[0]):
    scaler = MinMaxScaler()
    mySeries.append((MinMaxScaler().fit_transform(X_feat_ts[i,:].reshape(-1,1)).T).tolist()[0])
mySeries_pd = pd.DataFrame(mySeries)
X_bis       = to_time_series_dataset(mySeries)
print("Maxmin Transform and data serialisation:", time.time()-t)

# ============================================================================
from sklearn.cluster import DBSCAN
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def dbscansim(ts1, ts2):
    distance, _ = fastdtw(ts1, ts2, dist=euclidean)
    return distance

t = time.time()
clustering = DBSCAN(eps=300, min_samples=2, metric = dbscansim).fit(X_bis.reshape(X_bis.shape[0], X_bis.shape[1]))
print("elapsed", time.time()-t)




























