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

import Functions_Clustering as FC
from sklearn.preprocessing import MinMaxScaler
# Algorithms
#from minisom import MiniSom
from tslearn.barycenters import dtw_barycenter_averaging

from tslearn.utils import to_time_series_dataset
from tslearn.clustering import KShape, TimeSeriesKMeans, KernelKMeans



import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)

import matplotlib
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['mathtext.fontset'] = 'cm'



t_tot = time.time()

feat_pos = 1

parser = argparse.ArgumentParser(description='the position of the classifier in list classfiiers inside the script.')
parser.add_argument("-p", "--print_string", help="Prints the supplied argument.",  nargs='*')
args = parser.parse_args()
print(args.print_string)
feat_pos = int(args.print_string[0])

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
# ============================================================================
# ############## -------------------------------------------------------------
# Variables that were significant in both datasets
# ============================================================================

dict_labels = {'rr':'Respiratory rate','heart_rate':'Heart rate','temperature':'Temperature',
               'sbp':'Systolic bp', 'CREA':'Creatinine','UREA':'Urea','K':'Potassium', 'GFR':'GFR',
               'HCT':'HCT','HGB':'Haemoglobin','RBC':'RBC','MCV':'MCV','TLYMAB':'T-lymphocite Ab','ALB':'Albumin',
               'ALP':'Alk. Phosph.','BILI':'Bilirubin'}

df        = X_data_16_18.copy()
lst_admns = df['admission_id'].unique().tolist()


# admins_0 = X_data_16_18[X_data_16_18['Mortality'] == 0]['admission_id'].unique().tolist()
# admins_1 = X_data_16_18[X_data_16_18['Mortality'] == 1]['admission_id'].unique().tolist()
# samp_adm_0 = random.sample(admins_0, 300) 
# samp_adm_1 = random.sample(admins_1, 300)
# lst_admns  = samp_adm_0 + samp_adm_1

# df        = X_data_16_18[X_data_16_18['admission_id'].isin(lst_admns)]


mort_df   = df[['admission_id', 'Mortality']].groupby(by = ['admission_id']).mean()
mort_dict = dict(zip(mort_df.index, mort_df['Mortality']))

t1 = time.time()

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
dict_nclusters = {}
for n_ in [4,5,6,7]:
    print('starting ', n_, ' -----------------')
    som_x   = som_y = n_; n_clust = som_x * som_y
    
    # ============================================================================
    # SOM CLustering *************************************************************        
    som, df_results = FC.run_SOM_clustering(mySeries,lst_admns, mort_dict,som_x, som_y)
    win_map         = som.win_map(mySeries)
    FC.plot_som_series_dba_center(som_x, som_y, win_map, dict_labels[feature])
    FC.get_barplot_cluster_maps(df_results, 'clusterSOM', dict_labels[feature],  n_clust)
    
    # ============================================================================
    # K MEANS ********************************************************************  
    
    t = time.time()
    km     = TimeSeriesKMeans(n_clusters=n_clust, max_iter=500000, metric="dtw", random_state=0)
    y_pred_km = km.fit_predict(X_bis)
    df_results['clusterKM'] = y_pred_km
    FC.plot_KmnKs_series_centre(X_bis, y_pred_km, km, som_x, som_y, n_clust, dict_labels[feature], 'KM')
    FC.get_barplot_cluster_maps(df_results, 'clusterKM', dict_labels[feature],  n_clust)
    print("elapsed KM", time.time()-t)
    
    # ============================================================================
    # Kernel K MEANS *************************************************************  
    # t = time.time()
    # gak_kkm    = KernelKMeans(n_clusters=n_clust, kernel="gak", kernel_params={"sigma": "auto"}, n_init=20)
    # y_pred_kkm = gak_kkm.fit_predict(X_bis)
    # df_results['clusterKKM'] = y_pred_kkm
    # FC.plot_KmnKs_series_centre(X_bis, y_pred_kkm, gak_kkm, som_x, som_y, n_clust, dict_labels[feature], 'KKM')
    # FC.get_barplot_cluster_maps(df_results, 'clusterKKM', dict_labels[feature])
    # print("elapsed KKm", time.time()-t)
    
    # ============================================================================
    # K SHAPE ********************************************************************  
    
    t = time.time()
    ks = KShape(n_clusters=n_clust, n_init=1, random_state=0)
    y_pred_ks = ks.fit_predict(X_bis)
    df_results['clusterKS'] = y_pred_ks
    
    FC.plot_KmnKs_series_centre(X_bis, y_pred_ks, ks, som_x, som_y, n_clust, dict_labels[feature], 'KS')
    FC.get_barplot_cluster_maps(df_results, 'clusterKS', dict_labels[feature],  n_clust)
    print("elapsed KS", time.time()-t)
    
    # ============================================================================

    abbvs =  ['SOM', 'KM',  'KS'] #'KKM',
    df_clusters = pd.DataFrame()   
    for abv in abbvs:
        df_clusters['Total'+abv] = df_results.groupby(by= ['cluster'+abv]).count()[['admission_id']]
        df_clusters['deceased'+abv] = df_results[df_results['mortality'] ==1 ].groupby(by= ['cluster'+abv]).count()['admission_id']
        df_clusters['per_'+abv] = df_clusters['deceased'+abv]/df_clusters['Total'+abv]
    
    dict_nclusters[n_] = [df_results, df_clusters]
    #break
dict_results[feature] = dict_nclusters

pickle.dump(dict_results, open('Sub_results/results_clustering_'+ feature +'.pickle', 'wb'))
print("elapsed TOTAL", time.time()-t1)

#https://github.com/JustGlowing/minisom/blob/master/examples/Clustering.ipynb
#https://www.diva-portal.org/smash/get/diva2:1537700/FULLTEXT01.pdf



























