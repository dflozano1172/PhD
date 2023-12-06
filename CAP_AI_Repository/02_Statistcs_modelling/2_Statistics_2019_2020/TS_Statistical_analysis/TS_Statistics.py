
import pandas as pd
import numpy as np
#
import time, pickle

import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller

import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def t_test_p_val(mean_mort, mean_disc):
    rat_var = np.var(mean_disc)/ np.var(mean_mort)
    if (0.25 < rat_var) and (rat_var < 4):
        a = stats.ttest_ind(a=mean_mort, b=mean_disc, equal_var=True)
    else:
        a = stats.ttest_ind(a=mean_mort, b=mean_disc, equal_var=False)
    return a[1]
#import matplotlib.pyplot as plt

path = r'/home/d/dlr10/Documents/02_Statitics_modelling/DataSets/'

X_data_16_18 = pickle.load(open(path + 'df_ts_2016_18.pickle','rb'))
X_data_19_20 = pickle.load(open(path + 'df_ts_2019_20.pickle','rb'))

admn_train = X_data_16_18['admission_id'].unique().tolist()
admn_valid = X_data_19_20['admission_id'].unique().tolist()

features = ['rr', 'heart_rate', 'temperature', 'sbp', 'dbp', 'Oxygen_Saturation', 'CREA', 'UREA', 'K',
       'GFR', 'WBC', 'PLT', 'HCT', 'HGB', 'RBC', 'MCH', 'MCV', 'NEUAB',
       'TLYMAB', 'EOSAB', 'MONAB', 'BASAB', 'ALB', 'ALP', 'BILI']

#feature = 'rr'

dict_df_stats = {}
p_val_result  = []

for feature in features:
    
    t = time.time()
    df_test = []
    df_ts_1dff = []
    df_ts_2dff = []
    for idx, adm in enumerate(admn_valid):
        df  = X_data_19_20[X_data_19_20['admission_id'] ==  adm ]
        t_series = df[feature]
        
        adf_test = adfuller(t_series)
        p_val_1 = adf_test[1]
        
        df_diff = t_series.diff().dropna()
        adf_test_1 = adfuller(df_diff)
        p_val_2 = adf_test_1[1]
        
        df_diff_1 = df_diff.diff().dropna()
        adf_test_2 = adfuller(df_diff_1)
        p_val_3 = adf_test_2[1]
        
        df_test.append([adm, df_diff.mean(), p_val_1, df_diff_1.mean(), p_val_2, p_val_3, df['Mortality'].mean()])
        df_ts_1dff.append(df_diff.tolist())
        df_ts_2dff.append(df_diff_1.tolist())    

    print('elapsed:', time.time() - t)
        
    df_statistics = pd.DataFrame(df_test, columns = ['admn', 'mean_diff1','p_val1', 'mean_diff2', 'p_val2', 'p_val3', 'Mortality'])
    dict_df_stats[feature] = df_statistics
    
    df_       = df_statistics[(df_statistics['p_val2'] < 0.05)]
    mean_mort_ = df_[(df_['Mortality'] == 1)]['mean_diff1'].to_list()
    mean_disc_ = df_[(df_['Mortality'] == 0)]['mean_diff1'].to_list()
    m1_,CI11_,CI12_ = mean_confidence_interval(mean_mort_, confidence=0.95)
    m2_,CI21_,CI22_ = mean_confidence_interval(mean_disc_, confidence=0.95)
    p_val_1   = t_test_p_val(mean_mort_, mean_disc_)
    
    df_1      = df_statistics[(df_statistics['p_val3'] < 0.05)]
    mean_mort = df_1[(df_1['Mortality'] == 1)]['mean_diff2'].to_list()
    mean_disc = df_1[(df_1['Mortality'] == 0)]['mean_diff2'].to_list()
    m1,CI11,CI12 = mean_confidence_interval(mean_mort, confidence=0.95)
    m2,CI21,CI22 = mean_confidence_interval(mean_disc, confidence=0.95)
    p_val_2   = t_test_p_val(mean_mort, mean_disc)
        
    p_val_result.append([feature, 
                         m1_, CI11_,CI12_, 
                         m2_, CI21_,CI22_, p_val_1, 
                         m1, CI11,CI12,
                         m2, CI21,CI22, p_val_2])


pickle.dump([p_val_result], open('p_val_timeseries.pickle', 'wb'))





