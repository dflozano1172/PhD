#%reset -f
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
import matplotlib.pyplot as plt

import re, sqlite3, pickle, time, datetime, random, sys


############################################################################
# 0 SUB FUNCTIONS ##########################################################
############################################################################

# ========================================================================================
# This function finds the value of the first row of a column in a specific dataframe
# It is mainly designed to extract information of one admission from df_admissions
# and extract the information from df_patients
def find_columns_in_admdf_by_adm(admin, column, df_admns, field):
    admisn_ = df_admns[df_admns[field] == admin].copy()
    outcome = admisn_.iloc[0][column]
    return outcome


# ========================================================================================
# This function extract the peaks and troughs from the second to the one before the last 
# columns. It returns a dataframe of twice the number of columns with peaks and troughs in
# the labels.
def Extract_peaks_trough(X_data):
    # Getting in place labels 
    feat_Xdata      = X_data.columns.tolist()
    feat_Xdata_peak = [s + '_Peak' for s in feat_Xdata[1:-1]]
    feat_Xdata_trou = [s + '_Trough' for s in feat_Xdata[1:-1]]
    feat_Xdata_pnt  = ['admission_id'] + [x for pair in zip(feat_Xdata_peak,feat_Xdata_trou) for x in pair]
    # Extract the information of peaks and torughs
    X_data_copy = X_data.copy()
    X_data      = pd.DataFrame(columns = feat_Xdata_pnt)
    for idx, admin in enumerate(X_data_copy['admission_id'].unique().tolist()):
        #print("in", len(X_data_copy))
        row    = [admin]
        df_adm = X_data_copy[X_data_copy['admission_id'] == admin].copy()
        for feature in feat_Xdata[1:-1]:
            max_feat = df_adm[feature].max()
            min_feat = df_adm[feature].min()
            row = row + [max_feat, min_feat]
        X_data.at[len(X_data)] =  row
        X_data_copy = X_data_copy.drop(index = df_adm.index)
    return X_data

############################################################################
# 1  LOAD DATAFRAMES #######################################################
# df_patients, df_admissions, df_eobs_mixed 
############################################################################
def Load_data(years = '2016_2018'):
    DataFrame_path = '/home/d/dlr10/Documents/02_Statitics_modelling/DataFrames/'
    
    #------------------------- LOAD DATA -------------------------
    # Load DataFrames: df_patients, df_admissions, df_eobs_mix
    
    df_patients, df_admissions = pickle.load( open( DataFrame_path + years + '/df_patients_admissions_v4.pickle', 'rb'))
    df_eobs = pickle.load(open( DataFrame_path + years + '/df_eobs_heam_mixed_V2.pickle', 'rb'))[0]
    df_eobs = df_eobs.reset_index(drop=True)
    df_eobs['no_sample_series'] = df_eobs.apply(lambda x : x.name%144, axis = 1 )
    return df_patients, df_admissions, df_eobs

############################################################################
# 2 DATAFRAME EXTRACTION ###################################################
############################################################################


#------------------------- PARAMETERS OF EXTRACTION -------------------------
#timeSeries      = True
#peakstroughs    = False
#samp_to_extract = 0

# ----- extracting information from eobs -----
def Exctract_Xdata(df_patients, df_admissions, df_eobs, samp_to_extract, peakstroughs, timeSeries ):
    X_data = df_eobs.copy()
    X_data = X_data.rename({'temperature\r':'temperature'}, axis = 'columns' )
    X_data = X_data.drop(columns = ['time_since_prev_obs_in_mins', 'timestamp', 'timestamp_code'])
    if timeSeries == True:
        peakstroughs = False   
        # Extract the information of peaks and torughs
        X_data_copy = X_data.copy()
        X_data      = pd.DataFrame(columns = X_data.columns)
        for idx, admin in enumerate(X_data_copy['admission_id'].unique().tolist()):
            df_adm = X_data_copy[X_data_copy['admission_id'] == admin].copy()
            df_adm['sex']       = find_columns_in_admdf_by_adm(find_columns_in_admdf_by_adm(admin, 'patient_id', df_admissions, 'admission_id'), 'sex', df_patients, 'patient_id')
            df_adm['ethnicity'] = find_columns_in_admdf_by_adm(find_columns_in_admdf_by_adm(admin, 'patient_id', df_admissions, 'admission_id'), 'ethnic_origin', df_patients, 'patient_id')
            # ----- Retrieving information from admissions -----
            df_adm['age_at_admin'] = find_columns_in_admdf_by_adm(admin, 'age_at_admission\r', df_admissions, 'admission_id')
            df_adm['Comorb_score'] = find_columns_in_admdf_by_adm(admin, 'Comorbidity_score', df_admissions, 'admission_id')
            df_adm['Spcfc_Comorb'] = find_columns_in_admdf_by_adm(admin, 'Specific Comorbidity', df_admissions, 'admission_id')
            df_adm['Mortality']    = find_columns_in_admdf_by_adm(admin, 'mortal_admin', df_admissions, 'admission_id')
            
            X_data = pd.concat([X_data, df_adm])
        X_data['rr']        = pd.to_numeric(X_data['rr'], downcast = 'float')
        X_data['ews']       = pd.to_numeric(X_data['ews'], downcast = 'float')
        X_data['Confusion'] = pd.to_numeric(X_data['Confusion'], downcast = 'float')
    else:
        if peakstroughs == False:
            X_data              = X_data[X_data['no_sample_series'] == samp_to_extract].copy()
            X_data['rr']        = pd.to_numeric(X_data['rr'])
            X_data['ews']       = pd.to_numeric(X_data['ews'])
            X_data['Confusion'] = pd.to_numeric(X_data['Confusion'])
            X_data              = X_data.drop(columns = ['no_sample_series'])
        else:
            X_data = X_data[(48 * samp_to_extract <= X_data['no_sample_series']) &
                    (X_data['no_sample_series'] < 48 * (samp_to_extract+1))]
            X_data = Extract_peaks_trough(X_data)
            
        # ----- Retrieving information from patients -----
        X_data['sex']       = X_data['admission_id'].apply(lambda x: find_columns_in_admdf_by_adm(find_columns_in_admdf_by_adm(x, 'patient_id', df_admissions, 'admission_id'), 'sex', df_patients, 'patient_id'))
        X_data['ethnicity'] = X_data['admission_id'].apply(lambda x: find_columns_in_admdf_by_adm(find_columns_in_admdf_by_adm(x, 'patient_id', df_admissions, 'admission_id'), 'ethnic_origin', df_patients, 'patient_id'))
        # ----- Retrieving information from admissions -----
        X_data['age_at_admin'] = X_data['admission_id'].apply(lambda x: find_columns_in_admdf_by_adm(x, 'age_at_admission\r', df_admissions, 'admission_id'))
        X_data['Comorb_score'] = X_data['admission_id'].apply(lambda x: find_columns_in_admdf_by_adm(x, 'Comorbidity_score', df_admissions, 'admission_id'))
        X_data['Spcfc_Comorb'] = X_data['admission_id'].apply(lambda x: find_columns_in_admdf_by_adm(x, 'Specific Comorbidity', df_admissions, 'admission_id'))
        X_data['had_Prev_admin'] = X_data['admission_id'].apply(lambda x: find_columns_in_admdf_by_adm(x, 'had_Prev_admin', df_admissions, 'admission_id'))
        X_data['Mortality']    = X_data['admission_id'].apply(lambda x: find_columns_in_admdf_by_adm(x, 'mortal_admin', df_admissions, 'admission_id'))
    X_data = X_data.rename({'temperature\r':'temperature'}, axis = 'columns' )
    return X_data
    
############################################################################

#timeSeries      = True
#peakstroughs    = False
#samp_to_extract = 0
#df_patients, df_admissions, df_eobs = Load_data()
#Exctract_Xdata(df_patients, df_admissions, df_eobs, samp_to_extract, peakstroughs, timeSeries  )













