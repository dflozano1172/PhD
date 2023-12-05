
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re, sqlite3, pickle, time, datetime, random

from sklearn.model_selection import LeaveOneOut
import os

t = time.time()


min_adm = 17410
max_adm = 17415
"""
import argparse
parser = argparse.ArgumentParser(description='A test program.')
parser.add_argument("-p", "--print_string", help="Prints the supplied argument.",  nargs='*')
args = parser.parse_args()
min_adm = int(args.print_string[0])
max_adm = int(args.print_string[1])
#print(args.print_string)
"""
##############################################################################
#######################################
### Relevant fields for pre-processing
#######################################
reference_date = datetime.datetime(1970,1,1)

##### Fields in table patients ##########
patient_field         = 'patient_id'
age_field             = 'age_at_admission\r'
sex_field             = 'sex'
ethnic_field          = 'ethnic_origin'
death_ind_field       = 'death_indicator'
death_date_field      = 'date_of_death'
death_date_code_field = 'date_code_of_death'
mort_in_hosp_field    = 'Thirty_day_mort'
patients_cols = [patient_field,sex_field, ethnic_field, death_ind_field, death_date_field, death_date_code_field, 
                 mort_in_hosp_field]

##### Fields in table admissions ########
admission_field           = 'admission_id'
diagnosis_field           = 'episode_diagnoses'
admn_date_field           = 'admission_date_time'
admn_discharge_field      = 'discharge_date_time'
admn_date_code_field      = 'admission_date_code_time'
admn_discharge_code_field = 'discharge_date_code_time'
lengthofstay_field        = 'lengthofstay'
isPneumonia_field         = 'isPneumonia'
mortal_admin_field        = 'mortal_admin'
comorbidity_field         = 'Comorbidity_score'
icu_admin_field           = 'icu_count\r'
no_eobs_field             = 'no_obs_eobs'

 ##### Fields in table eObservations #####
eObs_time_field      = 'timestamp'
eObs_time_code_field = 'timestamp_code'
eObs_time_prev_obs   = 'time_since_prev_obs_in_mins'
eObs_resprate_field  = 'rr'
eObs_sbp_field       = 'sbp'
eObs_dbp_field       = 'dbp'
eObs_newscore_field  = 'ews'
eObs_heartrate_field = 'heart_rate'
eObs_temptr_field    = 'temperature\r'
eObs_oxygen_field    = 'Oxygen_Saturation'
##############################################################################

def retrieve_merge_pickles(name, path):
    df_eobs = pd.DataFrame()
    for i, file_name in enumerate(os.listdir(path)): 
        if name in file_name:
            df = pickle.load(open( path + file_name  , "rb" ))
            df_eobs = pd.concat([df_eobs, df])
            print("consolidation len", i,"length", len(df))
    return df_eobs
##############################################################################

root = '/home/d/dlr10/Documents/01_Preprocessing/00_Data_2019_2020/'
path   = root + 'DataFrame_pickles/Partials_eobs/'
name   = "new_eobs_3d_method2"
df_eobs_3d = retrieve_merge_pickles(name, path)

#name   = "new_eobs_2d_method2"
#df_eobs_2d  = retrieve_merge_pickles(name, path)
#name   = "new_eobs_1d_method2"
#df_eobs_2d  = retrieve_merge_pickles(name, path)
#name   = "new_eobs_12h_method2"
#df_eobs_2d  = retrieve_merge_pickles(name, path)

##############################################################################
pickle.dump(df_eobs_3d, open( "DataFrame_pickles/df_eobs_3d.pickle", "wb" ))
#pickle.dump(df_eobs_2d, open( "DataFrame_pickles/df_eobs_2d.pickle", "wb" ))
#pickle.dump(df_eobs_1d, open( "DataFrame_pickles/df_eobs_1d.pickle", "wb" ))
#pickle.dump(df_eobs_12h, open( "DataFrame_pickles/df_eobs_12h.pickle", "wb" ))

