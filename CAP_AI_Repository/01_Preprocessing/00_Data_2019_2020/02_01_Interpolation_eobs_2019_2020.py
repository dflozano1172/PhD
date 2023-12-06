import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re, sqlite3, pickle, time, datetime, random, math

from sklearn.model_selection import LeaveOneOut
import os

t = time.time()

import argparse
parser = argparse.ArgumentParser(description='A test program.')
parser.add_argument("-p", "--print_string", help="Prints the supplied argument.",  nargs='*')
args = parser.parse_args()
#min_adm = int(args.print_string[0])
#max_adm = int(args.print_string[1])
print(args.print_string)

batch = int(args.print_string[0])

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
# 0. LOAD DATA AND PARAMETERS
##############################################################################
data = pickle.load( open('DataFrame_pickles/df_patients_admissions_2019_2020_v2.pickle', 'rb'))
df_patients   = data[0]
df_admissions = data[1]
data = pickle.load( open('DataFrame_pickles/df_eobs_oxygen_2019_2020_v2.pickle', 'rb'))
df_eobs  = data[0]
df_oxyge = data[1]
##############################################################################

list_admissions    = df_admissions[admission_field].unique().tolist()
df_eobs            = df_eobs.drop(columns=[patient_field])
list_eObs_features = df_eobs.columns[3:]

size_batch = math.ceil(len(list_admissions)/5)
min_adm = size_batch * batch
max_adm = size_batch * (batch + 1)
##############################################################################
# 1. DEFINITION OF FUNCTIONS
##############################################################################
# 1.1. interpolate_by_method ===========================================
def interpolate_by_method(series, method, plot = 0):
    if ('polynomial' in method) or ('nearest' in method):
        interpolate = series.interpolate(method = method[:-1], order = int(method[-1]))
    else:
        interpolate = series.interpolate(method = method, order = method)
    if plot != 0: interpolate.plot()
    return interpolate

# 1.2. cross_validation_series ===========================================
def cross_validation_series(feature_series, times, interp_methods, upsamptimes):    
    cv     = LeaveOneOut()
    errors = np.zeros([len(feature_series),len(interp_methods)])
    h = 0
    for train_ix, test_ix in cv.split(times):
        
        feat_train, time_train = feature_series.iloc[train_ix], times.iloc[train_ix]
        feat_test, time_test   = feature_series.iloc[test_ix], times.iloc[test_ix]
             
        values      = [feat_train[time_train == tm].mean() for i,tm in enumerate(upsamptimes)]
        cv_series   = pd.Series(values, index=upsamptimes)
        
        for idx_method, method in enumerate(interp_methods):
            if (cv_series.count() < 4  and method == 'polynomial3') or (method == 'polynomial2' and cv_series.count() < 3): 
                errors[h, idx_method] = np.nan
                continue
            interp = interpolate_by_method(cv_series, method)
            errors[h, idx_method] = (interp.loc[time_test].values[0] - feat_test.values[0]) ** 2
        h = h + 1            
    errors  = np.nanmean(errors, axis=0)
    errors  = np.sqrt(errors) # RMSE
    methods = [interp_methods[i] for i, x in enumerate(errors == np.nanmin(errors)) if x]
    return methods[0], errors
# 1.3. select_methods_feature ============================================
def select_methods_feature(samp_adms_g_30_obs, list_features):
    list_features.append(eObs_oxygen_field)
    list_features.append('Assisted_O2')
    errors_total = np.zeros((len(samp_adms_g_30_obs),len(list_features) , len(interp_methods )))
    for idx, adm in enumerate(samp_adms_g_30_obs):
        print("fine-tuning function: ", idx, "admin", adm)
        
        t1 = time.time()
        adm_los  = df_admissions[df_admissions[admission_field] == adm].iloc[0][lengthofstay_field]
        adm_eobs = df_eobs[df_eobs[admission_field]==adm]
        adm_oxyg = df_oxyge[df_oxyge[admission_field]==adm][[eObs_time_field, eObs_oxygen_field,'Assisted_O2']].copy()
        if len(adm_eobs) == len(adm_oxyg):
            adm_eobs.insert(len(adm_eobs.columns),eObs_oxygen_field, adm_oxyg[eObs_oxygen_field].tolist())
            adm_eobs.insert(len(adm_eobs.columns),'Assisted_O2', adm_oxyg['Assisted_O2'].tolist())
        else:
            adm_eobs = adm_eobs.merge(adm_oxyg,how='left', left_on=eObs_time_field, right_on=eObs_time_field)
            
        times    = adm_eobs[eObs_time_field]        
        times_3d = times[times<= (times.min() + datetime.timedelta(days=3))]
        times_3d = times.iloc[:len(times_3d) + 1]
        
        if (adm_los.days    == 0 and adm_los.seconds < 3600 * 12) or (len(times_3d) <= 4) : continue
        
        times_3d = times_3d.apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour, min(15*(round(dt.minute / 15)),45)))
        upsamp_times    = pd.date_range(times_3d.min(), times_3d.max(), freq = '15T').tolist()
        #upsamp_features = np.zeros([len(upsamp_times),len(list_features)])
        
        for idx_feat, feature in enumerate(list_features):
            feature_series = adm_eobs[feature].iloc[:len(times_3d)]        
            
            method, errors = cross_validation_series(feature_series, times_3d, interp_methods, upsamp_times)
            
            errors_total[idx, idx_feat, :] = errors
            
            #values  = [feature_series[times_3d == tm].mean() for i,tm in enumerate(upsamp_times)]        
            #series_ = pd.Series(values, index=upsamp_times)
            #interp  = interpolate_by_method(series_, method)
            #upsamp_features[:, idx_feat] = interp
        
        print("fine-tuning function: ", idx, " ", adm, ": ", time.time() - t1)
    return errors_total

##############################################################################
# 2. FINE TUNING OF INTERPOLATION METHODS
##############################################################################

random.sample(range(1, 100), 3)

t = time.time()

interp_methods = ['linear','polynomial2', 'polynomial3', 'akima','pchip','nearest5']
np.random.seed(0)
no_Adm_fine_tune   = 500

t2 = time.time()
lst_adms_g_30_obs  = df_admissions[df_admissions[no_eobs_field]>30][admission_field].tolist()
samp_adms_g_30_obs = np.random.choice(lst_adms_g_30_obs, no_Adm_fine_tune, replace=False)  
errors_total       = select_methods_feature(samp_adms_g_30_obs, list_eObs_features.tolist())
error_av_total = np.mean(errors_total, axis = 0)
methods_feat_0 = [interp_methods[i] for err in error_av_total for i, x in enumerate(err == np.nanmin(err)) if x ]
print('Training the interpolation method finished', time.time() - t2)

##############################################################################
# 3. INTERPOLATION USING THE FINE TUNED METHODS
##############################################################################
print("")
new_eobs_3d = pd.DataFrame(columns = df_eobs.columns)
new_eobs_2d = pd.DataFrame(columns = df_eobs.columns)
new_eobs_1d = pd.DataFrame(columns = df_eobs.columns)
new_eobs_12h = pd.DataFrame(columns = df_eobs.columns)

if max_adm >= len(list_admissions): max_adm = len(list_admissions)
not_interpolated = []
result = []
for idx, adm in enumerate(list_admissions[min_adm:max_adm]):
    print("Interpolating: ", idx, "admin", adm, "...")
    
    t1 = time.time()
    adm_eobs = pd.DataFrame()
    adm_oxyg = pd.DataFrame()
    adm_los  = df_admissions[df_admissions[admission_field] == adm].iloc[0][lengthofstay_field]
    adm_eobs = df_eobs[df_eobs[admission_field]==adm].copy()
    adm_oxyg = df_oxyge[df_oxyge[admission_field]==adm][[eObs_time_field, eObs_oxygen_field,'Assisted_O2']].copy()
    if len(adm_eobs) == len(adm_oxyg):
        adm_eobs.insert(len(adm_eobs.columns),eObs_oxygen_field, adm_oxyg[eObs_oxygen_field].tolist())
        adm_eobs.insert(len(adm_eobs.columns),'Assisted_O2', adm_oxyg['Assisted_O2'].tolist())
    else:
        adm_eobs = adm_eobs.merge(adm_oxyg,how='left', left_on=eObs_time_field, right_on=eObs_time_field)
    
    
    list_eObs_features = adm_eobs.columns[3:]
    times    = adm_eobs[eObs_time_field]
    
    times_3d = times[times<= (times.min() + datetime.timedelta(days=3))]
    times_3d = times.iloc[:len(times_3d) + 1]
    
    if (adm_los.days    == 0 and adm_los.seconds < 3600 * 12) or (len(times_3d) <= 4) or any(adm_eobs.isna().sum() == len(adm_eobs)): continue
    
    times_3d = times_3d.apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour, min(15*(round(dt.minute / 15)),45)))
    upsamp_times    = pd.date_range(times_3d.min(), times_3d.max(), freq = '15T').tolist()
    upsamp_features = np.zeros([len(upsamp_times),len(list_eObs_features)])
    
    result.append([adm,len(times_3d)])
    
    for idx_feat, feature in enumerate(list_eObs_features):
        feature_series = adm_eobs[feature].iloc[:len(times_3d)]        
        if feature_series.notnull().sum() <= 1:
            upsamp_features[:, idx_feat] = feature_series[feature_series.notnull()].iloc[0]
            if adm not in not_interpolated: not_interpolated.append(adm)
            continue
        
        values  = [feature_series[times_3d == tm].mean() for i,tm in enumerate(upsamp_times)]        
        series_ = pd.Series(values, index=upsamp_times)
        interp  = interpolate_by_method(series_, methods_feat_0[idx_feat])
        upsamp_features[:, idx_feat] = interp
        
    df = pd.DataFrame(upsamp_features, columns = list_eObs_features)
    df[admission_field] = adm
    df[eObs_time_prev_obs] = 15
    df[eObs_time_field] = upsamp_times
    df = df.dropna()
    df['ews'] = df['ews'].apply(lambda x: round(x))
    df['Assisted_O2'] = df['Assisted_O2'].apply(lambda x: round(x))
    if (adm_los.days >= 3)    and (len(df) >= 4*24*3): new_eobs_3d  = pd.concat([new_eobs_3d, df.iloc[:4*24*3]])
    #if (adm_los.days >= 2)    and (len(df) >= 4*24*2): new_eobs_2d  = pd.concat([new_eobs_2d, df.iloc[:4*24*2]])
    #if (adm_los.days >= 1)    and (len(df) >= 4*24*1): new_eobs_1d  = pd.concat([new_eobs_1d, df.iloc[:4*24*1]])
    #if (adm_los.seconds >= 2) and (len(df) >= 4 *12) : new_eobs_12h = pd.concat([new_eobs_12h, df.iloc[:4*12]]) 
    print( idx, " ", adm, ": ", time.time() - t1)
##############################################################################
# 4. SUMMARY
##############################################################################

print("")
print("Interpolations not succesful:", not_interpolated)
print("")
print("Selected methods:")
print(methods_feat_0)    

result = pd.DataFrame(result, columns=['admission_id','initialdata'])
pickle.dump(result, open( "DataFrame_pickles/Partials_eobs/result_"+ str(min_adm) + "_" + str(max_adm) + ".pickle", "wb" ))
pickle.dump(new_eobs_3d, open( "DataFrame_pickles/Partials_eobs/new_eobs_3d_method2_"+ str(min_adm) + "_" + str(max_adm) + ".pickle", "wb" ))
#pickle.dump(new_eobs_2d, open( "DataFrame_pickles/Partials_eobs/new_eobs_2d_method2_"+ str(min_adm) + "_" + str(max_adm) + ".pickle", "wb" ))
#pickle.dump(new_eobs_1d, open( "DataFrame_pickles/Partials_eobs/new_eobs_1d_method2_"+ str(min_adm) + "_" + str(max_adm) + ".pickle", "wb" ))
#pickle.dump(new_eobs_12h, open( "DataFrame_pickles/Partials_eobs/new_eobs_12h_method2_"+ str(min_adm) + "_" + str(max_adm) + ".pickle", "wb" ))


print("Elapsed in all the process", time.time() - t)

