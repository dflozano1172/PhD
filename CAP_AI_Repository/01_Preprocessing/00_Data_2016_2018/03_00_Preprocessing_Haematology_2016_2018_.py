#%reset -f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re, sqlite3, pickle, time, datetime

##############################################################################
# O LOADING DATA
##############################################################################
t = time.time()
path  = r'/rfs/CAPAI_PhD_dflr2/dflr2/Codes_Data_2/Data/'
#file1 = path + '20210525_admissions.txt' 
#file2 = path + '20210525_eobs.txt'
file3 = path + '20210525_haem-results.txt'
#file4 = path + '20210525_icu.txt'
file5 = path + '20210525_meds.txt'
file6 = path + '20210525_micro-results.txt'
file7 = path + '20210525_oxygen.txt'
file8 = path + '20210525_prev_admissions.txt'
file9 = path + '20210525_spin.txt'
#df_admin = pd.read_csv(file1, sep='\t', lineterminator='\n')
#df_eobs  = pd.read_csv(file2, sep='\t', lineterminator='\n')
df_haemt = pd.read_csv(file3, sep='\t', lineterminator='\n')
#df_icu   = pd.read_csv(file4, sep='\t', lineterminator='\n')
df_meds  = pd.read_csv(file5, sep='\t', lineterminator='\n')
df_micro = pd.read_csv(file6, sep='\t', lineterminator='\n')
#df_oxyge = pd.read_csv(file7, sep='\t', lineterminator='\n')
#df_prev  = pd.read_csv(file8, sep='\t', lineterminator='\n')
df_spin  = pd.read_csv(file9, sep='\t', lineterminator='\n')
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
no_haematology_field      = 'no_haematology_eobs'

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

##### Fields in table haematological tests #####
test_code_field      = 'local_test_code'
test_time_field      = 'sample_collection_date_time'
test_time_code_field = 'sample_collection_date_code_time'
haemt_cols = [test_code_field, test_time_field, test_time_code_field]
#########################################################
# FORMATTING FIELD IN HAEMATOLOGY
#########################################################
t = time.time()
df_haemt = df_haemt.rename(columns = {'system_number':  'patient_id' })
df_haemt[test_time_field]      = pd.to_datetime(df_haemt[test_time_field], format='%Y-%m-%d %H:%M:%S')
df_haemt[test_time_code_field] = df_haemt[test_time_field] - reference_date
df_haemt[test_time_code_field] = df_haemt[test_time_code_field].apply(lambda x: x.days + (x.seconds/(24*3600)))
print("time elapsed: ", time.time() - t)
#########################################################
# LOAD PREPROCESSED DF_PATIENTS, DF_ADMISSIONS 
#########################################################
data = pickle.load( open(r'DataFrame_pickles/df_patients_admissions_2016_2018_v2.pickle', 'rb'))
df_patients   = data[0]
df_admissions = data[1]

##############################################################################
# 1. EXPLORING RAW HAEMATOLOGY DATA
##############################################################################
#display(df_haemt[df_haemt['local_test_code']=='BILI'].head(4))

t = time.time()

print("Tests in heamatology", len(df_haemt))
df_haematology = df_haemt[df_haemt[patient_field].isin(df_patients[patient_field].unique().tolist())]
print("Test in heamatology of patients that have eObs too: ", len(df_haematology))
print("")
print('Number of Haem records with Local test nan:', len(df_haematology[df_haematology['local_test_code'].isnull()]))
df_haematology = df_haematology.drop(df_haematology[df_haematology['local_test_code'].isnull()].index, axis = 0)
print("Number of Haem records adter dropping nan's values test:", len(df_haematology))
# Remove duplicates
df_haematology = df_haematology.drop_duplicates(keep = 'first')
print("")
print("Heamatology samples after removing duplicates: ", len(df_haematology))
print("")
print("elapsed",time.time()-t)

t = time.time()
df_haematology_new = pd.DataFrame(columns = df_haemt.columns)
df_haematology_new[admission_field ] = 0

patients_in_haemt = df_haematology[patient_field].unique().tolist()
t = time.time()
for pat in patients_in_haemt:
    pat_admissions = df_admissions[df_admissions[patient_field] == pat][admission_field].tolist()
    
    for pat_admin in pat_admissions:
        min_admin_date = df_admissions[(df_admissions[patient_field] == pat) & (df_admissions[admission_field]== pat_admin)][admn_date_code_field].min()
        max_admin_date = df_admissions[(df_admissions[patient_field] == pat) & (df_admissions[admission_field]== pat_admin)][admn_discharge_code_field].max()
        temp = df_haematology[(df_haematology[patient_field] == pat) 
                              & (df_haematology[test_time_code_field] >= int(min_admin_date) - 1) 
                              & (df_haematology[test_time_code_field] <= int(max_admin_date) + 1)].sort_values(by=test_time_field)
        temp[admission_field] = pat_admin
        df_haematology_new = pd.concat([df_haematology_new,temp]) if len(temp) > 0 else df_haematology_new
    
print("elapsed",time.time()-t)

print("number of records before filtering", len(df_haematology))
print("number of records after filtering", len(df_haematology_new))

pickling_data = [df_haematology_new]
pickle.dump(pickling_data, open('DataFrame_pickles/df_haematology_v0_1.pickle', 'wb'))

df_pikcled_data = pickle.load( open('DataFrame_pickles/df_haematology_v0_1.pickle', 'rb')) 
df_haematology_new = df_pikcled_data[0]
#display(df_haematology_new.head(10) )

t = time.time()
#####################################################################
#Extraction of blood codes frequencies per patient and admissions
#####################################################################
df_blood_codes = df_haematology_new[test_code_field].value_counts()
aa = []
for i,x in enumerate(np.array(df_blood_codes)): 
    if np.array(df_blood_codes)[:i].sum() < 0.95 * len(df_haematology_new): aa.append(x)
df_blood_codes = df_blood_codes[:len(aa)]
df_blood_codes = df_blood_codes.to_frame()

df_blood_codes['no_patients']   = [len(df_haematology_new[df_haematology_new[test_code_field] == code][patient_field].unique().tolist()) for code in df_blood_codes.index]
df_blood_codes['no_admissions'] = [len(df_haematology_new[df_haematology_new[test_code_field] == code][admission_field].unique().tolist()) for code in df_blood_codes.index]

A = df_blood_codes.index.tolist()

print("Number of tests to consider")
print(len(df_blood_codes))
#display(df_blood_codes.head(50))
print("elapsed",time.time()-t)


path = r'DataFrame_pickles/df_eobs_3d.pickle'
df_eobs = pickle.load( open(path, 'rb'))

list_admissions_eobs = df_eobs[admission_field].unique().tolist()
print("Number of admissions in eObs:",len(list_admissions_eobs))


df_haematology_new  = df_haematology_new[df_haematology_new[test_code_field].isin(df_blood_codes.index)]
df_haematology_new  = df_haematology_new[df_haematology_new[admission_field].isin(list_admissions_eobs)]
blood_admissions    = df_haematology_new[admission_field].unique().tolist()
test_result_field    = 'result'



print("Number of admissions in df_Haematology:",len(blood_admissions))



t = time.time()
DF_df = []
for i, admin in enumerate(blood_admissions):   
    t2 = time.time()
    temp = df_haematology_new[df_haematology_new[admission_field] == admin]
    temp_times_codes = temp[test_time_code_field].unique().tolist()    
    for date_time_code in temp_times_codes:
        DF = [] 
        DF.append(temp.iloc[0][admission_field])
        temp2 = temp[temp[test_time_code_field]==date_time_code]
        DF.append(temp2.iloc[0][test_time_field])
        DF.append(date_time_code)
        for code in df_blood_codes.index:            
            if code in temp2[test_code_field].tolist():
                DF.append(temp2[temp2[test_code_field] == code].iloc[0][test_result_field])
            else:
                DF.append(np.nan)
        DF_df.append(DF)
    
    if i % 1000 == 0: print(i,str(datetime.datetime.today()), time.time()- t2)   
    
cols_Haem          = [admission_field,test_time_field, test_time_code_field ] + df_blood_codes.index.tolist()
df_haematology_new = pd.DataFrame(DF_df, columns  = cols_Haem)
print("elapsed",time.time()-t)




t = time.time()
def processing_haematology(x):
    if type(x) == str and ('>' in x or '<' in x):
        a = 0.0015
        x_ = float(re.sub(r'[>|<]', '',x))
        if '>' == x[0]:
            x_ = x_ + a
        elif '<' == x[0]:
            x_ = x_ - a
    elif (type(x) == str and x[0].isnumeric() and x[-1].isnumeric()) or (type(x) == int or type(x) == float):
        x_ = float(x)
    else:
        x_ = np.nan
    return x_

df_haematology_new_v2 = pd.DataFrame(df_haematology_new.loc[:,[admission_field, test_time_field, test_time_code_field]])
haem_cols             = df_haematology_new.columns.tolist()[3:]
feature = haem_cols[0]

for feature in haem_cols :
    df_haematology_new_v2.loc[:,feature] = df_haematology_new[feature].apply(lambda x: processing_haematology(x))
print("elapsed",time.time()-t)



print("Length df_haematology", len(df_haematology_new))
df_haematology_new.head(5)





import random

def handling_errors(field, df_eobs_new,dict_sypmt_min_max):
    print(field)
    print("Values na in " + field,df_eobs_new[field].isna().sum())
    min_b = dict_sypmt_min_max[field][0]
    max_b = dict_sypmt_min_max[field][1]
    adm_withmin = df_eobs_new[min_b > df_eobs_new[field] ]['admission_id'].unique().tolist()
    adm_withmax = df_eobs_new[df_eobs_new[field] > max_b]['admission_id'].unique().tolist()
    adms_plot   = adm_withmin + adm_withmax
    if len(adms_plot) <= 5: adm_idxs    = range(len(adms_plot))
    else:                   adm_idxs    = random.sample(range(len(adms_plot)), 5)

    df_eobs_new2 = df_eobs_new.copy()
    df_eobs_new2[field] = df_eobs_new2[field].apply(lambda x: x if (min_b <= x) and (x <= max_b) else np.nan)
    
    print("values deleted ", df_eobs_new2[field].isna().sum()- df_eobs_new[field].isna().sum())
    print("Values na in after managing outliers for " + field,df_eobs_new2[field].isna().sum(), "of", 
          len(df_eobs_new2), "this is {:10.2f}".format(df_eobs_new2[field].isna().sum()*100/len(df_eobs_new2)), "%")

    fig = plt.figure(figsize = (20,5))

    for i,idx_adm in enumerate(adm_idxs):
        #print(i)
        adm_ = adms_plot[idx_adm]    
        x = df_eobs_new[df_eobs_new[admission_field] == adm_]['sample_collection_date_time']
        y = df_eobs_new[df_eobs_new[admission_field] == adm_][field]

        x1 = df_eobs_new2[df_eobs_new2[admission_field] == adm_]['sample_collection_date_time']
        y1 = df_eobs_new2[df_eobs_new2[admission_field] == adm_][field]

        ax = fig.add_subplot(1, 5, i+1)
        ax.plot(x,y, 'b-.')
        ax.plot(x1,y1, 'r-')

        ax.axes.get_xaxis().set_visible(False)
    plt.show()
    return df_eobs_new2



#display(df_haematology_new_v2.describe())



dict_blood_min_max = {'CREA':[20, 330], 'UREA':[0.5, 30],'K':[2,7],'EGFR':[20,200],'GFR':[20, 200],'WBC':[3, 55],
         'PLT':[90, 600],'HCT':[0.2, 0.75],'HGB':[70,200],'RBC':[2, 7],'MCH':[20, 42],'MCV':[70, 120],
         'NEUAB':[0.5, 15],'TLYMAB':[0.3,7],'EOSAB':[0, 1.6],'MONAB':[0.005,1.2],'BASAB':[0,0.5],'ALB':[20, 70],
         'ALP':[30,200],'BILI':[0, 50]}
#,'ALT':[1, 70],'TP':[50, 100],'PHOS':[0.1, 3],'CG':[13, 50],'INR':[0,3],'CA':[0.5, 3.5],'MG':[0.5,2],'PT':[8, 17]
# 'NRBCAD',
# 'ACA',
# 'CRP',
df_haematology_new_v2_n = df_haematology_new_v2[['admission_id','sample_collection_date_time'] +
                                                list(dict_blood_min_max.keys())].copy()
for feat in dict_blood_min_max.keys():
    df_haematology_new_v2_n = handling_errors(feat, df_haematology_new_v2_n, dict_blood_min_max)



#display(df_haematology_new_v2.describe())
#display(df_haematology_new_v2_n.describe())



print("number of features in heamatology",len(df_haematology_new_v2_n.columns))
#display(df_haematology_new_v2_n.columns)



df = pd.concat([df_haematology_new_v2_n.isna().sum(), df_haematology_new_v2_n.dtypes], axis=1)
df = df.rename(columns ={0:'nulls',1:'type'})
#display(df.sort_values(by=['nulls']))




t = time.time()
admin_obs = []
for adm in df_admissions[admission_field]:    
    admin_obs.append(len(df_haematology_new_v2_n[df_haematology_new_v2_n[admission_field] == adm]))
df_admissions.loc[:,no_haematology_field] = admin_obs
print("number of admissions", len(df_admissions))
print('times elapsed, during adding no_heamatology_field: ',time.time() - t)
df_admissions.head(4)




pickling_data = [df_patients, df_admissions]
pickle.dump(pickling_data, open('DataFrame_pickles/df_patients_admissions_v3.pickle', 'wb'))
pickling_data = [df_haematology_new_v2_n]
pickle.dump(pickling_data, open('DataFrame_pickles/df_haematology_v1.pickle', 'wb'))


