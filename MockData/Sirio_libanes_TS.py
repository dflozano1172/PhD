%reset -f
import numpy as np
import pandas as pd
import time
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import time

t = time.time()
z_save_list = dir()

path     = r'Kaggle_Sirio_Libanes_ICU_Prediction.xlsx'
raw_Data = pd.read_excel(path,na_values = np.NaN)
raw_Data = raw_Data.replace('', np.NaN)
#sufix    = ['_MEAN', '_MEDIAN','_DIFF_REL','_DIFF','_MAX','_MIN']
#cols     = raw_Data.columns.tolist()
raw_Data['AGE_PERCENTIL'] = pd.factorize(raw_Data['AGE_PERCENTIL'])[0] #transforms string data into numerica data

data     = raw_Data[['PATIENT_VISIT_IDENTIFIER', 'AGE_PERCENTIL', 'GENDER', 'BLOODPRESSURE_DIASTOLIC_MEAN',
                     'BLOODPRESSURE_SISTOLIC_MEAN',	'HEART_RATE_MEAN', 'RESPIRATORY_RATE_MEAN',	'TEMPERATURE_MEAN',	
                     'OXYGEN_SATURATION_MEAN', 'WINDOW', 'ICU']]

no_patients_positive = data[(data['ICU'] == 1) & (data['WINDOW'] == 'ABOVE_12')]['PATIENT_VISIT_IDENTIFIER'].unique().tolist()
no_patients_negative = data[(data['ICU'] == 0) & (data['WINDOW'] == 'ABOVE_12')]['PATIENT_VISIT_IDENTIFIER'].unique().tolist()


patients = data['PATIENT_VISIT_IDENTIFIER'].unique().tolist()
pat_full_pos = []
pat_miss_pos = []
pat_full_neg = []
pat_miss_neg = []
for patient in patients:
    pat_data = data[data['PATIENT_VISIT_IDENTIFIER'] == patient]
    pat_data = pat_data.assign(Time_Val = pat_data.index - pat_data.iloc[0].name)
    if not any(pat_data.isnull().any()):       
        if pat_data['ICU'].sum() == 0:
            pat_full_neg.append(pat_data.drop(columns=['WINDOW']).values.tolist())
        else:
            pat_full_pos.append(pat_data.drop(columns=['WINDOW']).values.tolist())
    else:
        if pat_data['ICU'].sum() == 0:
            pat_miss_neg.append(pat_data.drop(columns=['WINDOW']).values.tolist())
        else:
            pat_miss_pos.append(pat_data.drop(columns=['WINDOW']).values.tolist())
            
pat_full_pos = np.array(pat_full_pos)
pat_miss_pos = np.array(pat_miss_pos)
pat_full_neg = np.array(pat_full_neg)
pat_miss_neg = np.array(pat_miss_neg)

imp_neg = IterativeImputer(max_iter=10, random_state=0)
imp_neg.fit(pat_full_neg.mean(axis=0))
imp_pos = IterativeImputer(max_iter=10, random_state=0)
imp_pos.fit(pat_full_pos.mean(axis=0))

for i, patient in enumerate(np.rollaxis(pat_miss_pos,0)): pat_full_pos = np.insert(pat_full_pos, len(pat_full_pos),imp_pos.transform(patient), axis = 0)
for i, patient in enumerate(np.rollaxis(pat_miss_neg,0)): pat_full_neg = np.insert(pat_full_neg, len(pat_full_neg),imp_neg.transform(patient), axis = 0)
data_imputed = np.concatenate((pat_full_pos ,pat_full_neg),axis=0)

z_save_list.append('data_imputed')
z_save_list.append('save_list')
for z_name in dir():
    if z_name not in z_save_list:
        del globals()[z_name]



print(time.time() - t)