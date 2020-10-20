import numpy as np
import pandas as pd
import time
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import linear_model
import time

t = time.time()

path     = r'Kaggle_Sirio_Libanes_ICU_Prediction.xlsx'
raw_Data = pd.read_excel(path)
sufix    = ['_MEAN', '_MEDIAN','_DIFF_REL','_DIFF','_MAX','_MIN']
cols     = raw_Data.columns.tolist()
raw_Data['AGE_PERCENTIL'] = pd.factorize(raw_Data['AGE_PERCENTIL'])[0]


data     = raw_Data[['PATIENT_VISIT_IDENTIFIER', 'AGE_PERCENTIL', 'GENDER', 'BLOODPRESSURE_DIASTOLIC_MEAN',
                     'BLOODPRESSURE_SISTOLIC_MEAN',	'HEART_RATE_MEAN', 'RESPIRATORY_RATE_MEAN',	'TEMPERATURE_MEAN',	
                     'OXYGEN_SATURATION_MEAN', 'WINDOW', 'ICU']]



patients = data['PATIENT_VISIT_IDENTIFIER'].unique().tolist()


data[data['PATIENT_VISIT_IDENTIFIER'] == 1]
A = data[data['ICU'] == 1]
pos_patients = A['PATIENT_VISIT_IDENTIFIER'].unique().tolist()

###### Simple Imputer 
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit([[10, 2, 3], [4, np.nan, 6], [10, 7, 9]])
SimpleImputer()
X = [[np.nan, 7, 3], [4, np.nan, 6], [10, np.nan, 9]]
print(imp_mean.transform(X))

###### Iterative Imputer 

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit([[1, 2], [3, 6], [4, 8], [np.nan, 3], [7, np.nan]])
IterativeImputer(random_state=0)
X_test = [[np.nan, 2], [6, np.nan], [np.nan, 6]]
# the model learns that the second feature is double the first
print(np.round(imp.transform(X_test)))