
import os
import pandas as pd
from pathlib import Path
import sqlite3
from sqlite3 import Connection
import streamlit as st
import pickle 
from sqlalchemy import create_engine
import numpy as np
import matplotlib.pyplot as plt

URI_SQLITE_DB = "test.db"
path = os.getcwd()
data_imputed = pickle.load(open(path + "\data_imputed.pickle","rb"))
conn = sqlite3.connect(path + "\\test1.db")  #Mandatory line
c = conn.cursor()
c.execute("DROP TABLE IF EXISTS test1")
conn.commit()
data_imputed.to_sql('test1', con = conn)
conn.commit()
c.execute("SELECT * FROM test1")
    
df = pd.read_sql("SELECT * FROM test1", con=conn).drop(columns = ['index'])
conn.close()


no_demographics = 2 # Number of demographics variables
idx_target      = 8 # Index in the dataframe of the target variable
cols    = df.columns.tolist()
del cols[9:11]
cols = cols[no_demographics + 1:]


# Demographics AGE_PERCENTIL, GENDER
tg_vals = df['ICU'].unique().tolist()
val_age = df['AGE_PERCENTIL'].unique().tolist()
val_gen = df['GENDER'].unique().tolist()


plot_tensor = [len(df[(df['ICU'] == x) & (df['AGE_PERCENTIL'] == y) & (df['GENDER']==z) & (df['Time_Val'] == 4)])  for y in range(len(val_age)) for z in range(len(val_gen)) for x in range(len(tg_vals))]
plot_tensor = np.array(plot_tensor).reshape((len(val_age),len(val_gen), len(tg_vals)))

def BarPlot_stackedandpaired(plot_tensor, labels):
    # Begin plot
# plot_tensor.shape[0] number of columns in the chart
# plot_tensor.shape[1] number of series for each series 
# plot_tensor.shape[2] number of categories stacked in the columns
#plot_tensor = np.array([[[12, 30, 1, 8, 22],[28, 6, 16, 5, 10],[29, 3, 24, 25, 17]],
#                  [[12, 30, 1, 8, 22],[28, 6, 16, 5, 10],[29, 3, 24, 25, 17]]])
    barWidth = 0.25
    no_categs = plot_tensor.shape[2] 
    no_cols   = plot_tensor.shape[1]
    no_series = plot_tensor.shape[0]
    r_color = [['#1D0DF3','#45CD33'],['#467DBA','#689C64']]
    r_ncols = np.zeros((no_cols, no_series))
    fig = plt.figure(figsize = (2.3,2))
    for i in range(r_ncols.shape[0]): r_ncols[i, :] = np.arange(no_series) if i == 0 else [x + barWidth for x in r_ncols[i-1,:]]
    for i in range(no_categs):
        for j in range(no_cols):   
            plt.bar(r_ncols[j,:], plot_tensor[:,j,i], width=barWidth,bottom = plot_tensor[:,j,:i].sum(axis=1), color = r_color[i][j], label= labels[i+j])
    
    plt.xlabel('Age Percentil', fontweight='bold')
    plt.xticks([r + barWidth for r in range(no_series)], range(no_series))
    plt.legend(bbox_to_anchor=(1.0,1.0))
    #plt.show()
    return fig

def BarPlot_stacked(plot_tensor, labels):
    barWidth = 0.25
    no_categs = plot_tensor.shape[1] 
    no_series = plot_tensor.shape[0]
    r_color = ['#1D0DF3','#467DBA']    
    fig = plt.figure(figsize = (3,3))
    for i in range(no_categs):
        plt.bar(np.arange(no_series), plot_tensor[:,i], width=barWidth,bottom = plot_tensor[:,:i].sum(axis=1), color = r_color[i], label= labels[i])
    
    plt.xlabel('Age Percentil', fontweight='bold')
    plt.xticks([r + barWidth for r in range(no_series)], range(no_series))
    plt.legend(bbox_to_anchor=(1.0,1.0))
    #plt.show()
    return fig


def build_sidebar(cols):
    st.sidebar.header("Parameters")
    category = st.sidebar.radio('ICU',['Admitted','Not admitted'])
    feature  = st.sidebar.radio('Features',cols)
    gender   = st.sidebar.radio('Gender',['F','M','All'])
    return category, gender, feature

st.write('ICU is defined the patient that was admitted at any point in the time')
col1, col2 = st.beta_columns(2)

cat, gen, feature = build_sidebar(cols)
category = 1 if cat == 'Admitted' else 0
gender   = [0] if gen == 'M' else [1] if gen == 'F' else [0,1]
labels   = ['ICU 0_M', 'ICU 0_F','ICU 1_M','ICU 1_F']
col1.pyplot(BarPlot_stackedandpaired(plot_tensor, labels))  
labels   = ['ICU ' + cat + '_M','ICU ' + cat +'_F']  
col2.pyplot(BarPlot_stacked(plot_tensor[:,:,category], labels))  

#--------------------------------------------------------------------------------------
####### Second line in streamlit
#--------------------------------------------------------------------------------------
st.write('---')
col1, col2, col3 = st.beta_columns((0.80,2.0,4))
feature = col2.radio('Features_chart',cols)
gender  = col1.radio('Gender_chart',['F','M','All'])
gender  = [0] if gender == 'M' else [1] if gender == 'F' else [0,1]
categor = df['ICU'].unique().tolist()
color_main = [(43/255.0,72/255.0,106/255.0,1),(134/255.0,155/255.0,160/255.0,1), (163/255.0,155/255.0,176/255.0,1),
              (210/255.0,187/255.0,162/255.0,1),(90/255.0,119/255.0,2/255.0,1)]
line_main  = ['-','--','-.']
color_back = [(43/255.0,72/255.0,106/255.0,0.2),(134/255.0,155/255.0,160/255.0,0.2), (163/255.0,155/255.0,176/255.0,0.2),
              (210/255.0,187/255.0,162/255.0,0.2),(90/255.0,119/255.0,2/255.0,0.2)]

no_timestamps = len(df['Time_Val'].unique().tolist())
plot_tensor_mean = np.zeros((len(categor),no_timestamps))
plot_tensor_std  = np.zeros((len(categor),no_timestamps))
fig = plt.figure(figsize = (4,4))
for cat in categor:
    cat = int(cat)
    patients = df[(df['ICU'] == cat) & (df['GENDER'].isin(gender)) & (df['Time_Val'] == 4)]['PATIENT_VISIT_IDENTIFIER'].unique().tolist()

    for i in range(no_timestamps):
        plot_tensor_mean[cat,i] = df[(df['PATIENT_VISIT_IDENTIFIER'].isin(patients)) & (df['Time_Val'] == i)][feature].mean()
        plot_tensor_std[cat, i] = df[(df['PATIENT_VISIT_IDENTIFIER'].isin(patients)) & (df['Time_Val'] == i)][feature].std()
    error =  plot_tensor_std[cat, :]
    plt.plot(range(no_timestamps), plot_tensor_mean[cat,:], color = color_main[cat], linestyle = line_main[cat], label = 'cat_' + str(cat))
    plt.fill_between(range(no_timestamps), plot_tensor_mean[cat,:]-error, plot_tensor_mean[cat,:]+error, color = color_back[cat])
plt.xlabel('Time step', fontweight='bold')
plt.title(feature + ' for both categories in time')
plt.legend(bbox_to_anchor=(1.0,1.0))
col3.pyplot(fig)

#gender  = [0]
#--------------------------------------------------------------------------------------
####### Third line in streamlit
#--------------------------------------------------------------------------------------
st.write('---')

no_timestamps = len(df['Time_Val'].unique().tolist())

feature = st.radio('Feat_series',cols)
col1, col2 = st.beta_columns((2.0,2.0))

patients_admitted = set()
series_admin  = {}
series_Nadmin = {}
fig = plt.figure(figsize = (4,4))
for t in range(no_timestamps):
    t = int(t)
    patients = set(df[(df['ICU'] == 1) & (df['Time_Val'] == t)]['PATIENT_VISIT_IDENTIFIER'].unique().tolist())
    patients_admitted = patients - patients_admitted    
    series_admin[t]  = df[(df['PATIENT_VISIT_IDENTIFIER'].isin(patients_admitted)) & (df['Time_Val'] == t)][feature]
    series_Nadmin[t] = df[(df['PATIENT_VISIT_IDENTIFIER'].isin(patients_admitted) == False) & (df['Time_Val'] == t)][feature]

fig = plt.figure(figsize = (4,4))
plt.boxplot(series_admin.values())
plt.title("features dist for admitted patients")
plt.ylim([-1,1])
plt.xlabel('Time step', fontweight='bold')
#plt.xticks(series.keys())
col1.pyplot(fig)
fig = plt.figure(figsize = (4,4))
plt.boxplot(series_Nadmin.values())
plt.title("features dist for not admitted patients")
plt.ylim([-1,1])
plt.xlabel('Time step', fontweight='bold')
col2.pyplot(fig)
st.write('---')
