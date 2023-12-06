import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import time

matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['mathtext.fontset'] = 'cm'

##############################################################################
# ENCODING FUNCTIONS
##############################################################################

# ENCODING CATEGORICAL VARIABLES WITH TARGET ENCODER OR NORMALISE CONTINUOUS VARIABLES
# https://towardsdatascience.com/target-encoding-and-bayesian-target-encoding-5c6a6c58ae8c
def target_encoder_Binary(column, target, X_data):
    df = []
    for cat in X_data[column].unique():
        row = []
        row.append(len(X_data[(X_data[target]==0) & (X_data[column] == cat)]))
        row.append(len(X_data[(X_data[target]==1) & (X_data[column] == cat)]))
        df.append(row)
    df = pd.DataFrame(df, index = X_data[column].unique(), columns = ['0','1'])
    df['prob'] = df['1']/(df['1']+df['0'])
    col_encod = X_data[column].map(dict(zip(df.index, df['prob'])))
    return col_encod, df
############################################################
def Normalise_n_encode_train_set(X_data, feat_list, data_types):
    encoder = []
    X_data_norm_2 = pd.DataFrame()
    for feat in feat_list:
        if data_types[feat] == 'Continuous':
            X_data_feat = X_data[feat]
            mean        = X_data_feat.mean()
            std         = X_data_feat.std()
            X_data_norm_2[feat] = (X_data_feat - mean)/std
            encoder.append([feat, data_types[feat], [mean, std]])
        elif data_types[feat] == 'Categorical':
            X_data_norm_2[feat],df = target_encoder_Binary(feat, 'Mortality', X_data)
            encoder.append([feat, data_types[feat], df])
        elif data_types[feat] == 'Binary':
            X_data_norm_2[feat] = X_data[feat].copy()
            encoder.append([feat, data_types[feat], ""])
    encoder =pd.DataFrame(encoder, columns = ['feature','type','parameters'])
    return X_data_norm_2,encoder
############################################################
def Normalise_n_encode_val_set(val_data, norm_encoder, feat_list, data_types):
    X_data_norm_2 = pd.DataFrame()
    for feat in feat_list:
        if data_types[feat] == 'Continuous':
            X_data_feat = val_data[feat]
            mean, std   = norm_encoder[norm_encoder['feature'] == feat].iloc[0]['parameters']
            X_data_norm_2[feat] = (X_data_feat - mean)/std

        elif data_types[feat] == 'Categorical':
            df = norm_encoder[norm_encoder['feature'] == feat].iloc[0]['parameters']
            X_data_norm_2[feat] = val_data[feat].map(dict(zip(df.index, df['prob'])))
            X_data_norm_2[feat] = X_data_norm_2[feat].fillna(0)
        elif data_types[feat] == 'Binary':
            X_data_norm_2[feat] = val_data[feat].copy()
    return X_data_norm_2
##############################################################################
# PLOTTING FUNCTIONS 
##############################################################################
def history_plots_(history, SAVE = 0, name = ''):
    auc_keys = [x for x in history.history.keys() if 'auc' in x]
    loss = history.history["loss"];     val_loss = history.history["val_loss"]
    acc  = history.history["accuracy"]; val_acc  = history.history["val_accuracy"]
    auc  = history.history[auc_keys[0]];      val_auc = history.history[auc_keys[1]]
    epochs  = range(1, len(acc) + 1)
    f, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize = (8,4))
    ax0.plot(epochs, loss, "bo", label="Training Loss")
    ax0.plot(epochs, val_loss, "b", label="Validation Loss")
    ax0.set_title("Train-val Loss")
    ax1.plot(epochs, acc, "bo", label="Training Accuracy")
    ax1.plot(epochs, val_acc, "b", label="Validation Accuracy")
    ax1.set_title("Train-val Accuracy")
    ax2.plot(epochs, auc, "bo", label="Training AUC")
    ax2.plot(epochs, val_auc, "b", label="Validation AUC")
    ax2.set_title("Train-val AUC")
    plt.legend()
    if SAVE == 1: plt.savefig(name, transparent = True, bbox_inches = "tight")
    plt.show()
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================# ============================================================================# ============================================================================

##############################################################################
# SCALING FUNCTION ###########################################################

def scaling_feature_ts_(feature, X_data):
    t = time.time()
    X_feat_scal = []
    ls_admins = X_data['admission_id'].unique().tolist()
    for i, admin in enumerate(ls_admins):
        ts   = X_data[X_data['admission_id'] == admin][feature]
        mort = X_data[X_data['admission_id'] == admin]['Mortality'].mean()
        ts_min  = ts.min()
        ts_max  = ts.max()
        if (ts_min == ts_max): #Denominator is 0
            ts_scal = (ts - ts_min) / (ts_max)
        else:
            ts_scal = (ts - ts_min) / (ts_max - ts_min)
        
        X_feat_scal.append([admin, mort] + ts_scal.tolist())
    X_feat_scal = pd.DataFrame(X_feat_scal, columns = ['admission_id', 'Mortality']+ ["time_"+str(step) for step in range(len(ts_scal)) ])
    print("Elapsed time Scaling "+ feature+" data:", time.time()-t)
    return X_feat_scal

##############################################################################
# SCALING FUNCTION GLOBAL ####################################################

def scaling_feature_ts_global(feature, X_data):
    t = time.time()
    X_feat_scal = []
    ls_admins = X_data['admission_id'].unique().tolist()
    df_temp = X_data[['admission_id', feature, 'Mortality']].copy()
    ts_min  = df_temp[feature].min()
    ts_max  = df_temp[feature].max()
    df_temp[feature + '_scale'] = (df_temp[feature] - ts_min) / (ts_max - ts_min)
    
    ts   = np.asarray(df_temp[feature + '_scale']).reshape( len(ls_admins), 144)
    mrts = np.mean(np.asarray(df_temp['Mortality']).reshape(len(ls_admins), 144), axis = 1).reshape(len(ls_admins),1)
    
    X_feat_scal = np.concatenate((np.asarray(ls_admins).reshape(len(ls_admins),1), mrts, ts), axis = 1)
    X_feat_scal = pd.DataFrame(X_feat_scal, columns = ['admission_id', 'Mortality']+ ["time_"+str(step) for step in range(ts.shape[1]) ])
    print("Elapsed time Scaling "+ feature+" data:", time.time()-t)
    return X_feat_scal

##############################################################################
# VALIDATION FUNCTIONS #######################################################
def Validation_Samples_ts(ls, model, X_data, feature, feature_dict, val, save = False):
    rows = 2; cols = 5
    fig, axs = plt.subplots(rows, cols, figsize=(20, 10), facecolor='w', edgecolor='k')
    
    for i, n_ts in enumerate(ls[:10]):
        ts = X_data.iloc[n_ts][[x for x in X_data.columns if ('time' in x)]]
    
        x = np.array(ts)[0:96].reshape((1,96,1))
        x = x.astype(float)
        y_pred  = model.predict(x)
        
        ts_pred = y_pred.reshape((48,))
    
        axs[int(i/cols), i%cols].plot(range(144),ts)
        axs[int(i/cols), i%cols].plot(range(96,144), ts_pred)
        axs[int(i/cols), i%cols].set_title('Admission no ' + str(n_ts), fontsize = 15)    
        
        axs[int(i/cols), i%cols].tick_params(axis='x', labelsize= 16)
        axs[int(i/cols), i%cols].tick_params(axis='y', labelsize= 16)
        axs[int(i/cols), i%cols].axvline(x = 96, color = 'r', label = '3rd day admission', linestyle = '--')
        
        if i%cols != 0: axs[int(i/cols), i%cols].set_yticklabels([])
    fig.text(0.5, -0.03, 'Time Step', ha='center', fontsize = 20)
    fig.text(-0.01, 0.5, feature_dict[feature], va='center', rotation='vertical', fontsize = 20)
    plt.tight_layout()
    if save: plt.savefig('Plots/'+ feature+'_samples'+str(val)+'.png', transparent = True, bbox_inches = "tight")
    plt.show()
# ---------------------------------------------------------------------   
# ---------------------------------------------------------------------
def Validation_General_ts(ls, model, X_data, feature, feature_dict, val, save = False):
    a4_dims = (8, 8)
    fig, ax = plt.subplots(figsize=a4_dims)
    for n_ts in ls :
        
        color =  '#FA8072' if X_data.iloc[n_ts]['Mortality'] == 1 else '#9fc2e0'
        
        ts = X_data.iloc[n_ts][[x for x in X_data.columns if ('time' in x)]]
        x = np.array(ts)[:96].reshape((1,96,1))
        x = x.astype(float)
        y_pred = model.predict(x)
        y_pred_norm = y_pred.reshape((48,))
        ax.plot(range(96,144), y_pred_norm, color = color)
    fig.text(-0.03, 0.5, feature_dict[feature] + ' Scaled', va='center', rotation='vertical', fontsize = 20)
    fig.text(0.5, -0.03, 'Time Step', ha='center', fontsize = 20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    fig.tight_layout()
    if save: plt.savefig('Plots/'+ feature+'_gral'+str(val)+'.png', transparent = True, bbox_inches = "tight")
    plt.show()
    
def quantify_results(X_data, model, feature):
    t = time.time()
    results = []
    for n_ts in range(len(X_data['admission_id'].unique().tolist() )):
        admin = X_data.iloc[n_ts]['admission_id']
        ts = X_data.iloc[n_ts][[x for x in X_data.columns if ('time' in x)]]
        
        x = np.array(ts)[0:96].reshape((1,96,1))
        x = x.astype(float)
        y_real = np.array(ts)[:48]
        y_pred = model.predict(x)
        y_pred = y_pred.reshape((48,))
        
        mse_0 = (y_real[0] - y_pred[0])**2
        mse_l = (y_real[-1] - y_pred[-1])**2
        # 1 means correct gradient
        if (y_real[0] - y_real[-1]) * (y_pred[0] - y_pred[-1]) >= 0 : acc = 1 
        else: acc = 0
        
        results.append([admin, mse_0, mse_l, acc, y_pred])
    print("Elapsed time validating "+ feature+" data:", time.time()-t)
    return results








