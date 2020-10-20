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


data     = raw_Data[['PATIENT_VISIT_IDENTIFIER',	'AGE_ABOVE65', 'AGE_PERCENTIL', 'GENDER', 'BLOODPRESSURE_DIASTOLIC_MEAN',
                     'BLOODPRESSURE_SISTOLIC_MEAN',	'HEART_RATE_MEAN', 'RESPIRATORY_RATE_MEAN',	'TEMPERATURE_MEAN',	
                     'OXYGEN_SATURATION_MEAN', 'WINDOW', 'ICU']]
data_flat = data[data['WINDOW'] == 'ABOVE_12']

print("Admitted patients:", data_flat['ICU'].sum(), " Not admitted patients: ", len(data_flat[data_flat['ICU']==0]) )

X = data_flat.iloc[:, 2:-2]
Y = data_flat.iloc[:,-1]
print("")

stats_list = ['mean','std','min','max']
row_format ="{:>30}"+"{:>8}" * (len(stats_list))
print(row_format.format("", *stats_list))

fig, axs = plt.subplots(2,int(np.ceil(X.shape[1] / 2)), figsize =(15,5))
j = 0
for i, col in enumerate(X.columns):
    print(row_format.format(col, *["{:.2f}".format(X[col].mean()),"{:.2f}".format(X[col].std()), "{:.2f}".format(X[col].min()),
                                   "{:.2f}".format(X[col].max())]))
    
    k = i % 4
    j = j if k != 0 else j + 1
    sns.distplot(X[Y == 0][col], ax = axs[j-1,k],kde=False, norm_hist= False)
    sns.distplot(X[Y == 1][col], ax = axs[j-1,k],kde=False, norm_hist= False)
    
fig.legend(labels= [0,1])
plt.tight_layout()
fig.suptitle("Distribution of the variables", y=1.05, fontsize = 25)

plt.show()


print("")
fig, axs = plt.subplots(2,int(np.ceil(X.shape[1] / 2)), figsize =(15,5))
j = 0
print(row_format.format("", *stats_list))
for i,col in enumerate(X.columns):
    # https://kharshit.github.io/blog/2018/03/23/scaling-vs-normalization
    # Scale Data from 0 to 1, with out lossing the distribution
    X[col] =  ((X[col] - X[col].min())/(X[col].max() - X[col].min()))
    print(row_format.format(col, *["{:.2f}".format(X[col].mean()),"{:.2f}".format(X[col].std()), X[col].min(), X[col].max()]))
    
    k = i % 4
    j = j if k != 0 else j + 1
    sns.distplot(X[Y == 0][col], ax = axs[j-1,k],kde=False, norm_hist= False)
    sns.distplot(X[Y == 1][col], ax = axs[j-1,k],kde=False, norm_hist= False)
    
fig.legend(labels= [0,1])
plt.tight_layout()
fig.suptitle("Distribution of the variables after scaling", y=1.05, fontsize = 25)

plt.show()


##################### 1  Split the data set in Train_validatyion and test sets
test_portion = 0.2

train_rows_n = int(np.around( (1 - test_portion) * len(X), decimals = 0))
X_shuffle   = X.sample(frac=1)
x_train_val = np.array(X_shuffle)[:train_rows_n,:]
x_test      = np.array(X_shuffle)[train_rows_n:,:]
y_train_val = np.array(Y.loc[X_shuffle.index])[:train_rows_n]
y_test      = np.array(Y.loc[X_shuffle.index])[train_rows_n:]



##################### 2 K-Fold Cross Validation 
# https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation

#from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import auc
from sklearn.metrics import roc_curve







def model_evaluation(model_idx, model, x_valid, y_valid, mean_fpr):
         
        loss          = log_loss(y_valid, model.predict_proba(x_valid), eps=1e-15) #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
        accuracy      = model.score(x_valid,y_valid)
        fpr,tpr,_     = roc_curve(y_valid, model.decision_function(x_valid), drop_intermediate = False)
        #fpr,tpr,_     = roc_curve(y_valid, np.max(model.predict_proba(x_valid), axis = 1), drop_intermediate = False)
        #fpr,tpr,_     = roc_curve(y_valid, model.predict_proba(x_valid)[:,1], drop_intermediate = False)
        interp_tpr    = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        mod_tpr       = interp_tpr
        mod_auc       = auc(fpr, tpr)    
        return loss, accuracy, mod_tpr, mod_auc
    

folds = 4
model_names = ['Logistic Regression', 'SGD Log reg']
models      = [linear_model.LogisticRegression(C=1, max_iter = 1000, penalty = 'l2'), ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.fit
               linear_model.SGDClassifier(loss="log",  max_iter=1000, penalty="l2")   ## https://scikit-learn.org/stable/modules/sgd.html#sgd
               ]
def CrossValidationModelsInfo(folds, x_train_val, y_train_val, models, model_names):
    mean_fpr    = np.linspace(0, 1, 50)
    losses      = np.zeros((len(models), folds))
    accuracies  = np.zeros((len(models), folds))
    auroc_tpr = np.zeros((folds,len(mean_fpr),len(models)))
    mean_fprs_modls = []
    auc_modls   = np.zeros((len(model_names), folds))
    skf = StratifiedKFold(n_splits = folds)    
    
    for j, model in enumerate(models): # process each model
        model_name = model_names[j]
        for i, (train, valid) in enumerate(skf.split(x_train_val, y_train_val)):
            x_train = x_train_val[train]
            x_valid = x_train_val[valid]
            y_train = y_train_val[train]
            y_valid = y_train_val[valid]        
            
            ### Normalization Trainining set and validation set
            mean = x_train.mean(axis=0)
            std = x_train.std(axis=0)
            x_train -= mean    
            x_train /= std
            x_valid -= mean
            x_valid /= std
            
            #### Training Stage    
            fit_model                = model.fit(x_train, y_train)     
            #--------------------------------------------------------
            model_idx                = model_names.index(model_name)       
            loss, accuracy, mod_tpr, mod_auc = model_evaluation(model_name, fit_model, x_valid, y_valid, mean_fpr)
            losses[model_idx][i]     = loss
            accuracies[model_idx][i] = accuracy
            auroc_tpr[i,:,model_idx] = mod_tpr
            auc_modls[model_idx, i]  = mod_auc
        
    return losses, accuracies, auroc_tpr, auc_modls


losses, accuracies, auroc_tpr, auc_modls = CrossValidationModelsInfo(folds, x_train_val, y_train_val, models, model_names)
########################### 3 AUROC Curves
mean_fpr    = np.linspace(0, 1, 50)
colors = ['b', 'gray', 'y', 'c', 'r', 'k', 'm']
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
for mod in range(len(models)):    
    std_auc = np.std(auc_modls[mod,:])
    mean_tpr = np.mean(auroc_tpr[:,:,mod], axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    ax.plot(mean_fpr, mean_tpr, color= colors[mod],
            label=r'Mean ROC ' + model_names[mod] + ' (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic", xlabel = "False Positive Rate", ylabel = "True Positive Rate")
    ax.legend(loc="lower right")
plt.show()

########################### 4 Test Evaluation
for mod in models:
    model = mod.fit(x_train_val, y_train_val)
    y_predict   = model.predict(x_test)
    y_pred_prob = model.predict_proba(x_test)
    y_log_prob  = model.predict_log_proba(x_test)
    break
stats_list = ['val_acc','val_loss','test_acc','precision', 'specificity','sensitivity']
row_format ="{:>10}"+"{:>13}" * (len(stats_list))
print(row_format.format("", *stats_list))
val_accs = accuracies.mean(axis=1)
val_losses = losses.mean(axis=1)
for i in range(len(models)):
    val_acc = val_accs[i]
    val_loss = val_losses[i]
    model = models[i].fit(x_train_val, y_train_val)
    y_predict   = model.predict(x_test)
    tp = np.sum((y_test == 1) & (y_predict == 1))
    fp = np.sum((y_test == 0) & (y_predict == 1))
    fn = np.sum((y_test == 1) & (y_predict == 0))
    tn = np.sum((y_test == 0) & (y_predict == 0))

    precision = tp / (tp + fp)
    specificity = tn / (fp + tn)
    sensitivity = tp / (tp +fn)
    F1 = (2 * precision * sensitivity) /(sensitivity + precision)    
    
    print(row_format.format(model_names[i], *["{:.2f}".format(val_acc), "{:.2f}".format(val_loss), 
                                              "{:.2f}".format((tp+tn)/(tp+fp+fn+tn)),"{:.2f}".format(precision), 
                                              "{:.2f}".format(specificity), "{:.2f}".format(sensitivity) ]))
########################### 3 Learning Curves
