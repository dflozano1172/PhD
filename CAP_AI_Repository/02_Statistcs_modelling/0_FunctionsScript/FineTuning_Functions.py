import numpy as np
import pandas as pd
import re, sqlite3, pickle, time, datetime, random

from sklearn.model_selection import LeaveOneOut
import os
from textwrap import wrap

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

#from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import  make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix, f1_score

import matplotlib.pyplot as plt
# Functions brought and modified from:
# https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65

#==============================================================================
# Grid Search Wrapper
# Cross validate different parameters of the model
#==============================================================================
def grid_search_wrapper(clf, X_train, X_test, y_train, y_test, param_grid,  refit_score='auc'):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """    
    
    scorers = {'None':None,'precision_score': make_scorer(precision_score),
        'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score),
        'auc':'roc_auc'}

    skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit=refit_score,
                           cv=skf, return_train_score=False, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # make the predictions
    y_pred = grid_search.predict(X_test)
    
    print("")
    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)

    # confusion matrix on the test data.
    print('\nConfusion matrix of model optimized for {} on the test data:'.format(refit_score))
    print("")
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
    return grid_search, grid_search.best_params_


#==============================================================================
#==============================================================================
def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]
#==============================================================================
#==============================================================================

def best_thresold_conf_matrix(y_test, y_scores, thrs):
    # function above and view the resulting confusion matrix.
    
    opt_F1   = 0
    opt_t    = 0
    for t in thrs:
        y_pred_adj = adjusted_classes(y_scores, t)
        tmp_F1 = f1_score(y_test, y_pred_adj)
        if tmp_F1 > opt_F1 : 
            opt_t = t
            opt_F1 = tmp_F1
    y_pred_adj = adjusted_classes(y_scores, opt_t)
    cm = confusion_matrix(y_test, y_pred_adj)
    print(pd.DataFrame(cm,
                       columns=['pred_neg', 'pred_pos'], 
                       index=['neg', 'pos']))
    return t

#==============================================================================
#==============================================================================

def precision_recall_threshold(y_scores, y_test, p, r, thresholds, ax, t=0.5):
    """
    plots the precision recall curve and shows the current value for each
    by identifying the classifier's threshold (t).
    """
    
    # generate new class predictions based on the adjusted_classes
    
    
    # plot the curve
    #fig, ax = plt.subplots(figsize=(8,8))
    title = "Precision and Recall curve ^ = current threshold"
    ax.set_title('\n'.join(wrap(title,40)))
    ax.step(r, p, color='b', alpha=0.2, where='post')
    ax.fill_between(r, p, step='post', alpha=0.2,color='b')
    ax.set_ylim([0, 1.01]);
    ax.set_xlim([0, 1.01]);
    ax.set_xlabel('Recall');
    ax.set_ylabel('Precision');
    
    # plot the current threshold on the line
    close_default_clf = np.argmin(np.abs(thresholds - t))
    ax.plot(r[close_default_clf], p[close_default_clf], '^', c='k',
            markersize=15)
    #return ax

#==============================================================================
#==============================================================================  
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, ax):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    #fig = plt.figure(figsize=(8, 8))
    title = "Precision and Recall Scores as a function of the decision threshold"
    ax.set_title('\n'.join(wrap(title,40)))
    ax.plot(thresholds, precisions[:-1], "b--", label="Precision")
    ax.plot(thresholds, recalls[:-1], "g-", label="Recall")
    ax.set_ylabel("Score")
    ax.set_xlabel("Decision Threshold")
    ax.legend(loc='best')
    #return fig
#==============================================================================
#==============================================================================  
def plot_roc_curve(fpr, tpr, ax, label=None):
    """
    The ROC curve, modified from 
    Hands-On Machine learning with Scikit-Learn and TensorFlow; p.91
    """
    #fig = plt.figure(figsize=(8,8))
    
    title = 'ROC Curve'
    ax.set_title('\n'.join(wrap(title,40)))
    ax.plot(fpr, tpr, linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.axis([-0.005, 1, 0, 1.005])
    ax.set_xticks(np.arange(0,1, 0.05))
    ax.set_xticklabels(np.arange(0,1, 0.05), rotation=90)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.legend(loc='best')
    #return fig