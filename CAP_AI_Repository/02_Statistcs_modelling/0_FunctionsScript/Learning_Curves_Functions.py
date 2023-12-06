import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from tensorflow.keras.utils import to_categorical

def plot_learning_curve_sklearn(estimator,
    title,
    X, y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None: _, axes = plt.subplots(1, 3, figsize=(20, 5))

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores, axis=1)
    test_scores_std   = np.std(test_scores, axis=1)
    fit_times_mean    = np.mean(fit_times, axis=1)
    fit_times_std     = np.std(fit_times, axis=1)

    if ylim is not None: axes[0].set_ylim(*ylim)
    # Plot learning curve
    axes[0].set_title(title)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")
    axes[0].grid()
    axes[0].fill_between(train_sizes,   train_scores_mean - train_scores_std,
                                        train_scores_mean + train_scores_std,
        alpha=0.1, color="r", )
    axes[0].fill_between(train_sizes,   test_scores_mean - test_scores_std,
                                        test_scores_mean + test_scores_std,
        alpha=0.1, color="g", )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, "o-")
    axes[2].fill_between(
        fit_times_mean,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

###############################################################################
# HOW TO USE IT
###############################################################################
# fig, axes = plt.subplots(3, 2, figsize=(10, 15))

# X, y = load_digits(return_X_y=True)

# title = "Learning Curves (Naive Bayes)"
# # Cross validation with 100 iterations to get smoother mean test and train
# # score curves, each time with 20% data randomly selected as a validation set.
# cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

# estimator = GaussianNB()
# plot_learning_curve_sklearn(
#     estimator, title, X, y, axes=axes[:, 0], ylim=(0.7, 1.01), cv=cv, n_jobs=4
# )

# title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# # SVC is more expensive so we do a lower number of CV iterations:
# cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

# estimator = SVC(gamma=0.001)
# plot_learning_curve_sklearn(
#     estimator, title, X, y, axes=axes[:, 1], ylim=(0.7, 1.01), cv=cv, n_jobs=4
# )

# plt.show()
###############################################################################
#############################################################################################################################################################################################################################################
###############################################################################
###############################################################################
###############################################################################

def results_precision_recall_cm(Y_valid, y_predict, y_pred_proba, fig_size = None, n_bt = 1000):
    fpr, tpr, _ = metrics.roc_curve(Y_valid, y_pred_proba)
    brier       = metrics.brier_score_loss(Y_valid, y_pred_proba)
    
    pres_recl_f1 = metrics.precision_recall_fscore_support(Y_valid, y_predict, average='binary', pos_label = 1)
    pres_recl_f1 = list(pres_recl_f1[:3]) + [metrics.accuracy_score(Y_valid, y_predict), metrics.auc(fpr, tpr), brier, AUC_CI(Y_valid, y_pred_proba, n_bootstraps = n_bt)]
    #pres_recl_f1 = pres_recl_f1.append(AUC_CI(Y_valid, y_pred_proba))
    
    df = pd.DataFrame([pres_recl_f1], columns = ['Precision','Recall','F1_Score','Accuracy', 'AUC', 'brier', 'AUC_IC'])
    # confusion Matrix
    cm   = metrics.confusion_matrix(Y_valid, y_predict, labels=[1,0])
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[1,0])
    if fig_size == None: fig, ax = plt.subplots(figsize=(2,2))
    else:fig, ax = plt.subplots(figsize=fig_size)
    disp.plot(ax = ax)
        
    return df, [fpr,tpr]

#############################################################################################################################################################################################################################################
###############################################################################
###############################################################################
###############################################################################

def results_precision_recall_cm_multiclass(Y_valid, y_predict, y_pred_proba, label_classes, fig_size = None, save = False, name = None):
    
    
    cm = metrics.confusion_matrix(Y_valid, y_predict)
    # Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
    cm_df_ = pd.DataFrame(cm, index = label_classes, columns = label_classes)
    cm_df = 100*cm_df_ / len(Y_valid)
    cm_df = cm_df_.copy()
    cm_annot = np.array([["{:.2f}%".format(x) for x in item] for item in np.array(cm_df)])
    cm_annot = np.array([["{:.0f}".format(x) for x in item] for item in np.array(cm_df)])
    
    #Plotting the confusion matrix
    plt.figure(figsize=(9,8))
    c_map = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
    b = sns.heatmap(cm_df, annot=cm_annot, cmap =c_map,annot_kws={"fontsize":15}, fmt='', cbar = False)
    b.set_yticklabels(label_classes, size = 16)
    b.set_xticklabels(label_classes, size = 16)
    #plt.title('Confusion Matrix', fontsize = 20)
    plt.ylabel('Real Values', fontsize = 20)
    plt.xlabel('Predicted Values', fontsize = 20)
    if save:plt.savefig("cm_" + name + '.png', transparent = True, bbox_inches = "tight")
    plt.show()
    
    n_classes = len(label_classes)
    
    y_valid = to_categorical(Y_valid, num_classes = n_classes)
    y_predc = to_categorical(y_predict, num_classes = n_classes)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_valid[:, i], y_predc[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_valid.ravel(), y_predc.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
    plt.figure(figsize=(8.5,8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),)

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    #plt.title("Some extension of Receiver operating characteristic to multiclass")
    plt.legend(loc="lower right")
    if save:plt.savefig("AUROC_" + name + '.png', transparent = True, bbox_inches = "tight")
    plt.show()
        
    #return df, [fpr,tpr]
    return cm_df_

###############################################################################
#############################################################################################################################################################################################################################################
###############################################################################
###############################################################################
###############################################################################

# AUROC Confidence Interval
def AUC_CI(y_true, y_pred, n_bootstraps = 1000):
    #n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC to be defined: reject the sample
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
        #print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # Computing the lower and upper bound of the 90% confidence interval You can change the 
    # bounds percentiles to 0.025 and 0.975 to get a 95% confidence interval instead.
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(confidence_lower, confidence_upper))
    
    CI = ["{:0.2}".format(confidence_lower), "{:0.3}".format(confidence_upper)]
    
    return CI