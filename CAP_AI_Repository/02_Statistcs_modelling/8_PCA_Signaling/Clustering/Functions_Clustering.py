#=============================================================================
##############################################################################
##############################################################################
# IMPORT ALL TIME PACKAGES
##############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time

from tslearn.barycenters import dtw_barycenter_averaging

from tslearn.utils import to_time_series_dataset
from tslearn.clustering import KShape, TimeSeriesKMeans

from minisom import MiniSom

import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)


# ===================================================================================
# ===================================================================================
# ===================================================================================
# ===================================================================================

def run_SOM_clustering(mySeries_,lst_adms_, mort_dict_,som_x_, som_y_):
    t = time.time()
    som = MiniSom(som_x_, som_y_,len(mySeries_[0]), sigma=0.3, learning_rate = 0.01)    
    som.random_weights_init(mySeries_)
    som.train(mySeries_, 500000)
    
    # each neuron represents a cluster with np.ravel_multi_index we convert the bidimensional
    winner_coordinates = np.array([som.winner(x) for x in mySeries_]).T
    # coordinates to a monodimensional index
    cluster_index = np.ravel_multi_index(winner_coordinates, (som_x_,som_y_))
    cluster_index

    df_results = []
    for idx,adm in enumerate(lst_adms_):
        row = [adm, mort_dict_[adm], cluster_index[idx]]
        df_results.append(row)
    df_results = pd.DataFrame(df_results , columns=['admission_id', 'mortality', 'clusterSOM'])
    return som, df_results
    print("SOM of ", som_x_*som_y_, "clusters Algorithm elapsed", time.time()-t)
# ===================================================================================
# ===================================================================================
# ===================================================================================
# ===================================================================================

def plot_som_series_dba_center(som_x, som_y, win_map, feature):
    t = time.time()
    fig, axs = plt.subplots(som_x,som_y,figsize=(30,30))
    #fig.suptitle('Clusters')
    for x in range(som_x):
        for y in range(som_y):
            cluster = (x,y)
            if cluster in win_map.keys():
                for series in win_map[cluster]:
                    axs[cluster].plot(series,c="gray",alpha=0.5) 
                axs[cluster].plot(dtw_barycenter_averaging(np.vstack(win_map[cluster])),c="red") # I changed this part
            cluster_number = x*som_y+y+1
            axs[cluster].set_title(f"Cluster {cluster_number}", fontsize = 20)
            axs[cluster].tick_params(axis='both', labelsize=15)

    fig.text(0.5, 0.1, 'Time step', ha='center', fontsize = 25)
    fig.text(0.1, 0.5, 'Scaled ' + feature, va='center', rotation='vertical', fontsize = 25)
    plt.savefig('Images/Clustering_clusters_SOM_' + feature + '_'+ str(som_x*som_y)+'.png', transparent = True, bbox_inches = "tight")
    plt.show()
    print("SOM plot time elapsed", time.time()-t)
    
# ===================================================================================
# ===================================================================================
# ===================================================================================
# ===================================================================================
def get_barplot_cluster_maps(df_results, cluster_col, feature, n_clust):
    df_ = df_results.groupby(by = ['mortality',cluster_col]).count()

    clsts0 = df_[df_.index.get_level_values(0) == 0].index.get_level_values(1)
    miss_0 = set(range(n_clust)) - set(clsts0)
    for cl in miss_0:
        df_.loc[(0, cl), :] = 0
    clsts1 = df_[df_.index.get_level_values(0) == 1].index.get_level_values(1)
    miss_1 = set(range(n_clust)) - set(clsts1)
    for cl in miss_1:
        df_.loc[(1, cl), :] = 0
            
    df_ = df_.sort_values(by = ['mortality',cluster_col])

    y0 = df_[df_.index.get_level_values(0) == 0]['admission_id'].values
    y1 = df_[df_.index.get_level_values(0) == 1]['admission_id'].values
    
    plt.figure(figsize=(15,6))
    plt.title("Cluster Distribution for SOM", fontsize  = 20)
    plt.bar(range(len(y0)),y0, label = 'Discharged', color ='#9fc2e0', edgecolor='gray')
    plt.bar(range(len(y1)),y1, label = 'Deceased', color =  '#FA8072', edgecolor='gray', bottom = y0)
    
    plt.xticks(range(len(y0)), ["Cluster {price:.0f}".format(price = x+1) for x in range(len(y0))], rotation =90)
    plt.ylabel('Number of admins in cluster', fontsize = 18)
    plt.tick_params(axis='both', labelsize=17)
    plt.legend(fontsize = 20, frameon=False)
    plt.title(cluster_col + ' ' + feature + ' no clsuters' + str(len(y0)), fontsize = 15)
    plt.savefig('Images/Clustering_BarPlot_'+ cluster_col + '_' + feature + '_'+ str(len(y0))+'.png', transparent = True, bbox_inches = "tight")
    
    plt.show()
# ===================================================================================
# ===================================================================================
# ===================================================================================
# ===================================================================================    
def plot_KmnKs_series_centre(X_bis_, y_pred_, k_clust, som_x_, som_y_, n_clust_, feature, name ):
    fig = plt.figure(figsize=(30,30))
    for yi in range(n_clust_):
        plt.subplot(som_x_, som_y_, yi + 1)
        
        for xx in X_bis_[y_pred_ == yi]:
            plt.plot(xx.ravel(), "gray", alpha=.2)
        if name != 'KKM' :plt.plot(k_clust.cluster_centers_[yi].ravel(), "r-")
        plt.title('Cluster %d' % (yi + 1), fontsize = 15)
        plt.ylim(0, 1)
        plt.tick_params(axis='both', labelsize=15)
    #fig.text(0.5, 0.1, 'Time step', ha='center', fontsize = 25)
    #fig.text(0.1, 0.5, 'Scaled ' + feature, va='center', rotation='vertical', fontsize = 25)
    plt.tick_params(axis='both', labelsize=15)
    plt.savefig('Images/Clustering_clusters_'+ name + '_' + feature + '_'+ str(som_x_*som_y_)+'.png', transparent = True, bbox_inches = "tight")
    #plt.title("Euclidean $k$-means")
    plt.show()