# -*- coding: utf-8 -*-
"""
Completed on Thu Dec 19 18:44:40 2018
Workdir = F:\jTKount\1215 Work_Travel_company_tem[
Filename = Cluster3d.py
Describe: Cluster the 2D space with timline in 3D；
          Using KMeans/DBSCAN/GMM/AP refer to sklearn
          Consider the outlier detection
@author: OrenLi1042420545
"""

import os
#import csv
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn import mixture
from sklearn import metrics
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
from numpy import array, zeros, inf
from numpy.linalg import norm
#from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from pyexcel_xls import get_data
#from dateutil.parser import parse
from collections import defaultdict

def dtw(x, y, dist=lambda x, y: norm(x-y, ord=1)):
    x = array(x)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    y = array(y)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    r, c = len(x), len(y)
    D = zeros((r+1, c+1))
    D[0, 1:] = inf
    D[1:, 0] = inf
    for i in range(r):
        for j in range(c):
            D[i+1, j+1] = dist(x[i], y[j])
    for i in range(r):
        for j in range(c):
            D[i+1, j+1] += min(D[i, j], D[i, j+1], D[i+1, j])
    D = D[1:, 1:]
    dist = D[-1, -1]
    return dist

def read_xls_file(path):
    xls_data = get_data(path)
    print("Get data type:", type(xls_data))
    for sheet_n in xls_data.keys():
        print(sheet_n, ":", xls_data[sheet_n])
    return xls_data

def second2time(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h>23:
        h -= 24
    return ("%d:%02d:%02d" % (h, m, s))

def get_counts2(sequence):
    counts = defaultdict(int) 
    for x in sequence:
        counts[x] += 1
    return counts

def cal_hour2day(time):
    day_double = (time.hour+time.minute/60.0)/24.0
    return day_double
      
if __name__ == '__main__':
#   sourcedata = pd.read_csv("./Rand_Gen_GPS3d数据.csv")
# #############################################################################    
    method = 'KMeans'  
    # ['DBSCAN', 'KMeans', 'GMM', 'AP']
    add_time_ornot = 0 
    # decide when proprecess data;add time in cluster-fit or not
    model_sel = 'aic' 
    # only in GMM sel_model_param
    activity_type = '睡觉'
    # choose datas filter by this to decide special activities like sleeping
    outlierDetect = 1 
    # only under KMeans when activity_type not None; clean noise to show
# #############################################################################
# reading data setload
    pth_str = r'Travel1215.xlsx'
    outPath = os.path.split(pth_str)
    outType = os.path.splitext(outPath[1])[1]
    outName = os.path.splitext(outPath[1])[0]
    file_name = outName
    os.getcwd()
# changing working dir
    os.chdir('F:\\jTKount\\1215 Work_Travel_company_tem[')  
# #############################################################################
#  read orgin data
    dic_TimSpace = read_xls_file(pth_str)
    df_Tsp = pd.DataFrame(dic_TimSpace['GPSies'][1:], 
                          columns=dic_TimSpace['GPSies'][0])
    df_dat = np.array(df_Tsp[['Longitude', 'Latitude','Time']])
#  fit the scale  
    ScalaMM = MinMaxScaler().fit(df_dat)
# #############################################################################
# reading the timedivide dictionary
    pth_timeDiv = 'dic_trail_range.xlsx'
    dic_timeDiv = read_xls_file(pth_timeDiv)
    df_timD = pd.DataFrame(dic_timeDiv['Sheet1'][1:], 
                           columns=dic_timeDiv['Sheet1'][0])
    df_timDt = df_timD[['start', 'end']].T
    df_timDt.columns = df_timD['活动类型']
# 转换一下时间的数据格式 从小时制转成float的day的数值类型
    df_timDt = df_timDt.applymap(lambda x: cal_hour2day(x))
# #############################################################################
# filter the data
    if activity_type != None:
        if df_timDt[activity_type]['start'] > df_timDt[activity_type]['end']:
            df_dat = df_dat[(df_dat[:,2] > df_timDt[activity_type]['start']) | 
                            (df_dat[:,2] < df_timDt[activity_type]['end'])]
        else:
            df_dat = df_dat[(df_dat[:,2] > df_timDt[activity_type]['start']) & 
                            (df_dat[:,2] < df_timDt[activity_type]['end'])]  
# scale the data    
    DF_dd = ScalaMM.transform(df_dat)
# ############################################################################## 
# Example settings
    outliers_fraction = 0.15
# model_set param     
    anomaly_algorithms = [
            ("Robust covariance", 
             EllipticEnvelope(contamination=outliers_fraction)),
            ("One-Class SVM", 
             svm.OneClassSVM(nu=outliers_fraction, 
                             kernel="rbf",gamma=0.1)),
            ("Isolation Forest", 
             IsolationForest(contamination=outliers_fraction,
                             random_state=42)),
            ("Local Outlier Factor", 
             LocalOutlierFactor(n_neighbors=35, 
                                contamination=outliers_fraction))]
# ##############################################################################  
# fit the model    
    if add_time_ornot:
        DF_use = DF_dd    
    else:
        DF_use = DF_dd[:,:2]
    if method == 'DBSCAN':
        db = DBSCAN(eps=.009, min_samples = 30, metric=dtw, algorithm='auto', 
                    leaf_size=80, p=None, n_jobs=-1).fit(DF_use)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] =True    
        labels = db.labels_
    elif method == 'KMeans':
        # k means determine k
        silhouette = []
        K = range(2, 8)
        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(DF_use)
            silhouette.append(
                    metrics.silhouette_score(DF_use, 
                                             kmeanModel.labels_,
                                             metric='euclidean'))
            index = np.argmax(silhouette)
        kmeans_bestmodel = KMeans(n_clusters=K[index]).fit(DF_use)           
        labels = kmeans_bestmodel.labels_
        cluster_point = kmeans_bestmodel.cluster_centers_
        # anomaly&outlier detection in Kmeans
        if activity_type != None:
            for name, algorithm in anomaly_algorithms:
                if name == "Local Outlier Factor":
                    outlier_ = (algorithm.fit_predict(DF_use) + 1) // 2
                else:
                    outlier_ = (algorithm.fit(DF_use).predict(DF_use) + 1) // 2
            core_samples_mask = np.zeros_like(
                    kmeans_bestmodel.labels_, dtype=bool)
            core_samples_mask[outlier_==1] =True
        else:
            core_samples_mask = np.zeros_like(
                    kmeans_bestmodel.labels_, dtype=bool)
            core_samples_mask = True
    elif method == 'GMM':
        if model_sel == 'aic':
            lowest_aic = np.infty
            aic = []
        elif model_sel == 'bic':
            lowest_bic = np.infty
            bic = []
        n_components_range = range(2, 8)
        cv_types = ['spherical', 'tied', 'diag', 'full']
        max_iters = range(20, 100, 10)
        Init_Params = ['kmeans', 'random']
        for cv_type in cv_types:
            for n_components in n_components_range:
                for max_iter in max_iters:
                    for init_params in Init_Params:
                        # Fit a Gaussian mixture with EM
                        gmm = mixture.GaussianMixture(
                                n_components = n_components,
                                max_iter = max_iter,
                                init_params = init_params,
                                covariance_type = cv_type)
                        gmm.fit(DF_use)
                        # select the gmm-model by aic or bic
                        if model_sel == 'aic':
                            aic.append(gmm.aic(DF_use))
                            if aic[-1] < lowest_aic:
                                lowest_aic = aic[-1]
                                best_gmm = gmm      
                        elif model_sel == 'bic':
                            bic.append(gmm.bic(DF_use))
                            if bic[-1] < lowest_bic:
                                lowest_bic = bic[-1]
                                best_gmm = gmm
        labels = best_gmm.predict(DF_use)
        cluster_point = best_gmm.means_ 
    elif method == 'AP':
        af = AffinityPropagation(damping=.8, preference=-150).fit(DF_use)
        cluster_point = af.cluster_centers_
        labels = af.predict(DF_use)  
# #############################################################################
#  fig params and plt params; take colors from plt.cm.Spectral
    mpl.rcParams['legend.fontsize'] = 10
    plt.rcParams['font.sans-serif']=['SimHei'] # show the chinese label
    plt.rcParams['axes.unicode_minus']=False # show the '-' character
# noneed the pca;
#    pca = PCA(n_components=3).fit(DF_dd)   
#    pca_3d = pca.transform(DF_dd) 
#open here and help u decrease the feature and take key ones
    fig = plt.figure(1)
    fig.set_size_inches(25.1, 16.8)
    ax = fig.add_subplot(111, projection='3d')
    n_clusters_ = len(set(labels))  
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
 # ############################################################################
# control drawing; divide outlier point format and anomaly point format   
    DF_clusterCenter = pd.DataFrame()
    for k, col in zip(unique_labels, colors):
        if k == -1:
        # some grey for the noise/outlier points
            col = [0.6, 0.6, 0.6, 1]
        class_member_mask = (labels == k) # take the kth classes label 
        if method == 'DBSCAN' or 'KMeans':
            xy = DF_dd[class_member_mask & core_samples_mask]
            type1 = ax.plot(xy[:, 0], xy[:, 1], xy[:, 2], 'o', 
                            markerfacecolor=tuple(col), # color to tuple
                            markeredgecolor='k', markersize=12)
        elif method == 'GMM' or 'AP':
            xy = DF_dd[class_member_mask]
            type1 = ax.plot(xy[:, 0], xy[:, 1], xy[:, 2], 'o', 
                            markerfacecolor=tuple(col), # color to tuple
                            markeredgecolor='k', markersize=8)  
        if method == 'DBSCAN' or 'KMeans':
            xy = DF_dd[class_member_mask & ~core_samples_mask]
            type2 = ax.plot(xy[:, 0], xy[:, 1], xy[:, 2], 'o', 
                            markerfacecolor=tuple(col),
                            markeredgecolor='k', 
                            markersize=4, alpha=0.08) # draw noises
            if k != -1:
                DF_clusterCenter.loc[k, 'k'] = k + 1
                DF_clusterCenter.loc[k, 'GPS_Longitude'] = np.mean(
                        DF_dd[class_member_mask & core_samples_mask][:,0])
                DF_clusterCenter.loc[k, 'GPS_Latitude'] = np.mean(
                        DF_dd[class_member_mask & core_samples_mask][:,1])
                DF_clusterCenter.loc[k, 'Time'] = np.mean(
                        DF_dd[class_member_mask & core_samples_mask][:,2])
                if len(DF_dd[class_member_mask & core_samples_mask]):
                    DF_clusterCenter.loc[k, 'Time_st'] = np.min(
                            DF_dd[class_member_mask & core_samples_mask][:,2])
                    DF_clusterCenter.loc[k, 'Time_ed'] = np.max(
                            DF_dd[class_member_mask & core_samples_mask][:,2])
                else:
                    DF_clusterCenter_bak = DF_clusterCenter.copy()
                    DF_clusterCenter_bak.loc[k, 'Time_st'] = np.nan
                    DF_clusterCenter_bak.loc[k, 'Time_ed'] = np.nan
                    DF_clusterCenter = DF_clusterCenter_bak.dropna(axis=0,
                                                                   how='any')
        elif method == 'GMM' or 'AP':
            DF_clusterCenter.loc[k, 'k'] = k + 1
            DF_clusterCenter.loc[k, 'GPS_Longitude'] = cluster_point[k, 0]
            DF_clusterCenter.loc[k, 'GPS_Latitude'] = cluster_point[k, 1]
            if add_time_ornot:
                DF_clusterCenter.loc[k, 'Time'] = cluster_point[k, 2]
            else:
                DF_clusterCenter.loc[k, 'Time'] = np.mean(
                        DF_dd[class_member_mask][:,2])
            DF_clusterCenter.loc[k, 'Time_st'] = np.min(
                    DF_dd[class_member_mask][:,2])
            DF_clusterCenter.loc[k, 'Time_ed'] = np.max(
                    DF_dd[class_member_mask][:,2])
 # ############################################################################  
 # invertransform minMaxscale; add the time processing    
    DF_CCenter = ScalaMM.inverse_transform(
            DF_clusterCenter[['GPS_Longitude', 'GPS_Latitude', 'Time']])
    DF_CCenter = pd.DataFrame(
            DF_CCenter, columns=['GPS_Longitude', 'GPS_Latitude', 'Time'])
    DF_CCenter['k'] = DF_clusterCenter['k']
    DF_CCenter['Times'] = DF_CCenter['Time'].map(
            lambda x:second2time(x*24*3600))
    DF_CCenter['Times_st'] = DF_clusterCenter['Time_st'].map(
            lambda x:second2time((ScalaMM.data_range_[2]*x+
                                  ScalaMM.data_min_[2])*24*3600))
    DF_CCenter['Times_ed'] = DF_clusterCenter['Time_ed'].map(
            lambda x:second2time((ScalaMM.data_range_[2]*x+
                                  ScalaMM.data_min_[2])*24*3600))
    dic_ink = get_counts2(labels)
    DF_CCenter['num_ink'] = DF_CCenter.index.map(dic_ink)
 # ############################################################################
 # add the axis label & title; control the save png format
    ax.set_xlabel("GPS_Longitude")
    ax.set_ylabel("GPS_Latitude")
    ax.set_zlabel('Time')
    if method == "DBSCAN":
        plt.title(('Clustering of DBSCAN in 3D Scatter\n'  
                   'Estimates number of clusters: %d') % (n_clusters_-1))
    elif method == 'KMeans':
        plt.title(('Clustering of KMeans in 3D Scatter\n' 
                   ' Estimates number of clusters: %d') % (n_clusters_))
    elif method == 'GMM':
        plt.title(('Clustering of GMM in 3D Scatter\n'
                   ' Estimates number of clusters: %d') % (n_clusters_))
    elif method == 'AP':
        plt.title(('Clustering of AP in 3D Scatter\n'
                   ' Estimates number of clusters: %d') % (n_clusters_))
    plt.savefig(file_name+str(np.random.rand())+'特征3D聚类.png')
    plt.show()