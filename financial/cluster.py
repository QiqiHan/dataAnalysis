import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import LocalOutlierFactor
from sklearn.externals import joblib


#关于聚类的优化
#目的：满足一级指标中数量，增加两个聚类的原因是因为统计总交易金额的聚类，表达能力有限，有些具体的交易细节不能表现，比如现金收付 或者款项划转情况
#增强最大交易额度这个维度的原因，为了表现规定限额这个条件，平均交易金额相对累积总交易金额，更有表现用户行为的能力

#聚类的目的是为了识别交易中的异常行为，因此是一个识别离群点的问题
#目前聚类采取了总交易额与频数作为两个维度去识别离群点。根据论文中对于日最大交易金额的限制增加可以为该聚类增加最大交易额度这个维度
#交易中交易的类别主要涉及两大类，现金交易和转账交易
#增加两个聚类，分别是检测现金收付和款项划转去细化模型的表达能力

#在聚类的处理中，意识到一个问题:如何量化某点的具体信息即局部密度，用来反应该用户交易的异常程度。
#因此找到了一个算法—局部异常系数，用于检测点相对于临近点的密度偏差，个人认为该模型，相对于聚类更加具有表现力。
def Dbscan_(name):
    Dbscan(name,0)
def DbscanAndPlt(name,nrows,subplt):
    clf,x,y_pred = Dbscan(name,nrows)
    print(y_pred)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    ax = plt.subplot(subplt,projection = '3d')
    ax.scatter(x[:, 0], x[:, 1],x[:,2] , c=y_pred)
    ax.set_xlabel('avg')
    ax.set_ylabel('freq')
    ax.set_zlabel('max')
    plt.show()
def DbscanAndDump(name,nrows):
    path = "E://trainData//financial//841//data//model//%s.model" % name
    clf,x,y_pred = Dbscan(name,nrows)
    joblib.dump(clf,path)
def loadDbscan(model):
    path = "E://trainData//financial//841//data//model//%s.model" % model
    clf = joblib.load(path)
    return clf
def Dbscan(name,rows):
    path = "E://trainData//financial//841//data//count//%s.csv" % name
    if rows != 0:
        client_statistic = pd.read_csv(path,encoding='UTF-8',nrows = rows)
    else:
        client_statistic = pd.read_csv(path,encoding='UTF-8')
    x = client_statistic[["avg","frequency","max"]].as_matrix()
    #DBSCAN
    X = StandardScaler().fit_transform(x)
    dbscan = DBSCAN(eps = 0.2, min_samples = 2)
    y_pred = dbscan.fit_predict(X)
    # plt.subplot2grid((1,2),(0,1))
    labels = dbscan.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # core = dbscan.core_sample_indices_
    print(y_pred)
    print(n_clusters_)
    plt.show()
    # centroids = dbscan.
    return dbscan,x,y_pred
def Lof_(name):
    Lof(name,0)
def LofAndDump(name, nrows):
    path = "E://trainData//financial//841//data//model//%s.model" % name
    clf,x,y_pred = Lof(name,nrows)
    joblib.dump(clf,path)
def LofAndPlt(name,nrows,subplt):
    clf,x,y_pred = Lof(name,nrows)
    print(y_pred)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    ax = plt.subplot(subplt,projection = '3d')
    ax.scatter(x[:, 0], x[:, 1],x[:,2] , c=y_pred)
    ax.set_xlabel('avg')
    ax.set_ylabel('freq')
    ax.set_zlabel('max')
    ax.set_title(name)
    plt.show()
def loadLof(model):
    path = "E://trainData//financial//841//data//model//%s.model" % model
    clf = joblib.load(path)
    return clf
def Lof(name,rows):
    path = "E://trainData//financial//841//data//count//%s.csv" % name
    if rows != 0:
        client_statistic = pd.read_csv(path,encoding='UTF-8',nrows = rows)
    else:
        client_statistic = pd.read_csv(path,encoding='UTF-8')
    x = client_statistic[["avg","frequency","max"]].as_matrix()
    outliers_fraction = 0.2
    # clf =  LocalOutlierFactor(n_neighbors = 10,contamination=outliers_fraction)
    clf =  LocalOutlierFactor(n_neighbors = 20)
    X = StandardScaler().fit_transform(x)
    y_pred = clf.fit_predict(X)
    scores_pred = clf.negative_outlier_factor_
    print(scores_pred)
    return clf,x,y_pred

    # z = clf.decision_function(np.c_[])
np.set_printoptions(suppress=True)
# highDimensionCluster()
# Lof_("totalCount")
# Lof("totalCount",111,10000)
# LofAndPlt("totalCount",50000,111)