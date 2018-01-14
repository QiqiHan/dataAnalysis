import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#该方法是单个账户的信息的可视化
def accountAnalysis(name):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    data = pd.read_csv('E:/trainData/financial/841/data/transaction.csv',encoding='UTF-8')
    client = data[data["客户姓名"] == name][["交易金额","帐户余额"]]
    # client.hist()
    # client.plot(kind = 'density',subplots = True ,layout = (1,2))
    # plt.show()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig = plt.figure()
    fig.set(alpha=0.1)  # 设定图表颜色alpha参数
    fig.subplots_adjust(top = 0.9,bottom = 0.1)
    plt.subplot2grid((2,2),(0,0))
    plt.ylabel(u"次数")
    plt.xlabel(u"交易金额")
    plt.hist(client["交易金额"])
    plt.subplot2grid((2,2),(0,1))
    plt.ylabel(u"次数")
    plt.xlabel(u"余额")
    plt.hist(client["帐户余额"], color = 'g')
    plt.subplot2grid((2,2),(1,0))
    plt.ylabel(u"次数")
    plt.xlabel(u"交易金额")
    client["交易金额"].plot(kind = 'density')
    plt.subplot2grid((2,2),(1,1))
    plt.ylabel(u"次数")
    plt.xlabel(u"余额")
    client["帐户余额"].plot(kind = 'density',color = 'g')
    plt.show()

    # plt.subplot2grid((3,2),(2,0))
    # plt.scatter(client["交易金额"],client["帐户余额"])
#这个是对总体数据的可视化
def totalfigureShow():
    data = pd.read_csv('E:/trainData/financial/841/data/transaction.csv',encoding='UTF-8')
    print("数据统计:")
    print(data.count())
    print('总数据量:')
    print(len(data))
    price = copy.deepcopy(data[["客户帐号","客户姓名","客户编号","交易金额","手续费","贷方发生额","借方发生额","帐户余额","对方帐号32"]])
    print(price.describe())
    client = copy.deepcopy(data[["客户帐号","客户姓名"]])
    client.drop_duplicates(subset = ['客户帐号'],keep = 'first' ,inplace = True)
    print(client)
    clientData = client.as_matrix()
    length = len(clientData)
    client = pd.DataFrame(clientData,index =range(0,length),columns = ["客户帐号","客户姓名"])
    c = price.groupby('客户姓名')["交易金额"].count()
    print("客户交易次数")
    print(c)
    description = copy.deepcopy(data[["摘要代码","摘要描述"]])
    description.drop_duplicates(subset=['摘要代码'],keep = 'first',inplace =True)
    # description.to_csv("E://trainData//financial//841//data//description.csv",index = False ,mode = 'a')
    # print(description)
    # print(description.count())
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig = plt.figure()
    fig.set(alpha=0.1)  # 设定图表颜色alpha参数
    plt.subplot2grid((2,2),(0,0))
    plt.scatter(data["帐户余额"],data["交易金额"])
    plt.ylabel(u"交易金额")
    plt.xlabel(u"余额")
    client_statistic = pd.read_csv('E://trainData//financial//841//data//clientCount1.csv',encoding='UTF-8')
    plt.subplot2grid((2,2),(0,1))
    plt.scatter(client_statistic["次数"],client_statistic["平均交易金额"])
    plt.ylabel(u"平均交易金额")
    plt.xlabel(u"次数")
    plt.subplot2grid((2,2),(1,0))
    plt.xlabel("平均交易金额")
    plt.boxplot(client_statistic["平均交易金额"])
    plt.subplot2grid((2,2),(1,1))
    plt.xlabel("次数")
    plt.boxplot(client_statistic["次数"])
    plt.show()
#一个比较粗略的聚类模型
def clusterAnalysis():
    client_statistic = pd.read_csv('E://trainData//financial//841//data//clientCount1.csv',encoding='UTF-8')
    x = client_statistic[["平均交易金额","次数"]].as_matrix()
    x_,x_t=train_test_split(x,test_size=0.3,random_state=0)

    num = len(x_)
    clf = KMeans(n_clusters=2)
    clf.fit(x_)
    centroids = clf.labels_
    y_pre = clf.predict(x_t)
    scsocre = silhouette_score(x_,clf.labels_,metric='euclidean')
    print("轮廓系数")
    print(scsocre)

    mark = ['or', 'ob']
    plt.subplot2grid((1,2),(0,0))
    plt.xlabel(u"Kmeans")
    #画出所有样例点 属于同一分类的绘制同样的颜色
    for i in range(num):
        plt.plot(x_[i][0], x_[i][1], mark[clf.labels_[i]]) #mark[markIndex])
    mark = ['Dr', 'Db']
    # 画出质点，用特殊图型
    centroids =  clf.cluster_centers_
    for i in range(2):
        plt.plot(centroids[i][0], centroids[i][1], mark[i], markersize = 12)
    print(centroids) #显示中心点坐标
    # plt.show()

    #DBSCAN
    X = StandardScaler().fit_transform(x)
    dbscan = DBSCAN(eps = 0.2, min_samples = 2)
    y_pred = dbscan.fit_predict(X)
    plt.subplot2grid((1,2),(0,1))
    plt.xlabel(u"DBScan")
    plt.scatter(x[:, 0], x[:, 1], c=y_pred)
    plt.show()
    labels = dbscan.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    core = dbscan.core_sample_indices_
    print(core)
    # centroids = dbscan.