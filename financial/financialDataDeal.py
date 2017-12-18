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

def conversion():
    with open('E:/trainData/financial/841/data/transaction.txt','r',encoding='UTF-8') as file:
        with open('E:/trainData/financial/841/data/transaction.csv', 'w',newline='',encoding='UTF-8') as dstfile:
             title = file.readline().split('$')
             writer = csv.DictWriter(dstfile, fieldnames=title)
             writer.writeheader()    #   写入表头
             while True:
                 line = file.readline()
                 if not line:
                     break
                 line = line.replace('$',',')
                 dstfile.write(line)
def count():
    data = pd.read_csv('E:/trainData/financial/841/data/transaction.csv',encoding='UTF-8')
    # print(data.columns)
    # print(data.count())
    print('总数据量:')
    print(len(data))
    price = copy.deepcopy(data[["客户帐号","客户姓名","客户编号","交易金额","手续费","贷方发生额","借方发生额","帐户余额","对方帐号32"]])
    # print(price.describe())
    client = copy.deepcopy(data[["客户帐号","客户姓名"]])
    client.drop_duplicates(subset = ['客户帐号'],keep = 'first' ,inplace = True)
    print(client)

    clientData = client.as_matrix()
    length = len(clientData)
    client = pd.DataFrame(clientData,index =range(0,length),columns = ["客户帐号","客户姓名"])
    # c = price.groupby('客户帐号')["交易金额"].count()
    for i in range(0,length):
        p = price[price["客户帐号"] == clientData[i][0]][["交易金额"]].as_matrix()
        total = sum(p)
        size =  len(p)
        # client.loc[i,"客户帐号"] = clientData[i][0]
        # client.loc[i,"客户姓名"] = clientData[i][1]
        client.loc[i,"总交易金额"] = total
        client.loc[i,"次数"] = size
        client.loc[i,"平均交易金额"] = total / size
    client.to_csv("E://trainData//financial//841//data//clientCount.csv", index=False,mode = 'a')
def figureShow():
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
    c = price.groupby('客户帐号')["交易金额"].count()
    print("客户交易次数")
    print(c)
    description = copy.deepcopy(data[["摘要代码","摘要描述"]])
    description.drop_duplicates(subset=['摘要代码'],keep = 'first',inplace =True)
    # print(description)
    # print(description.count())
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig = plt.figure()
    fig.set(alpha=0.1)  # 设定图表颜色alpha参数
    plt.subplot2grid((2,2),(0,0))
    plt.scatter(data["帐户余额"],data["交易金额"])
    plt.ylabel(u"交易金额")
    plt.xlabel(u"余额")
    client_statistic = pd.read_csv('E://trainData//financial//841//data//clientCount.csv',encoding='UTF-8')
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
def clusterAnalysis():
    client_statistic = pd.read_csv('E://trainData//financial//841//data//clientCount.csv',encoding='UTF-8')
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


np.set_printoptions(suppress=True)
figureShow()
# conversion()
# clusterAnalysis()