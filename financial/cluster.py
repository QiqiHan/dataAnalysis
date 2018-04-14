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
import financial
from scipy.stats import pearsonr
import math

#关于聚类的优化
#目的：满足一级指标中数量，增加两个聚类的原因是因为统计总交易金额的聚类，表达能力有限，有些具体的交易细节不能表现，比如现金收付 或者款项划转情况
#增强最大交易额度这个维度的原因，为了表现规定限额这个条件，平均交易金额相对累积总交易金额，更有表现用户行为的能力

#聚类的目的是为了识别交易中的异常行为，因此这是一个识别离群点的问题
#目前聚类采取了总交易额与频数作为两个维度去识别离群点。根据论文中对于日最大交易金额的限制增加可以为该聚类增加最大交易额度这个维度
#交易中交易的类别主要涉及两大类，现金交易和转账交易
#增加两个聚类，分别是检测现金收付和款项划转去细化模型的表达能力

#需要考虑现金收付或款项转换占交易总数据中的频率

#在聚类的处理中，意识到一个问题:如何量化某点的具体信息即局部密度，用来反应该用户交易的异常程度。
#因此找到了一个算法—局部异常系数，用于检测点相对于临近点的密度偏差，该模型，相对于聚类更加具有表现力。

#正态分布 —>多元高斯分布 、卡方统计量
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
#http://scikit-learn.org/stable/auto_examples/covariance/plot_outlier_detection.html#sphx-glr-auto-examples-covariance-plot-outlier-detection-py
def main(name):
    # name = "totalCount"
    path = "E://trainData//financial//841//data//count//%s.csv" % name
    client_statistic = pd.read_csv(path,encoding='UTF-8')
    x = client_statistic[["avg","frequency","max"]].as_matrix()
    clf = LocalOutlierFactor(n_neighbors = 10)
    X = StandardScaler().fit_transform(x)
    y = clf.fit_predict(X)

    size = len(x)
    print("total size: " + str(size))
    out_path = "E://trainData//financial//841//data//threat//%s.csv"%name
    data = client_statistic.as_matrix()
    with open(out_path, 'w',newline='',encoding='UTF-8') as dstfile:
        title = ["name","account","total","avg","frequency","max"]
        writer = csv.DictWriter(dstfile, fieldnames=title)
        writer.writeheader()    #   写入表头
        count = 0
        for i in range(size):
            if y[i] == -1:
                # line = ",".join(str(data[i]))
                #过滤短期交易频率小于5且交易总额不大的账户
                if data[i][4] <= 5 and data[i][2] <= 100000:
                    continue
                #过滤短期交易频率较大但是交易总额不大的账户
                if data[i][4] <= 20 and data[i][2] <= 50000:
                    continue
                count = count + 1
                line = data[i][0]+","+str(data[i][1])+","+str(data[i][2])+","+str(data[i][3])+","+str(data[i][4])+","+str(data[i][5])+'\n'
                dstfile.write(line)
        dstfile.close()
        print("correlation size: "+str(count))
        print("可疑率：")
        print(count/size)
#计算交易相似性
#这里有一步操作是比较耗时的：根据name过滤出客户对应账号相对应的账户明细
def correlationAnalysis():
    data = pd.read_csv('E:/trainData/financial/841/data/transaction.csv',encoding='UTF-8')
    blackpd = pd.read_csv("E://trainData//financial//841//data//count//threatClient.csv",encoding = 'UTF-8')
    blackList = blackpd.as_matrix()
    size = len(blackList)
    print("total size: " + str(size))
    out_path = "E://trainData//financial//841//data//count//correlationClient.csv"
    count = 0
    with open(out_path, 'w',newline='',encoding='UTF-8') as dstfile:
        title = ["name","account","total","avg","frequency","max","in_p","out_p"]
        writer = csv.DictWriter(dstfile, fieldnames=title)
        writer.writeheader()    #   写入表头
        for i in range(size):
            in_p,out_p = detectInAndOut(blackList[i][0],data)
            if in_p > 0.2 or out_p > 0.2:
                line = blackList[i][0]+","+str(blackList[i][1])+","+str(blackList[i][2])\
                       +","+str(blackList[i][3])+","+str(blackList[i][4])+","+str(blackList[i][5])\
                       +","+str(in_p)+","+str(out_p)+'\n'
                dstfile.write(line)
                count = count + 1
            if count % 100 == 0 :
                print(count)
        dstfile.close()
    print("black size: "+str(count))
    print("相似率：")
    print(count/size)

def detectInAndOut(name,data):
    # data = pd.read_csv('E:/trainData/financial/841/data/transaction.csv',encoding='UTF-8')
    result_in = {}
    result_out = {}
    result_in_count = 0
    result_in_sum = 0
    result_out_count = 0
    result_out_sum = 0
    filter = data[data['客户姓名'] == name][["客户帐号","客户姓名","客户编号","交易金额","手续费","贷方发生额","借方发生额","帐户余额","对方帐号32","摘要描述"]]
    filterData = filter.as_matrix()
    #为了查看用户用心临时的操作
    # filter.to_csv("E:/trainData/financial/841/data/t.csv")
    size = len(filterData)
    inType = ["转帐存入"]
    outType = ["转帐支取"]
    #需要统计次数和金额
    for i in range(size):
        if filterData[i][9] in inType :
            if filterData[i][8] in result_in:
                account = result_in[filterData[i][8]]
                account["sum"] = account["sum"] + filterData[i][3]
                account["count"] = account["count"] + 1
                account["transaction"].append(filterData[i][3])
                result_in[filterData[i][8]] = account
            else:
                account = {}
                account["sum"] = filterData[i][3]
                account["transaction"] = [filterData[i][3]]
                account["count"] = 1
                result_in[filterData[i][8]] = account
            result_in_count = result_in_count + 1
            result_in_sum = result_in_sum + filterData[i][3]
        if filterData[i][9] in outType:
            if filterData[i][8] in result_out:
                account = result_out[filterData[i][8]]
                account["sum"] = account["sum"] + filterData[i][3]
                account["count"] = account["count"] + 1
                account["transaction"].append(filterData[i][3])
                result_out[filterData[i][8]] = account
            else:
                account = {}
                account["sum"] = filterData[i][3]
                account["count"] = 1
                account["transaction"] = [filterData[i][3]]
                result_out[filterData[i][8]] = account
            result_out_count = result_out_count + 1
            result_out_sum = result_out_sum + filterData[i][3]
    # 分散转入,集中转出的情况
    # 算法思想，先分析数据是否频繁交易
    # 在频繁交易的情况下，分析数据是 分散转入，集中转换还是集中转入，分散转出的情况
    # 这两种情况，转入、转出额度应该是差距不大，并且分散交易部分数据应该具有操作上的相似性
    # 提取频率较大账户的分散交易，打乱数据，转换成两个相同的数组,分析这两部分数组 是否具有操作上的相关性(相似性)
    # 该算法 假设洗黑钱的行为操作具有习惯性，即会频繁的进行相似的交易
    in_p = 0
    out_p = 0
    if len(result_in) != 0 and result_in_count > 5:
        in_p = detectCorrelation(result_in)
    if len(result_out) != 0 and result_out_count > 5:
        out_p = detectCorrelation(result_out)
    # if result_out_count >= 2 and result_in_count / result_out_count >= 4 :
    #     result = detectCorrelation(result_in)
    #     return result
    # elif result_in_count >= 2 and result_out_count / result_in_count >= 4 :
    #     result = detectCorrelation(result_out)
    #     return result
    #表示没有分析数据相关性的情况，因为不符合转入转出的要求
    return in_p,out_p

def detectCorrelation(result):

    #转入转出额度差距不是很大
    transaction = []
    keys = result.keys()
    #这部分代码可以优化，比如去掉频率很低的交易账户
    for key in keys:
        account = result[key]
        transaction.extend(account["transaction"])
    transaction_size = len(transaction)
    half = int(transaction_size/2)
    transaction_size = half * 2
    transaction_1 = transaction[0:half]
    transaction_2 = transaction[half : transaction_size]
    r,pvalue = pearsonr(transaction_1,transaction_2)
    #说明交易相关性比较高
    return r

#交易稳定性分析
def StabilityAnalysis():
    data = pd.read_csv('E:/trainData/financial/841/data/transaction.csv',encoding='UTF-8')
    blackpd = pd.read_csv("E://trainData//financial//841//data//count//threatClient.csv",encoding = 'UTF-8')
    blackList = blackpd.as_matrix()
    size = len(blackList)
    print("total size: " + str(size))
    out_path = "E://trainData//financial//841//data//count//stabilityClient.csv"
    count = 0
    with open(out_path, 'w',newline='',encoding='UTF-8') as dstfile:
        title = ["name","account","total","avg","frequency","max","rate","std"]
        writer = csv.DictWriter(dstfile, fieldnames=title)
        writer.writeheader()    #   写入表头
        for i in range(size):
            in_p,out_p = detectAccount(blackList[i][0],data)
            line = blackList[i][0]+","+str(blackList[i][1])+","+str(blackList[i][2]) \
                   +","+str(blackList[i][3])+","+str(blackList[i][4])+","+str(blackList[i][5]) \
                   +","+str(in_p)+","+str(out_p)+'\n'
            dstfile.write(line)
            count = count + 1
        dstfile.close()


#分析用户的个人行为
#一段时间交易行为的均值和方差判断异常操作的次数，即离均值u超过3倍方差的点所占比率
#方差描述数据的偏离程度，方差越大说明异常交易越多
def detectAccount(name,data):
    filter = data[data['客户姓名'] == name][["交易金额"]]
    filterData = filter.as_matrix()
    size = len(filterData)
    sum1 = 0
    sum2 = 0

    #对于数据量小于2的用户不采取操作
    if size <= 2:
        return 0,0

    for i in range(size):
        sum1 = sum1 +filterData[i]
        sum2 = sum2 +filterData[i]**2
    mean = sum1/size
    std = round(math.sqrt(sum2/size-mean**2) ,3)

    upperBound = mean + 3*std
    lowerBound = mean - 3*std
    if lowerBound < 0 :
        lowerBound = 0
    num = 0
    for i in range(size):
       if filterData[i] >= upperBound or filterData[i] <= lowerBound:
           num = num + 1
    rate = round(num / size,3)
    return rate,std


np.set_printoptions(suppress=True)

LofAndPlt("totalCount",50000,111)
# correlationAnalysis()
# main("totalCount")
# main("transferCount")
# data = pd.read_csv('E:/trainData/financial/841/data/transaction.csv',encoding='UTF-8')
# detectAccount("李玲",data)
# StabilityAnalysis()

# highDimensionCluster()
# Lof_("totalCount")
# Lof("totalCount",111,10000)
# LofAndPlt("totalCount",50000,111)