import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

#filterFileSortByClient()和conversionByClient()将transaction中的内容根据client名字排序，并且过滤到重复加入的title
def conversionByClient():
    data = pd.read_csv('E:/trainData/financial/841/data/transaction.csv',encoding='UTF-8')
    print(len(data))
    price = copy.deepcopy(data[["客户帐号","客户姓名","客户编号","交易金额","手续费","贷方发生额","借方发生额","帐户余额","对方帐号32","摘要描述"]])
    print(len(price))
    # print(price.describe())
    client = copy.deepcopy(data[["客户帐号","客户姓名"]])
    client.drop_duplicates(subset = ['客户姓名'],keep = 'first' ,inplace = True)
    clientData = client.as_matrix()
    length = len(clientData)
    for i in range(0,length):
        # p = price[price["客户帐号"] == clientData[i][0]][["交易金额"]].as_matrix()
        p = price[price["客户姓名"] == clientData[i][1]]
        p.to_csv("E://trainData//financial//841//data//sortByClient.csv", index=False,mode = 'a')
def filterFileSortByClient():
    data = pd.read_csv('E:/trainData/financial/841/data/sortByClient.csv',encoding='UTF-8')
    print(len(data.as_matrix()))
    p = data[data["客户帐号"] != "客户帐号"]
    p.to_csv('E:/trainData/financial/841/data/sortByClient1.csv',encoding='UTF-8')
#将原始txt文件转换成csv格式的数据，由于原始数据中存在一些问题，因此转换中过滤掉了那部分数据
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
                 if len(line.split(',')) == 36:
                    # print(line)
                    dstfile.write(line)
def filter():
    data = pd.read_csv('E:/trainData/financial/841/data/transaction.csv',encoding='UTF-8')
    afterFilter = data[~data["摘要描述"]
        .isin( ["税后利息","手续费","话费","水费","材料费","账户管理费","查询费","错帐调整","运杂费","劳务费"])]
    afterFilter.to_csv("E://trainData//financial//841//data//filterData.csv", index=False,mode = 'a')
#统计交易信息中关于客户的总交易量，平均交易量，最大交易量和交易频数，并持久化
def countMean():
    data = pd.read_csv('E:/trainData/financial/841/data/transaction.csv',encoding='UTF-8')
    price = data[["客户帐号","客户姓名","客户编号","交易金额","手续费","贷方发生额","借方发生额","帐户余额","对方帐号32"]]
    size = len(data)
    client = price.as_matrix()
    result = {}
    count = 0
    for i in range(0,size):
        key = client[i][1]
        count = count + 1
        if count % 1000 == 0:
            print(count)
        if  key in result:
            account = result.get(key)
            account["price"] = account["price"] + client[i][3]
            account["count"] = account["count"] + 1
            if client[i][3] > account["max"]:
                account["max"] = client[i][3]
            result[key] = account
        else:
            account = {}
            account["name"] = client[i][1]
            account["account"] = client[i][0]
            account["price"] = client[i][3]
            account["max"] = client[i][3]
            account["count"] = 1
            result[key] = account
    with open('E:/trainData/financial/841/data/totalCount.csv', 'w',encoding='utf-8',newline='') as dstfile:
        header = ['name',"account","total","avg","frequency","max"]
        #写入方式选择wb，否则有空行
        writer = csv.DictWriter(dstfile, fieldnames=header)
        writer.writeheader()    #   写入表头
        keys = result.keys()
        for k in keys :
            account = result[k]
            mean = account["price"] / account["count"]
            line = account["name"]+","+account["account"]+","+str(account["price"])+","+str(mean)+","+str(account["count"])+","+str(account["max"])+'\n'
            dstfile.write(line)
        dstfile.close()
#根据交易类别，统计交易信息中关于客户的总交易量，平均交易量，最大交易量和交易频数，并持久化
#目前主要分两大类：现金交易 和 转账交易
def countMeanByType(type,name):
    storePath = 'E:/trainData/financial/841/data/count/%s.csv' % name

    data = pd.read_csv('E:/trainData/financial/841/data/transaction.csv',encoding='UTF-8')
    price = data[["客户帐号","客户姓名","客户编号","交易金额","手续费","贷方发生额","借方发生额","帐户余额","对方帐号32","摘要描述"]]
    # cashType =["现金支付","现金存入","ATM取款","网络ATM取款","ATM存款"]
    afterPrice = price[price["摘要描述"].isin(type)]
    size = len(afterPrice)
    client = afterPrice.as_matrix()
    result = {}
    count = 0
    for i in range(0,size):
        key = client[i][1]
        count = count + 1
        if count % 1000 == 0:
            print(count)
        if  key in result:
            account = result.get(key)
            account["price"] = account["price"] + client[i][3]
            account["count"] = account["count"] + 1
            if client[i][3] > account["max"]:
                account["max"] = client[i][3]
            result[key] = account
        else:
            account = {}
            account["name"] = client[i][1]
            account["account"] = client[i][0]
            account["price"] = client[i][3]
            account["max"] = client[i][3]
            account["count"] = 1
            result[key] = account

    with open(storePath, 'w',encoding='utf-8',newline='') as dstfile:
        header = ['name',"account","total","avg","frequency","max"]
        #写入方式选择wb，否则有空行
        writer = csv.DictWriter(dstfile, fieldnames=header)
        writer.writeheader()    #   写入表头
        keys = result.keys()
        for k in keys :
            account = result[k]
            mean = account["price"] / account["count"]
            line = account["name"]+","+account["account"]+","+str(account["price"])+","+str(mean)+","+str(account["count"])+","+str(account["max"])+'\n'
            dstfile.write(line)
        dstfile.close()

#统计用户总体交易信息，但是很慢，目前改成countMean方法了
def count():
    #这种写法统计似乎很慢，我算是服了这个索引
    data = pd.read_csv('E:/trainData/financial/841/data/transaction.csv',encoding='UTF-8')
    # print(data.columns)
    # print(data.count())
    print('总数据量:')
    print(len(data))
    price = copy.deepcopy(data[["客户帐号","客户姓名","客户编号","交易金额","手续费","贷方发生额","借方发生额","帐户余额","对方帐号32"]])
    # print(price.describe())
    client = copy.deepcopy(data[["客户帐号","客户姓名"]])
    client.drop_duplicates(subset = ['客户帐号'],keep = 'first' ,inplace = True)
    # client.to_csv("E://trainData//financial//841//data//client.csv", index=False,mode = 'a')
    # print(client)
    clientData = client.as_matrix()
    length = len(clientData)
    client = pd.DataFrame(clientData,index =range(0,length),columns = ["客户帐号","客户姓名"])
    count = 0
    # c = price.groupby('客户帐号')["交易金额"].count()
    for i in range(0,length):
        # p = price[price["客户帐号"] == clientData[i][0]][["交易金额"]].as_matrix()
        p = price[price["客户帐号"] == clientData[i][1]][["交易金额"]].as_matrix()
        total = sum(p)
        size =  len(p)
        count = count + 1
        print(count)
        # client.loc[i,"客户帐号"] = clientData[i][0]
        # client.loc[i,"客户姓名"] = clientData[i][1]
        if size > 0 :
            client.loc[i,"总交易金额"] = total
            client.loc[i,"次数"] = size
            client.loc[i,"平均交易金额"] = total / size
    client.to_csv("E://trainData//financial//841//data//clientCount.csv", index=False,mode = 'a')
#将单个用户的信息持久化成相应文件
def storeByClient(name):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    data = pd.read_csv('E:/trainData/financial/841/data/transaction.csv',encoding='UTF-8')
    client = data[data["客户姓名"] == name][["客户帐号","客户姓名","客户编号","交易金额","手续费","贷方发生额","借方发生额","帐户余额","对方帐号32","摘要描述"]]
    client.to_csv("E://trainData//financial//841//data//clientInfo.csv", index=False,mode = 'a')

np.set_printoptions(suppress=True)

# count()
# conversion()
# clusterAnalysis()
# accountAnalysis("王秀兰")
# storeByClient("王秀兰")
# filter()
cashType = ["现金支付","现金存入","ATM取款","网络ATM取款","ATM存款"]
transferType = ["转账存入","转账支取","跨行转入","汇出境外"]
countMeanByType(cashType,"cash")
# countMean()
