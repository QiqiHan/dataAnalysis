import csv
import pandas as pd
import numpy as np
import copy
from scipy.stats import pearsonr

#分析用户的个人行为
#一段时间交易行为的均值 和 方差 判断异常操作的次数，即离均值u超过3倍方差的点 所占比率

def detect():
    blackLists = []
    client  = pd.read_csv('E://trainData//financial//841//data//client.csv',encoding='UTF-8')
    data = pd.read_csv('E:/trainData/financial/841/data/transaction.csv',encoding='UTF-8')
    price = copy.deepcopy(data[["客户帐号","客户姓名","客户编号","交易金额","手续费","贷方发生额","借方发生额","帐户余额","对方帐号32"]])
    clientList = client.as_matrix()
    size = len(clientList)
    for i in range(size):
        name = clientList[i][1]
        client = price[price["客户姓名"] == name]
        # client.to_csv("E://trainData//financial//841//data//client.csv", index=False,mode = 'a')
        count = client[u'对方帐号32'].value_counts().as_matrix()
        clientdata = price[price["客户姓名"] == name][["交易金额"]].as_matrix()
        frequency = len(clientdata)
        max_count = max(count)
        if frequency == 0:
            continue
        maxPrice = max(clientdata)
        if frequency > 40:
            blackLists.append(name)
            continue
        if max_count/frequency > 0.7:
            blackLists.append(name)
            continue
        if maxPrice > 200000:
            blackLists.append(name)
            continue
        totalPrice = sum(clientdata)
        if totalPrice > 100000:
            blackLists.append(name)
            continue
        if detectInAndOut(name,data) == 1:
            blackLists.append(name)
            continue
        if detectDenseTransaction(name,data) == 1:
            blackLists.append(name)
            continue
    return blackLists
def cashBasis(name):
    data = pd.read_csv('E:/trainData/financial/841/data/transaction.csv',encoding='UTF-8')
    client = data[data['客户姓名'] == name][["客户帐号","客户姓名","客户编号","交易金额","手续费","贷方发生额","借方发生额","帐户余额","对方帐号32","摘要描述"]]
    filterData = data[data['客户姓名'] == name][["交易金额","摘要描述"]].as_matrix()
    payType =["现金支付","现金存入","ATM取款","网络ATM取款","ATM存款"]
    size = len(filterData)
    sum = 0
    frequency = 0
    maxValue = 0
    for i in range(size):
        if filterData[i][1] in payType:
            sum = sum + filterData[i][0]
            frequency = frequency +1
            maxValue = max(maxValue,filterData[i][1])
    return sum,frequency,maxValue
def moenyTransfer(name):
    data = pd.read_csv('E:/trainData/financial/841/data/transaction.csv',encoding='UTF-8')
    filterData = data[data['客户姓名'] == name][["交易金额","摘要描述"]].as_matrix()
    payType =["转账存入","转账支取","跨行转入","汇出境外"]
    size = len(filterData)
    sum = 0
    frequency = 0
    maxValue = 0
    for i in range(size):
        if filterData[i][1] in payType:
            sum = sum + filterData[i][0]
            maxValue = max(maxValue,filterData[i][0])
            frequency = frequency + 1
    return sum,frequency,maxValue
#检测短期内资金分散转入 集中转出或资金集中转入分散转出的情况
def detectInAndOut(name,data):
    # data = pd.read_csv('E:/trainData/financial/841/data/transaction.csv',encoding='UTF-8')
    result_in = {}
    result_out = {}
    result_in["count"] = 0
    result_in["sum"] = 0
    result_out["count"] = 0
    result_out["sum"] = 0
    filterData = data[data['客户姓名'] == name][["客户帐号","客户姓名","客户编号","交易金额","手续费","贷方发生额","借方发生额","帐户余额","对方帐号32","摘要描述"]].as_matrix()
    size = len(filterData)
    inType = ["转账存入"]
    outType = ["转账支取"]
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
                result_in[filterData[i][8]] = filterData[i][3]
            result_in["count"] = result_in["count"] + 1
            result_in["sum"] = result_in["sum"] + filterData[i][3]
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
                result_out[filterData[i][8]] = filterData[i][3]
            result_out["count"] = result_out["count"] + 1
            result_out["sum"] = result_out["sum"] + filterData[i][3]
    # 分散转入,集中转出的情况
    # 算法思想，先分析数据是否频繁交易
    # 在频繁交易的情况下，分析数据是 分散转入，集中转换还是集中转入，分散转出的情况
    # 这两种情况，转入、转出额度应该是差距不大，并且分散交易部分数据应该具有操作上的相似性
    # 提取频率较大账户的分散交易，打乱数据，转换成两个相同的数组,分析这两部分数组 是否具有操作上的相关性(相似性)
    # 该算法 假设洗黑钱的行为操作具有习惯性，即会频繁的进行相似的交易
    if len(result_in) == 0 or len(result_out) == 0 :
        return 0
    if result_out["count"] > 3 and result_in["count"] / result_out["count"] > 5 :
        result = detectCorrelation(result_in,result_out)
        return result
    elif result_in["count"] > 3 and result_out["count"] / result_in["count"] > 5 :
        result = detectCorrelation(result_out,result_in)
        return result
    return 0
def detectCorrelation(result_1,result_2):
    difference = abs(result_1["sum"]/result_2["sum"]-1)
    #转入转出额度差距不是很大
    if difference < 0.2:
        transaction = []
        keys = result_1.keys()
        #这部分代码可以优化，比如去掉频率很低的交易账户
        for key in keys:
            account = result_1[key]
            transaction.append(account["transaction"])
        transaction_size = len(transaction)
        transaction_1 = transaction[0:transaction_size/2]
        transaction_2 = transaction[transaction/2+1 : -1]
        r,pvalue = pearsonr(transaction_1,transaction_2)
        #说明交易相关性比较高
        if r > 0.8 :
            return 1
        else:
            return 0
    else:
        return 0
#检测短期内相同收付人之间频繁发生资金收付，且交易金额接近大额交易标准
def detectDenseTransaction(name,data):
    filterData = data[data['客户姓名'] == name][["客户帐号","客户姓名","客户编号","交易金额","手续费","贷方发生额","借方发生额","帐户余额","对方帐号32","摘要描述"]].as_matrix()
    size = len(filterData)
    result = {}
    result["sum"] = 0
    result["count"] = 1
    for i in range(size):
        key = filterData[i][8]
        if key in result :
            account = result[key]
            account["sum"] = account["sum"] + filterData[i][3]
            account["count"] = account["count"] + 1
            account["transaction"].append(filterData[i][3])
            result[key] = account
        else :
            account["name"] = filterData[i][1]
            account["sum"] =  filterData[i][3]
            account["count"] = 1
            account["transaction"] = [filterData[i][3]]
            result[key] = account
        result["sum"] = result["sum"] + filterData[i][3]
        result["count"] = result["count"] + 1
    keys = result.keys()
    if result["sum"] == 0 or result["count"] == 0 :
        return 0
    for key in keys:
        account = result[key]
        if account["count"] / result["count"] > 0.5  or account["count"] > 10:
            #接近大额交易标准
            if abs(account["sum"]/account["count"]*200000 - 1) < 0.25:
                return 1
    return 0

# detect()
# cashBasis("王秀兰")
# print(moenyTransfer("马丽娟"))