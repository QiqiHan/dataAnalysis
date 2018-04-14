import csv
import os
import pandas as pd

#将原始文件转换成csv格式的文件
def conversion(name):
    in_path = "E:/trainData/financial/data/ls/%s" % name
    out_path = "E:/trainData/financial/data/transaction/%s.csv" % name
    count = 0
    with open(in_path,'r',encoding='UTF-8') as file:
        with open(out_path, 'w',newline='',encoding='UTF-8') as dstfile:
            with open('E:/trainData/financial/data/transaction_title.txt','r',encoding='UTF-8') as titlefile:
                line = titlefile.readline()
                title = line.split(",")
                writer = csv.DictWriter(dstfile, fieldnames=title)
                writer.writeheader()    #   写入表头
                while True:
                    line = file.readline()
                    if not line:
                        break
                    # line = line.replace('',',')
                    line = line.replace(chr(1),',')
                    if len(line.split(',')) == 106  and line.find('"') == -1:
                        dstfile.write(line)
                titlefile.close()
        dstfile.close()
    file.close()

def conversionFiles(path):
    for file in os.listdir(path):
           print("deal:" + file)
           conversion(file)
def fileInfo():
    data = pd.read_csv('E:/trainData/financial/data/transaction/000000_0.csv',encoding='UTF-8')
    print(data.describe())

#过滤掉不需要的列，将文件持久化到新的目录
def filterFiles(path):
    for file in os.listdir(path):
        print("deal:" + file)
        filterFile(file)
def filterFile(name):
    in_path = "E:/trainData/financial/data/transaction/%s" % name
    out_path = "E:/trainData/financial/data/ts/%s" % name
    data = pd.read_csv(in_path)
    new_data = data[["编号","银行交易_交易时间","本方卡号","本方账号","交易类型","借贷标志","交易金额","交易余额","交易对方名称","交易对方账号"
                    ,"交易对方卡号","交易对方证件号码","交易对手余额","交易摘要"]]
    new_data.to_csv(out_path)


#用来转化文件的
path = "E:\\trainData\\financial\data\\transaction"
# conversionFiles(path)
# conversion("000002_0")
# fileInfo()
filterFiles(path)
# line = "250446174903,2017-03-21 00:00:00,,,010301100313089,,,,1694.64,,20170321,EK010000428255100,,,,,,,",9999,9999,428255100,EK010000,BTER,,,,,,,,,,,,,,,,,,,,,,乌鲁木齐市,,,,,,,,1665103092412378,20120625,王文义, wang wen yi,110001,65230219530515361X,中国,新疆维吾尔自治区,昌吉回族自治州,阜康市,010301100313089,,,,,,,,,,,,,,,,,,1694.64,,,,,,,,,,BTER,金融服务平台渠道,,,,,,,,,,,,99,"
# line = "250446173403,2017-07-23 00:00:00,,,045101101215096,,,,-244800.00,,20170723,EK010000475770582,郝景福,014201100133287,,34222219701011605X,,,网银转账,9999,9999,475770582,EK010000,EBNK,,,,,,,,,,,,,,,,,,,,,,乌鲁木齐市,,,,,,,,1637511313009908,20141015,杨立, yang li,110001,350181198407021613,中国,福建省,福州市,福清市,014201100133287,,045101101215096,, hao jing fu,,,,,,,,,,,,,,,244800.00,,,,,,,,,EBNK,网银渠道,,,,,,,,,,,,30,"
# print(line.find(chr(1)))
# lines = line.split(",")
# len(lines)