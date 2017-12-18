import pandas as pd
import numpy as np
import csv

def countData(x,train,model):
    print(model)
    y = np.array(train[model]).reshape(1,len(train))[0]
    count = 0
    for i in range(0,len(x)):
        if x[i] >= 15 and y[i] < 15:
            count = count+1
        if x[i] < 15 and y[i] >= 15:
            count = count+1
    print("count:")
    print(count)
    print(count/len(x))

def  predictP(z,y) :
    c = 0
    P = 0
    for i in range(0,len(y)):
        count1 = 0
        count2 = 0
        for j in range(0,10):
            if(z[i][j] < 15):
                count1 = count1 + 1
            else:
                count2 = count2 + 1
        c = c + 1
        if count1 > count2 and y[i] >= 15:
            P = P +1
        if count1 <= count2 and y[i] < 15 :
            P = P + 1
        if c % 10000 == 0 :
            print(c)
    print("错误率")
    print(P)
    print(P/len(y))



reader = pd.read_csv('E:\\match\\1213\\allTrainData_1213.csv',iterator = True)
# x=train[['model1','model2','model3','model4','model5','model6','model7','model8','model9','model10']]
# y=train[['data']]
# y = train.as_matrix(['data'])
# for i in range(1,11):
#     m = 'model'+str(i)
#     countData(y,train,m)
# print(len(y))

chunkSize = 100000
loop = True
with open('E:\\match\\1213\\allTrainLabelData.csv', 'w',newline='') as dstfile:
    header = ['xid', 'yid', 'date', 'hour','model0','model1','model2','model3','model4','model5','model6','model7','model8','model9','model10']
    #写入方式选择wb，否则有空行
    count = 0
    writer = csv.DictWriter(dstfile, fieldnames=header)
    writer.writeheader()    #   写入表头
    while loop:
        try:
            train = reader.get_chunk(chunkSize)
            z = train.as_matrix(['xid','yid','date','hour','data','model1','model2','model3','model4','model5','model6','model7','model8','model9','model10'])
            result = {}
            for i in range(0,len(z)):
                result['xid'] = z[i][0]
                result['yid'] = z[i][1]
                result['date'] = z[i][2]
                result['hour'] = z[i][3]
                for j in range(4,len(z[0])):
                    name = 'model' + str(j-4)
                    if z[i][j] >= 15 :
                        result[name] = 0
                    else: result[name] = 1
                writer.writerow(result)
                result = {}
                count = count + 1
                if count % 10000 == 0 :
                        print(count)
        except StopIteration:
            loop = False
            dstfile.close()
            print ("Iteration is stopped.")



