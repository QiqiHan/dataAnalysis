# coding=UTF-8

import csv

def read():
    res = {}
    count = 0
    with open('E:/match/In_situMeasurementforTraining_201712.csv','r') as file:
        file.readline()
        while True:
            line = file.readline();
            if not line:
                break
            row = line.split(',')
            key = '{}-{}-{}-{}'.format(row[0],row[1], row[2], row[3])
            res[key] = row[4]
            # count = count +1
            # if count % 100 == 0:
            #     print count
    return res

def write():
    # map = read()
    result = {}
    print ("start")
    with open('E:/match/ForecastDataforTesting_201712.csv', 'r') as csvfile:
        #过滤掉第一行标题
        with open('E:/match//1213//new_testData201712.csv', 'w',newline='') as dstfile:
            header = ['xid', 'yid', 'date', 'hour', 'model1','model2','model3','model4','model5','model6','model7','model8','model9','model10'
                      ,'m1','m2','m3','m4','m5','m6','m7','m8','m9','m10']
            #写入方式选择wb，否则有空行
            writer = csv.DictWriter(dstfile, fieldnames=header)
            writer.writeheader()    #   写入表头
            csvfile.readline()
            readers = []
            count = 0
            co = 0
            while count < 10:
                line = csvfile.readline()
                if not line:
                    break;
                count = count + 1
                readers.append(line)
            while(len(readers) != 0):
                # reader = csv.reader(csvfile)
                first = readers[0]
                first = first.split(',')
                result['xid'] = first[0]
                result['yid'] = first[1]
                result['date'] = first[2]
                result['hour'] = first[3]
                # key = '{}-{}-{}-{}'.format(first[0],first[1], first[2], first[3])
                # result['data'] = eval(map[key])
                num = 1
                for row in readers:
                    row = row.split(',')
                    key = 'model'+str(num)
                    key1 = 'm'+str(num)
                    result[key] = row[5].strip('\n')
                    n = float(row[5].strip('\n'))
                    if n < 15 :
                        result[key1] = 1
                    else:
                        result[key1] = 0
                    num = num + 1
                # print(result)
                # for key in result:
                #     writer.writerow([key,result[key]])
                writer.writerow(result)  # 批量写入
                co = co + 1
                if co%1000 == 0 :
                    print (co)
                result = {}
                readers = []
                count = 0
                while count < 10:
                    line = csvfile.readline()
                    if not line:
                        break;
                    count = count + 1
                    readers.append(line)
        dstfile.close()

def writeByFilter():
    map = read()
    result = {}
    print ("start")
    with open('E:/match/ForecastDataforTraining_201712.csv', 'r') as csvfile:
        #过滤掉第一行标题
        with open('E:/match/1213//allTrainData_13_17.csv', 'w',newline='') as dstfile:
            header = ['xid', 'yid', 'date', 'hour','data','model1','model2','model3','model4','model5','model6','model7','model8','model9','model10']
            #写入方式选择wb，否则有空行
            writer = csv.DictWriter(dstfile, fieldnames=header)
            writer.writeheader()    #   写入表头
            csvfile.readline()
            readers = []
            count = 0
            co = 0
            while count < 10:
                line = csvfile.readline()
                if not line:
                    break;
                count = count + 1
                readers.append(line)
            while(len(readers) != 0):
                # reader = csv.reader(csvfile)
                first = readers[0]
                first = first.split(',')
                result['xid'] = first[0]
                result['yid'] = first[1]
                result['date'] = first[2]
                result['hour'] = first[3]
                key = '{}-{}-{}-{}'.format(first[0],first[1], first[2], first[3])
                result['data'] = eval(map[key])
                name = 'model'
                num = 1
                s = 0
                for row in readers:
                    row = row.split(',')
                    key = 'model'+str(num)
                    result[key] = row[5].strip('\n')
                    num = num + 1
                    s = s + float(row[5])
                # total = sum([float(i) for i in list(result.values())[4:14]])
                abs_value = abs(15-s/10);
                # print(result)
                # for key in result:
                #     writer.writerow([key,result[key]])
                if abs_value < 2:
                    writer.writerow(result)  # 批量写入
                    co = co + 1
                    if co%1000 == 0 :
                        print (co)
                result = {}
                readers = []
                count = 0
                while count < 10:
                    line = csvfile.readline()
                    if not line:
                        break;
                    count = count + 1
                    readers.append(line)
        dstfile.close()

writeByFilter()
# write()