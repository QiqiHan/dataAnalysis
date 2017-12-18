import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

def splitFrame(id,num):
    data[id] = chunk['xid,yid,date,hour,model1,model2,model3,model4,model5,model6,model7,model8,model9,model10,real_wind'].map(lambda x: x.split(',')[num])
    return data[id]
reader = pd.read_table('E:/BaiduYunDownload/points.csv', iterator=True)
with open('E:/testData.csv', 'r', newline='') as xxfile:
        chunk = reader.get_chunk(100000)

        data = pd.DataFrame(columns = ['xid','yid','date','hour','model1','model2','model3','model4','model5','model6','model7','model8','model9','model10','real_wind'])

        data['xid'] = chunk['xid,yid,date,hour,model1,model2,model3,model4,model5,model6,model7,model8,model9,model10,real_wind'].map(lambda x: x.split(',')[0])  # 分别处理新旧两列
        data['yid'] = chunk['xid,yid,date,hour,model1,model2,model3,model4,model5,model6,model7,model8,model9,model10,real_wind'].map(lambda x: x.split(',')[1])
        data['date'] = chunk['xid,yid,date,hour,model1,model2,model3,model4,model5,model6,model7,model8,model9,model10,real_wind'].map(lambda x: x.split(',')[2])
        data['hour'] = chunk['xid,yid,date,hour,model1,model2,model3,model4,model5,model6,model7,model8,model9,model10,real_wind'].map(lambda x: x.split(',')[3])
        splitFrame('model1',4)
        splitFrame('model2',5)
        splitFrame('model3',6)
        splitFrame('model4',7)
        splitFrame('model5',8)
        splitFrame('model6',9)
        splitFrame('model7',10)
        splitFrame('model8',11)
        splitFrame('model9',12)
        splitFrame('model10',13)
        splitFrame('real_wind',14)
        print(data)
        le_x=LabelEncoder().fit(data['xid'])
        X_label=le_x.transform(data['xid'])
        ohe_x=OneHotEncoder(sparse=False).fit(X_label.reshape(-1,1))
        X_ohe=ohe_x.transform(X_label.reshape(-1,1))

        le_y=LabelEncoder().fit(data['yid'])
        Y_label=le_y.transform(data['yid'])
        ohe_y=OneHotEncoder(sparse=False).fit(Y_label.reshape(-1,1))
        Y_ohe=ohe_y.transform(Y_label.reshape(-1,1))

        le_date=LabelEncoder().fit(data['date'])
        Date_label=le_date.transform(data['date'])
        ohe_date=OneHotEncoder(sparse=False).fit(Date_label.reshape(-1,1))
        Date_ohe=ohe_date.transform(Date_label.reshape(-1,1))

        le_hour=LabelEncoder().fit(data['hour'])
        Hour_label=le_hour.transform(data['hour'])
        ohe_hour=OneHotEncoder(sparse=False).fit(Hour_label.reshape(-1,1))
        Hour_ohe=ohe_hour.transform(Hour_label.reshape(-1,1))


        train_x=data[['xid','yid','date','hour','model1','model2','model3','model4','model5','model6','model7','model8','model9','model10']].as_matrix()
        train_y=data[['data']].as_matrix()

        x_tr_i,x_te_i,y_tr,y_te=train_test_split(train_x,train_y,test_size=0.3,random_state=0)
        x_tr=x_tr_i[:,:]
        x_te=x_te_i[:,:]

        print(x_te[1])

        randomf = RandomForestRegressor(n_estimators=100,max_depth=5,random_state=0)
        randomf.fit(x_tr,y_tr)
        print(randomf.score(x_te,y_te))

        gdbt = GradientBoostingRegressor(n_estimators=600,max_depth=5,random_state=0)
        gdbt.fit(x_tr,y_tr)
        print(gdbt.score(x_te,y_te))
        with open('E:/testFinal.csv', 'w', newline='') as rrfile:

                xxfile.readline()
                for line in xxfile:
                        linetest = line.strip('\n').split(',')
                        a = np.array(linetest)
                        print(a)
                        data_Y = gdbt.predict([a])
                        linetest.append(str(data_Y[0]))
                        print(linetest)
                        line0 = ','.join(linetest) + '\n'
                        print(line0)
                        rrfile.write(line0)


