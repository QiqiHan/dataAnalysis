import pandas as pd

from sklearn.externals import joblib


model = joblib.load("train_model1.m")
reader = pd.read_csv('E:/BaiduYunDownload/windcla/model1cla.csv',iterator = True)
chunkSize = 100000
loop = True
count = 0
size = 0
while loop:
    try:
        chunk = reader.get_chunk(chunkSize)
        y=chunk[['real_wind']].as_matrix()
        Y = model.predict(chunk[['model1','model2','model3','model4','model5','model6','model7','model8','model9','model10']])
        # result = pd.DataFrame({'xid':chunk['xid'].as_matrix(),'yid':chunk['yid'].as_matrix(),'date':chunk['date'].as_matrix(),
        #                        'hour':chunk['hour'].as_matrix(), 'wind':Y.astype(np.float32)})
        # result.to_csv("E:\match\\1213\\testPoints.csv", index=False,mode = 'a')
        size = size + len((Y))
        for i in range(0,len(Y)):
            if Y[i] == 1 and y[i] == 0:
                count = count+1
            if Y[i] == 0 and y[i] == 1:
                count = count+1
    except StopIteration:
        loop = False
        print("count:")
        print(count)
        print("size:")
        print(size)
        print(count/size)
        print ("Iteration is stopped.")