import pandas as pd
import matplotlib.pyplot as plt


# reader = pd.read_csv('E:\\match\\1213\\new_testData201712.csv',iterator = True)
# plot_importance(model)
# plt.show()
def Count():
    reader = pd.read_csv('E:\\match\\1213\\allTrainData_1213.csv',iterator = True)
    chunkSize = 100000
    loop = True
    count = 0;
    size = 0;
    while loop:
        try:
            chunk = reader.get_chunk(chunkSize)
            chunk['num'] = chunk.model1.apply(lambda x: 1 if x < 15 else 0)
            columns = ['model2','model3','model4','model5','model6','model7','model8','model9','model10']
            for var in columns:
                chunk['num'] =chunk['num'] + chunk[var].apply(lambda x: 1 if x < 15 else 0)
            y=chunk[['data']].as_matrix()
            # Y = chunk[['model1','model2','model3','model4','model5','model6','model7','model8','model9','model10']].mean(axis = 1).as_matrix()
            # chunk['mean'] = chunk[['model1','model2','model3','model4','model5','model6','model7','model8','model9','model10']].mean(axis = 1)
            Y = chunk['num'].as_matrix()
            size = size + len((Y))
            for i in range(0,len(Y)):
                if Y[i] < 5 and y[i] < 15:
                    count = count+1
                if Y[i] >= 5 and y[i] >= 15:
                    count = count+1
        except StopIteration:
            loop = False
            print("count:")
            print(count)
            print("size:")
            print(size)
            print(count/size)
            print ("Iteration is stopped.")
def figure():
    # x547 y421
    reader = pd.read_csv('E:\\match\\1213\\allTrainData_1213.csv',iterator = True)
    count = 1*18+3
    while count > 0:
        count = count-1
        reader.get_chunk(230693)
    chunk = reader.get_chunk(230693)
    plt.subplot2grid((2,5),(0,0))
    plt.xlabel(u"model1")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    data1 = chunk.loc[(chunk['data'] > 15) & (chunk['model1'] <= 15)]
    # chunk.loc[(chunk.data >15)&(chunk.model1 <= 15)]['label'] =1
    data = chunk.loc[(chunk['data'] <= 15) & (chunk['model1'] > 15)]
    plt.scatter(data1.xid,data1.yid ,c= 'k')
    plt.scatter(data.xid,data.yid,c = 'k')


    plt.subplot2grid((2,5),(0,1))
    plt.xlabel(u"model2")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    data1 = chunk.loc[(chunk['data'] > 15) & (chunk['model2'] <= 15)]
    # chunk.loc[(chunk.data >15)&(chunk.model1 <= 15)]['label'] =1
    data = chunk.loc[(chunk['data'] <= 15) & (chunk['model2'] > 15)]
    plt.scatter(data1.xid,data1.yid ,c= 'k')
    plt.scatter(data.xid,data.yid,c = 'k')

    plt.subplot2grid((2,5),(0,2))
    plt.xlabel(u"model3")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    data1 = chunk.loc[(chunk['data'] > 15) & (chunk['model3'] <= 15)]
    # chunk.loc[(chunk.data >15)&(chunk.model1 <= 15)]['label'] =1
    data = chunk.loc[(chunk['data'] <= 15) & (chunk['model3'] > 15)]
    plt.scatter(data1.xid,data1.yid ,c= 'k')
    plt.scatter(data.xid,data.yid,c = 'k')

    plt.subplot2grid((2,5),(0,3))
    plt.xlabel(u"model4")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    data1 = chunk.loc[(chunk['data'] > 15) & (chunk['model4'] <= 15)]
    # chunk.loc[(chunk.data >15)&(chunk.model1 <= 15)]['label'] =1
    data = chunk.loc[(chunk['data'] <= 15) & (chunk['model4'] > 15)]
    plt.scatter(data1.xid,data1.yid ,c= 'k')
    plt.scatter(data.xid,data.yid,c = 'k')

    plt.subplot2grid((2,5),(0,4))
    plt.xlabel(u"model5")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    data1 = chunk.loc[(chunk['data'] > 15) & (chunk['model5'] <= 15)]
    # chunk.loc[(chunk.data >15)&(chunk.model1 <= 15)]['label'] =1
    data = chunk.loc[(chunk['data'] <= 15) & (chunk['model5'] > 15)]
    plt.scatter(data1.xid,data1.yid ,c= 'k')
    plt.scatter(data.xid,data.yid,c = 'k')

    plt.subplot2grid((2,5),(1,0))
    plt.xlabel(u"model6")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    data1 = chunk.loc[(chunk['data'] > 15) & (chunk['model6'] <= 15)]
    # chunk.loc[(chunk.data >15)&(chunk.model1 <= 15)]['label'] =1
    data = chunk.loc[(chunk['data'] <= 15) & (chunk['model6'] > 15)]
    plt.scatter(data1.xid,data1.yid ,c= 'k')
    plt.scatter(data.xid,data.yid,c = 'k')

    plt.subplot2grid((2,5),(1,1))
    plt.xlabel(u"model7")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    data1 = chunk.loc[(chunk['data'] > 15) & (chunk['model7'] <= 15)]
    # chunk.loc[(chunk.data >15)&(chunk.model1 <= 15)]['label'] =1
    data = chunk.loc[(chunk['data'] <= 15) & (chunk['model7'] > 15)]
    plt.scatter(data1.xid,data1.yid ,c= 'k')
    plt.scatter(data.xid,data.yid,c = 'k')

    plt.subplot2grid((2,5),(1,2))
    plt.xlabel(u"model8")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    data1 = chunk.loc[(chunk['data'] > 15) & (chunk['model8'] <= 15)]
    # chunk.loc[(chunk.data >15)&(chunk.model1 <= 15)]['label'] =1
    data = chunk.loc[(chunk['data'] <= 15) & (chunk['model8'] > 15)]
    plt.scatter(data1.xid,data1.yid ,c= 'k')
    plt.scatter(data.xid,data.yid,c = 'k')

    plt.subplot2grid((2,5),(1,3))
    plt.xlabel(u"model9")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    data1 = chunk.loc[(chunk['data'] > 15) & (chunk['model9'] <= 15)]
    # chunk.loc[(chunk.data >15)&(chunk.model1 <= 15)]['label'] =1
    data = chunk.loc[(chunk['data'] <= 15) & (chunk['model9'] > 15)]
    plt.scatter(data1.xid,data1.yid ,c= 'k')
    plt.scatter(data.xid,data.yid,c = 'k')

    plt.subplot2grid((2,5),(1,4))
    plt.xlabel(u"model10")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    data1 = chunk.loc[(chunk['data'] > 15) & (chunk['model10'] <= 15)]
    # chunk.loc[(chunk.data >15)&(chunk.model1 <= 15)]['label'] =1
    data = chunk.loc[(chunk['data'] <= 15) & (chunk['model10'] > 15)]
    plt.scatter(data1.xid,data1.yid ,c= 'k')
    plt.scatter(data.xid,data.yid,c = 'k')

    plt.show()
    # plt.scatter(data[:,0],data[:,1],c = data[:,15])
    # plt.show()


figure()