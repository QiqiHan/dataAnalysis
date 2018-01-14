import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from xgboost import plot_importance
import copy
import csv
#
# def computeAccuracy(y_submission,y_test):
#     size = len((y_submission))
#     count = 0
#     for i in range(0,len(y_submission)):
#         if y_submission[i] >= 15 and y_test[i] < 15:
#             count = count+1
#         if y_submission[i] < 15 and y_test[i] >= 15:
#             count = count+1
#     return 1-count/size
#
#
#
# # # train = pd.read_csv('E:\\match\\1213\\allTrainData_13_17.csv')
# train = pd.read_csv('E:\\match\\1213\\trainData_1213.csv' ,nrows = 1000000)
# train['num'] = train.model1.apply(lambda x: 1 if x < 15 else 0)
# columns = ['model2','model3','model4','model5','model6','model7','model8','model9','model10']
# for var in columns:
#     train['num'] =train['num'] + train[var].apply(lambda x: 1 if x < 15 else 0)
# print(train['num'])
# x=train[['xid','yid','hour','model1','model2','model3','model4','model5','model6','model7','model8','model9','model10','num']].as_matrix()
# y=train[['data']].as_matrix()
# y =np.array(y).reshape(1,len(y))[0]
# xgb_model = xgb.XGBRegressor(max_depth=6,
#                              learning_rate=0.1,
#                              n_estimators=500,
#                              silent=True,
#                              objective='reg:linear',
#                              nthread=3,
#                              subsample=0.8,
#                              seed = 1003);
# # x_,x_t,y_,y_t=cross_validation.train_test_split(x,y,test_size=0.3,random_state=0)
# skf = list(StratifiedKFold(y, 5,random_state= 111))
# for i, (train, test) in enumerate(skf):
#     print ("Fold", i)
#     X_train = x[train]
#     y_train = y[train]
#     X_test = x[test]
#     y_test = y[test]
#     xgb_model.fit(X_train, y_train)
#     y_submission = xgb_model.predict(X_test)
#     print("Score: %0.5f" % (computeAccuracy(y_submission, y_test)))





    # size = len((y_submission))
    # count = 0
    # for i in range(0,len(y_submission)):
    #     if y_submission[i] >= 15 and y_test[i] < 15:
    #         count = count+1
    #     if y_submission[i] < 15 and y_test[i] >= 15:
    #         count = count+1
    # print(count)
    # print(size)
    # print(count / size)



# xgb_model.fit(x_,y_)
# y_pre = xgb_model.predict(x_t);
# y_tt = np.array(y_t).reshape(1,len(y_pre))[0]
# count = 0
# for i in range(0,len(y_pre)):
#     if y_pre[i] >= 15 and y_tt[i] < 15:
#         count = count+1
#     if y_pre[i] < 15 and y_tt[i] >= 15:
#         count = count+1
# print("count:")
# print(count)
# print(count/len(y_pre))
# joblib.dump(xgb_model,'E:\\match\\1213\\xgb6.model')


# param_test1 = {"learning_rate" : [0.01,0.05,0.1,0.2], }
# gsearch1 = GridSearchCV(estimator = xgb_model,
#                         param_grid = param_test1,iid=False,cv=2,verbose=2 )
# gsearch1.fit(x,y)
# for score in gsearch1.grid_scores_:
#     print(score)
# print(gsearch1.best_params_)
# print(gsearch1.best_score_)
# xgb_model = joblib.load('E:\\match\\1213\\xgb.model')
# xgb_model.fit(x_,y_)
# y_pre = xgb_model.predict(x_t);
# joblib.dump(xgb_model,'E:\\match\\1213\\xgb.model')

# print("R-squared value:")
# print(xgb_model.score(x_t,y_t))
# print("mean_squared_error:")
# print(mean_squared_error(y_t,y_pre))
# print('mean_absolute_error:')
# print(mean_absolute_error(y_t,y_pre))
# print(".")
# count = 0
# xgb_model = joblib.load('E:\\match\\1213\\xgb.model')




model = joblib.load('E:\\match\\1213\\xgb6.model')
reader = pd.read_csv('E:\\match\\1213\\testData201712.csv',iterator = True)
# plot_importance(model)
# plt.show()
# reader = pd.read_csv('E:\\match\\1213\\allTrainData_1213.csv',iterator = True)
chunkSize = 100000
loop = True
count = 0;
size = 0;
# with open('E:/match//1213//abnormalPoint.csv', 'w',newline='') as dstfile:
# header = ['xid', 'yid', 'date', 'hour', 'data','model1','model2','model3','model4','model5','model6','model7','model8','model9','model10','predict']
# writer = csv.DictWriter(dstfile, fieldnames=header)
# writer.writeheader()    #   写入表头
while loop:
    try:
        chunk = reader.get_chunk(chunkSize)
        chunk['num'] = chunk.model1.apply(lambda x: 1 if x < 15 else 0)
        columns = ['model2','model3','model4','model5','model6','model7','model8','model9','model10']
        for var in columns:
            chunk['num'] =chunk['num'] + chunk[var].apply(lambda x: 1 if x < 15 else 0)
        data = chunk.as_matrix()
        # y=chunk[['data']].as_matrix()
        Y = model.predict(chunk[['xid','yid','hour','model1','model2','model3','model4','model5','model6','model7','model8','model9','model10','num']])
        result = pd.DataFrame({'xid':chunk['xid'].as_matrix(),'yid':chunk['yid'].as_matrix(),'date':chunk['date'].as_matrix(),
                               'hour':chunk['hour'].as_matrix(), 'wind':Y.astype(np.float32)})
        result.to_csv("E:\match\\1213\\testPoints1.csv", index=False,mode = 'a')
        # size = size + len((Y))
        # # lines = []
        # for i in range(0,len(Y)):
        #     if Y[i] >= 15 and y[i] < 15:
        #         count = count+1
        #         # line = data[i]
        #         # result = ','.join(map(str,line))+','+str(Y[i])+'\n'
        #         # lines.append(result)
        #         # line.to_csv("E:\match\\1213\\abnormalPoints.csv", index=False,mode = 'a')
        #     if Y[i] < 15 and y[i] >= 15:
        #         count = count+1
        #         # line = data[i]
        #         # result = ','.join(map(str,line))+','+str(Y[i])+'\n'
        #         # lines.append(result)
        #         # line.to_csv("E:\match\\1213\\abnormalPoints.csv", index=False,mode = 'a')
        # # dstfile.writelines(lines)
        # print(count)
    except StopIteration:
        loop = False
        print("count:")
        print(count)
        print("size:")
        print(size)
        print(count/size)
        print ("Iteration is stopped.")



#
# train = pd.read_csv('E:\\match\\1213\\trainData_1213.csv' ,nrows = 1000000)
# train['num'] = train.model1.apply(lambda x: 1 if x < 15 else 0)
# columns = ['model2','model3','model4','model5','model6','model7','model8','model9','model10']
# for var in columns:
#     train['num'] =train['num'] + train[var].apply(lambda x: 1 if x < 15 else 0)
# x=train[['xid','yid','hour','model1','model2','model3','model4','model5','model6','model7','model8','model9','model10','num']].as_matrix()
# y=train[['data']].as_matrix()
# X_dtrain, X_deval, y_dtrain, y_deval = cross_validation.train_test_split(x, y, random_state=1026, test_size=0.3)
# dtrain = xgb.DMatrix(X_dtrain, y_dtrain)
# deval = xgb.DMatrix(X_deval, y_deval)
# watchlist = [(deval, 'eval')]
# params = {
#     'booster': 'gbtree',
#     'objective': 'reg:linear',
#     'subsample': 0.8,
#     'colsample_bytree': 0.85,
#     'eta': 0.05,
#     'max_depth': 5,
#     'seed': 2016,
#     'silent': 0,
#     'eval_metric': 'rmse'
# }
# clf = xgb.train(params, dtrain, 100, watchlist, early_stopping_rounds=5)
# pred = clf.predict(xgb.DMatrix(X_deval))