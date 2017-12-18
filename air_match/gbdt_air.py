import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import cross_validation, metrics
from sklearn.model_selection import GridSearchCV
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib


train = pd.read_csv('E:\\match\\1213\\trainData_1213.csv',nrows=200000)
# train = pd.read_csv('E:\\match\\1213\\trainData_1213.csv')
# # train = train.dropna()
# # train = train[0:100000]
x=train[['xid','yid','hour','model1','model2','model3','model4','model5','model6','model7','model8','model9','model10']]
y=train[['data']]
x_,x_t,y_,y_t=train_test_split(x,y,test_size=0.3,random_state=0)
gbdt = GradientBoostingRegressor(learning_rate=0.1, n_estimators = 100 ,max_features='sqrt',max_depth=5,random_state=0,subsample=0.7)
# gbdt = joblib.load('E:\\match\\1213\\gbdt2.model')
gbdt.fit(x_,y_)
y_pre = gbdt.predict(x_t);
print("R-squared value:")
print(gbdt.score(x_t,y_t))
print("mean_squared_error:")
print(mean_squared_error(y_t,y_pre))
print('mean_absolute_error:')
print(mean_absolute_error(y_t,y_pre))
print(".")
y_tt = np.array(y_t).reshape(1,len(y_pre))[0]
count = 0
for i in range(0,len(y_pre)):
    if y_pre[i] >= 15 and y_tt[i] < 15:
        count = count+1
    if y_pre[i] < 15 and y_tt[i] >= 15:
        count = count+1
print("count:")
print(count)
print(count/len(y_pre))
# joblib.dump(gbdt,'E:\\match\\1213\\gbdt2.model')
# y_p = gbdt.predict(x_t)
# x_t['y_t'] = y_t['data'].as_matrix()
# x_t['y_p'] = y_p
# x_t.to_csv("E:\match\\1213\\result.csv", index=False)


# reader = pd.read_csv('E:\\match\\newTest.csv',iterator = True)
# d = reader.get_chunk(2);
# print(d[['xid','yid']])

# chunkSize = 100000
# loop = True
# while loop:
#     try:
#         chunk = reader.get_chunk(chunkSize)
#         Y = gbdt.predict(chunk[['model1','model2','model3','model4','model5','model6','model7','model8','model9','model10']])
#         result = pd.DataFrame({'xid':chunk['xid'].as_matrix(),'yid':chunk['yid'].as_matrix(),'date':chunk['date'].as_matrix(),
#                                'hour':chunk['hour'].as_matrix(), 'wind':Y.astype(np.float32)})
#         result.to_csv("E:\match\\testPoints.csv", index=False,mode = 'a')
#     except StopIteration:
#         loop = False
#         print ("Iteration is stopped.")


# param_test1 = {"n_estimators" : [100,200,300,400,500],
#                "max_depth" : [3,4,5,6]
#                }
# # 0.1 105  max_depth 3
# gsearch1 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators = 105 ,max_features='sqrt',max_depth=5,random_state=0,subsample=0.7),
#                         param_grid = param_test1,iid=False,cv=4,verbose=2 )
# gsearch1.fit(x,y)
# for score in gsearch1.grid_scores_:
#     print(score)
# print(gsearch1.best_params_)
# print(gsearch1.best_score_)
