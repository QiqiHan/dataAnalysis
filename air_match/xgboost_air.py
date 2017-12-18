import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import learning_curve

# train = pd.read_csv('E:\\match\\1213\\allTrainData_13_17.csv')
train = pd.read_csv('E:\\match\\1213\\trainData_1213.csv')
x=train[['xid','yid','hour','model1','model2','model3','model4','model5','model6','model7','model8','model9','model10']]
y=train[['data']]
xgb_model = xgb.XGBRegressor(max_depth=6,
                             learning_rate=0.1,
                             n_estimators=400,
                             silent=True,
                             objective='reg:linear',
                             nthread=3,
                             subsample=0.8);
x_,x_t,y_,y_t=train_test_split(x,y,test_size=0.3,random_state=0)
xgb_model.fit(x_,y_)

y_pre = xgb_model.predict(x_t);
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
joblib.dump(xgb_model,'E:\\match\\1213\\xgb4.model')


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




# model = joblib.load('E:\\match\\1213\\xgb4.model')
# # reader = pd.read_csv('E:\\match\\1213\\new_testData201712.csv',iterator = True)
# reader = pd.read_csv('E:\\match\\1213\\allTrainData_1213.csv',iterator = True)
# chunkSize = 100000
# loop = True
# count = 0;
# size = 0;
# while loop:
#     try:
#         chunk = reader.get_chunk(chunkSize)
#         y=chunk[['data']].as_matrix()
#         Y = model.predict(chunk[['xid','yid','hour','model1','model2','model3','model4','model5','model6','model7','model8','model9','model10']])
#         # result = pd.DataFrame({'xid':chunk['xid'].as_matrix(),'yid':chunk['yid'].as_matrix(),'date':chunk['date'].as_matrix(),
#         #                        'hour':chunk['hour'].as_matrix(), 'wind':Y.astype(np.float32)})
#         # result.to_csv("E:\match\\1213\\testPoints.csv", index=False,mode = 'a')
#         size = size + len((Y))
#         for i in range(0,len(Y)):
#             if Y[i] >= 15 and y[i] < 15:
#                 count = count+1
#             if Y[i] < 15 and y[i] >= 15:
#                 count = count+1
#     except StopIteration:
#         loop = False
#         print("count:")
#         print(count)
#         print("size:")
#         print(size)
#         print(count/size)
#         print ("Iteration is stopped.")