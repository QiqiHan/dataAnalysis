import pandas as pd
import numpy as np
from sklearn import cross_validation, linear_model
from sklearn.model_selection import train_test_split



train = pd.read_csv('E:\\match\\1213\\trainData_1213.csv' )
train['num'] = train.model1.apply(lambda x: 1 if x < 15 else 0)
columns = ['model2','model3','model4','model5','model6','model7','model8','model9','model10']
for var in columns:
    train['num'] =train['num'] + train[var].apply(lambda x: 1 if x < 15 else 0)
# print(train['num'])
x=train[['xid','yid','hour','model1','model2','model3','model4','model5','model6','model7','model8','model9','model10','num']].as_matrix()
y=train[['data']].as_matrix()
y =np.array(y).reshape(1,len(y))[0]
x_,x_t,y_,y_t=train_test_split(x,y,test_size=0.3,random_state=1)


clf = linear_model.RidgeCV(
    alphas=np.linspace(0, 200), cv=100)
clf.fit(x_,y_)
print (clf.coef_, clf.intercept_)
y_pre = clf.predict(x_t)
count = 0
size = len(y_pre)
for i in range(0,len(y_pre)):
    if y_pre[i] >= 15 and y_t[i] < 15:
        count = count+1
    if y_pre[i] < 15 and y_t[i] >= 15:
        count = count+1
print(count)
print(size)
print(count / size)