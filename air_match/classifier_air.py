import pandas as pd #数据分析
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib


if __name__=='__main__':
    reader = pd.read_csv('E:\\match\\1213\\allTrainLabelData.csv',iterator = True)

    # train = pd.read_csv('E:\\match\\1213\\trainLabelData.csv',nrows = 1000000)
    model = joblib.load('E:\\match\\1213\\xgb_classifier.model')
    chunkSize = 100000
    loop = True
    count = 0
    size = 0
    while loop:
        try:
            train = reader.get_chunk(chunkSize)
            x=train[['xid','yid','hour','model1','model2','model3','model4','model5','model6','model7','model8','model9','model10']].as_matrix()
            y = train[['model0']].as_matrix()
            Y = model.predict(x)
            size = size + len((Y))
            for i in range(0,len(Y)):
                if Y[i] != y[i]:
                    count = count+1
        except StopIteration:
            loop = False
            print("count:")
            print(count)
            print("size:")
            print(size)
            print(count/size)
            print ("Iteration is stopped.")

    # x_,x_t,y_,y_t=train_test_split(x,y,test_size=0.3,random_state=0)
    # parameters = {'criterion':['entropy']}
    # rf=RandomForestClassifier(n_estimators=300, n_jobs=3, verbose=1 , max_features= 10)
    # gs_clf =  GridSearchCV(rf_clf, parameters, n_jobs=3, verbose=True ,cv =2)
    # gs_clf.fit(x,y)
    # rf.fit(x_, y_)
    # rf = joblib.load('E:\\match\\1213\\randomForest.model')
    # y_pre = rf.predict(x_t)
    # print(rf.score(x_t,y_t))
    # print(classification_report(y_pre,y_t))

    # joblib.dump(rf,'E:\\match\\1213\\randomForest.model')
    # print()
    # for params, mean_score, scores in gs_clf.grid_scores_:
    #     print("%0.3f (+/-%0.03f) for %r"  % (mean_score, scores.std() * 2, params))
    # print()

    # model = xgb.XGBClassifier(silent=1, max_depth=10,
    #                          n_estimators=100, learning_rate=0.05,nthread=3,subsample= 0.7,seed= 27, n_jobs= 3)
    # model = joblib.load('E:\\match\\1213\\xgb_classifier.model')
    # model.fit(x_,y_)
    # y_pre = model.predict(x)
    # print(model.score(x,y))
    # print(classification_report(y_pre,y))
    # joblib.dump(model,'E:\\match\\1213\\xgb_classifier.model')

    # parameters = {'n_estimators' : [100,200,300,400],
    #               'max_depth' : [6,10,14,18]
    #              }
    # model = xgb.XGBClassifier(silent=1, max_depth=10,
    #                           n_estimators=100, learning_rate=0.05,nthread=3,subsample= 0.7,seed= 27, n_jobs= 3)
    # gsearch1 = GridSearchCV(model, parameters, n_jobs=3,
    #                    cv=2,
    #                    scoring='roc_auc',
    #                    verbose=2)
    # gsearch1.fit(x,y)
    # for score in gsearch1.grid_scores_:
    #     print(score)
    # print(gsearch1.best_params_)
    # print(gsearch1.best_score_)