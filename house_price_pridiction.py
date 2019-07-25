import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import OneHotEncoder 
import operator

#数据预处理函数：消除空值
def NaN_preprocessing(x):
    column = list(x)
    for i in column:
        #标签特征将空值替换为最频繁出现的标签
        if type(x[i][0]) == type('string'):
            x[i][x[i].isnull()] = x[i].dropna().mode().values
        #数值特征将空值替换为特征平均值
        else:
            x[i] = x[i].fillna(x[i].mean())

#将特征数据分成数值特征和标签特征的函数
def divided_data(x,x_continues,x_discrete):
    column = list(x)
    for i in column:
        if type(x[i][0]) == type('string'):
            x_discrete[i] = x[i]
        else:
            x_continues[i] = x[i]

#处理训练数据函数
def get_train_data(x_train_dataset, y_train_dataset, column_drop_list):
    #消除训练集空值
    NaN_preprocessing(x_train_dataset)

    x_train_continues = pd.DataFrame()
    x_train_discrete = pd.DataFrame()
    #分割训练集特征
    divided_data(x_train_dataset, x_train_continues, x_train_discrete)
    #数值特征进行单变量特征选取
    feature_dict = {}
    transformer = SelectKBest(score_func=f_regression, k=30)
    x_train_continues_choose = transformer.fit_transform(x_train_continues.values, y_train_dataset.values)
    #对单变量特征选取结果进行排序，获取被删除的特征名
    score = transformer.scores_
    column = list(x_train_continues)
    for i in range(len(score)):
        feature_dict[column[i]] = score[i]
    column_sort = sorted(feature_dict.items(), key=operator.itemgetter(1))
    column_drop = len(column) - 30
    for i in range(column_drop):
        column_drop_list.append(column_sort[i][0])

    x_train_discrete = x_train_discrete.drop(['Utilities','Condition2','HouseStyle','RoofMatl','Exterior1st','Exterior2nd','Heating','Electrical','GarageQual'], axis=1)
    #标签特征进行特征编码
    enc = OneHotEncoder(sparse=False)
    x_train_discrete_onehot = enc.fit_transform(x_train_discrete.values)

    x_train = np.append(x_train_continues_choose, x_train_discrete_onehot, axis = 1)

    return x_train

#处理测试数据函数
def get_test_data(x_test_dataset, column_drop_list):
    #消除测试集空值
    NaN_preprocessing(x_test_dataset)

    x_test_continues = pd.DataFrame()
    x_test_discrete = pd.DataFrame()
    #分割测试集特征
    divided_data(x_test_dataset, x_test_continues, x_test_discrete)
    #根据训练数据单变量特征选取结果对测试数据进行处理
    x_test_continues = x_test_continues.drop(column_drop_list, axis = 1)
    x_test_continues_choose = x_test_continues.values

    x_test_discrete = x_test_discrete.drop(['Utilities','Condition2','HouseStyle','RoofMatl','Exterior1st','Exterior2nd','Heating','Electrical','GarageQual'], axis=1)
    #标签特征进行特征编码
    enc = OneHotEncoder(sparse=False)
    x_test_discrete_onehot = enc.fit_transform(x_test_discrete.values)

    x_test = np.append(x_test_continues_choose, x_test_discrete_onehot, axis = 1)

    return x_test

#训练模型XGBoost函数
def xgboost_train(x_train, y_train, x_test, predic_result, Test_Id):
    #XGBoost训练过程
    model = xgb.XGBRegressor(max_depth=6, learning_rate=0.05, n_estimators=500, silent=False, objective='reg:squarederror', n_jobs=3)
    model.fit(x_train, y_train)
    #对测试集进行预测
    preds = model.predict(x_test)

    predic_result['Id'] = Test_Id
    predic_result['SalePrice'] = preds

if __name__ == '__main__':
    #读取训练数据集和测试数据集
    dataset = pd.read_csv(r"C:\Users\king\Desktop\Coursera\Mashine Learining\Kaggle\house-prices-advanced-regression-techniques\train.csv")
    x_test_dataset = pd.read_csv(r"C:\Users\king\Desktop\Coursera\Mashine Learining\Kaggle\house-prices-advanced-regression-techniques\test.csv")
    
    #将训练数据集分成自变量和因变量
    Test_Id = x_test_dataset['Id']
    x_train_dataset = dataset.drop(['Id','SalePrice'], axis=1)
    x_test_dataset = x_test_dataset.drop('Id', axis=1)
    y_train_dataset = dataset[['SalePrice']]

    #将空值大于25%的特征删除
    x_train_dataset = x_train_dataset.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'], axis =1)
    x_test_dataset = x_test_dataset.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'], axis =1)

    column_drop_list = []
    #处理训练数据集
    x_train = get_train_data(x_train_dataset, y_train_dataset, column_drop_list)
    y_train = y_train_dataset.values
    #处理测试数据集
    x_test = get_test_data(x_test_dataset, column_drop_list)
    #预测最终房价
    predic_result = pd.DataFrame()
    xgboost_train(x_train, y_train, x_test, predic_result, Test_Id)
    #输出结果
    predic_result.to_csv(r"C:\Users\king\Desktop\Coursera\Mashine Learining\Kaggle\house-prices-advanced-regression-techniques\predict.csv", index=False)