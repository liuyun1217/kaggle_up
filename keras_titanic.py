#import kaggle
import os
import csv
import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import keras
from keras.utils import multi_gpu_model

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"



data_base_path = './competitions/titanic/'
test_file = data_base_path+'test.csv'
train_file = data_base_path+'train.csv'
gender_file = data_base_path+'gender_submission.csv'

data_train = pd.read_csv(train_file)

from keras.models import Sequential
model = Sequential()

from keras.layers import Dense
from keras import regularizers

model.add(Dense(units=100,activation='relu',input_dim=12))
model.add(Dense(units=100,activation='tanh',input_dim=12))
model.add(Dense(units=100,activation='softplus',input_dim=12))
model.add(Dense(units=1,activation='sigmoid',kernel_regularizer=regularizers.l2(0.01)))
#model.add(Dense(units=1,activation='selu',kernel_regularizer=regularizers.l2(0.01)))
#model.compile(loss='binary_crossentropy',
model = multi_gpu_model(model,2)
model_para = model
model_para.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])


### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges

    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = 1
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = 0
    return df

data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)
#print(data_train['Cabin'])



dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
#df = pd.concat([data_train,dummies_Sex,dummies_Pclass,dummies_Cabin],axis=1)
df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
#df.drop(['Sex','Pclass'],axis=1,inplace=True)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

df['Age_scaled']  = preprocessing.scale(df['Age'])
df['Fare_scaled']  = preprocessing.scale(df['Fare'])
#print(df.describe())
train_df = df.filter(regex='Survived|Age_.*|Sex_.*|Pclass_.*|Fare_.*|Cabin_.*|Embarked_.*')
train_np = train_df.as_matrix()
# y即Survival结果
y = train_np[:, 0]
# X即特征属性值
X = train_np[:, 1:]


model_para.fit(X, y, epochs=5000, batch_size=50)

data_test = pd.read_csv(test_file)
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')


df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
#df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'], age_scale_param)
#df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'], fare_scale_param)

df_test['Age_scaled']  = preprocessing.scale(df_test['Age'])
df_test['Fare_scaled']  = preprocessing.scale(df_test['Fare'])
#print(data_test)

test = df_test.filter(regex='Age_.*|Sex_.*|Pclass_.*|Fare_.*|Cabin_.*|Embarked_.*')
#print(test)

#classes = model.predict(test, batch_size=128)
classes = model_para.predict(test, batch_size=128)
cls = np.transpose(classes)[0]>0.5
#print(np.transpose(classes)[0])
#print(cls)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':cls.astype(np.int32)})
result.to_csv("./keras_predictions.csv", index=False)
