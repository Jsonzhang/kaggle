# coding=utf-8

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

train_df = pd.read_csv('./data/train.csv', index_col=False).head(600)
test_df = pd.read_csv('./data/train.csv', index_col=False).tail(291)
valid_df = pd.concat([train_df.tail(100), test_df.head(100)], axis=0)

target_df = pd.read_csv('./data/test.csv', index_col=False)


def processFeatures(data):
  # source = data.loc[:, ['Pclass', 'SibSp', 'Parch', 'Fare']]
  source = data

  dummies_embarked = pd.get_dummies(source['Embarked'], prefix= 'Embarked')
  dummies_embarked = pd.get_dummies(source['Sex'], prefix= 'Embarked')
  source = pd.concat([source, dummies_embarked], axis=1)

  source['Sex'] = source['Sex'].map(lambda x : 1 if x == 'male' else 0 )

  source = set_missing_ages(source)
  return source.filter(regex='Age|SibSp|Parch|Fare|Embarked_.*|Sex|Pclass')

def set_missing_ages(df):
  # 把已有的数值型特征取出来丢进Random Forest Regressor中
  age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
  # 乘客分成已知年龄和未知年龄两部分
  known_age = age_df[age_df.Age.notnull()].as_matrix()
  unknown_age = age_df[age_df.Age.isnull()].as_matrix()
  y = known_age[:, 0]
  X = known_age[:, 1:]
  rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
  rfr.fit(X, y)
  predictedAges = rfr.predict(unknown_age[:, 1::])
  df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
  return df



X = processFeatures(train_df)
# Name Ticket Cabin
y = train_df['Survived']
testX = processFeatures(test_df)
testY = test_df['Survived']
validX = processFeatures(valid_df)
validY = valid_df['Survived']

target_df['Fare'] = target_df.Fare.fillna(0)
targetX = processFeatures(target_df)

reg = LogisticRegression()
reg.fit(X, y)

print("trainingSet:", np.count_nonzero(reg.predict(X) == y) / float(len(y)))
print("validset:" , np.count_nonzero(reg.predict(validX) == validY) / float(len(validY)))
print("testSet", np.count_nonzero(reg.predict(testX) == testY) / float(len(testY)))

result = reg.predict(targetX)

submission = pd.DataFrame({
    "PassengerId": target_df["PassengerId"],
    "Survived": result
})
submission.to_csv('./submission.csv', index=False)

