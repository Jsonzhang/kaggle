# coding=utf-8

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

train_df = pd.read_csv('./data/train.csv', index_col=False).head(600)
test_df = pd.read_csv('./data/train.csv', index_col=False).tail(291)

def getTargetFeatures(data):
  source = data.loc[:, ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
  source['Age'] = source['Age'].fillna(0)
  source['Fare'] = source['Fare'].fillna(0)
  return source

X = getTargetFeatures(train_df)
y = train_df['Survived']
testX = getTargetFeatures(test_df)
testY = test_df['Survived']


reg = LogisticRegression()
reg.fit(X, y)

print(np.count_nonzero(reg.predict(X) == y) / float(len(y)))
print(np.count_nonzero(reg.predict(testX) == testY) / float(len(testY)))