# coding=utf-8

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

train_df = pd.read_csv('./data/train.csv', index_col=False).head(600)
test_df = pd.read_csv('./data/train.csv', index_col=False).tail(291)
valid_df = pd.concat([train_df.tail(100), test_df.head(100)], axis=0)
target_df = pd.read_csv('./data/test.csv', index_col=False)


def processFeatures(data):
    # source = data.loc[:, ['Pclass', 'SibSp', 'Parch', 'Fare']]
    source = data

    dummies_embarked = pd.get_dummies(source['Embarked'], prefix='Embarked')
    dummies_sex = pd.get_dummies(source['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(source['Pclass'], prefix='Pclass')
    source = pd.concat([source, dummies_embarked, dummies_sex, dummies_Pclass], axis=1)

    source['Title'] = source.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    source['Title'] = source['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    source['Title'] = source['Title'].replace('Mlle', 'Miss')
    source['Title'] = source['Title'].replace('Ms', 'Miss')
    source['Title'] = source['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    source['Title'] = source['Title'].map(title_mapping)
    source['Title'] = source['Title'].fillna(0)

    source['Sex'] = source['Sex'].map(lambda x: 1 if x == 'male' else 0)

    source['isAlone'] = 0
    source['FamilySize'] = source['SibSp'] + source['Parch'] + 1
    source.loc[source['FamilySize'] == 1, 'IsAlone'] = 1

    source = set_missing_ages(source, ['Age', 'Fare', 'Parch', 'SibSp', 'Pclass'], 'Age')

    source = source.filter(regex='isAlone|Title|Age|SibSp|Parch|Fare|Embarked_.*|Sex|Pclass_.*')

    return preprocessing.MinMaxScaler().fit_transform(source)


def set_missing_ages(df, features, target):
    target_df = df[features]
    known = target_df[target_df[target].notnull()].as_matrix()
    unknown = target_df[target_df[target].isnull()].as_matrix()
    y = known[:, 0]
    X = known[:, 1:]
    if len(unknown):
        rfr = RandomForestRegressor(
            random_state=0, n_estimators=2000, n_jobs=-1)
        rfr.fit(X, y)
        predicted = rfr.predict(unknown[:, 1::])
        df.loc[(df[target].isnull()), target] = predicted
    return df


# Name Ticket Cabin
X = processFeatures(train_df)
y = train_df['Survived']
testX = processFeatures(test_df)
testY = test_df['Survived']
validX = processFeatures(valid_df)
validY = valid_df['Survived']

target_df.Fare = target_df.Fare.fillna(target_df['Fare'].dropna().median())
targetX = processFeatures(target_df)

# reg = svm.SVC()
# reg = LogisticRegression()
reg = KNeighborsClassifier(n_neighbors = 3)

reg.fit(X, y)


print("trainingSet:", round(reg.score(X, y) * 100, 2))
print("validset:",round(reg.score(validX, validY) * 100, 2))
print("testSet:", round(reg.score(testX, testY) * 100, 2))


result = reg.predict(targetX)

submission = pd.DataFrame({
    "PassengerId": target_df["PassengerId"],
    "Survived": result
})

submission.to_csv('./submission.csv', index=False)

"""
first submit:

trainingSet: 0.8016666666666666
validset: 0.77
testSet: 0.7938144329896907
final: 0.76555

second submit:

trainingSet: 0.793333333
validset: 0.775
testSet: 0.807560137
final: 0.76076

svm :
trainingSet: 79.17
validset: 75.0
testSet: 77.66

l2:
trainingSet: 80.67
validset: 78.0
testSet: 79.38

l1:
trainingSet: 81.5
validset: 78.0
testSet: 79.04

"""
