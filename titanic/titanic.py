# coding=utf-8

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def processFeatures(data):
    source = data
    source.Cabin = source.Cabin.str.extract('([A-Z])\d+', expand=False)
    source.Cabin.fillna('NULL', inplace=True)

    source.Fare.fillna(source['Fare'].dropna().median(), inplace=True)
    dummies_embarked = pd.get_dummies(source['Embarked'], prefix='Embarked')
    dummies_cabin = pd.get_dummies(source['Cabin'], prefix='Cabin')
    dummies_Pclass = pd.get_dummies(source['Pclass'], prefix='Pclass')
    source = pd.concat([source, dummies_embarked, dummies_Pclass, dummies_cabin], axis=1)

    source['Title'] = source.Name.str.extract(' ([A-Za-z]+)\.', expand=False).map(survived_rate, na_action=None)
    source['Title'].fillna(0.5, inplace=True)
    
    source['Sex'] = source['Sex'].map(lambda x: 1 if x == 'male' else 0)
    source['isChild'] = source['Age'].map(lambda x: 1 if x <= 16 else 0)
    source['isOld'] = source['Age'].map(lambda x: 1 if x > 60 else 0)
    source['isAlone'] = 0
    source['FamilySize'] = source['SibSp'] + source['Parch'] + 1

    source['SibSp'] = source['SibSp'].map(sib_rate, na_action=None)
    source['SibSp'].fillna(0.5, inplace=True)

    source.loc[source['FamilySize'] == 1, 'isAlone'] = 1
    source = set_missing_ages(source, ['Age', 'Fare', 'Parch', 'SibSp', 'Pclass'], 'Age')
    source = source.filter(regex='isOld|isChild|isAlone|Title|Age|Fare|Embarked_.*|Cabin_.*|Sex|Pclass_.*')

    return preprocessing.MinMaxScaler().fit_transform(source)

def set_missing_ages(df, features, target):
    # 根据所坐舱位等数字讯息推断年龄
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

def set_missing_ages_2(df, feature):
    # 根据姓名求年龄中位数
    df['Age'].fillna(-1, inplace=True)
    titles = df['Name'].unique()
    medians = dict()
    for title in titles:
        median = df.Age[(df["Age"] != -1) & (df['Name'] == title)].median()
        medians[title] = median
        
    for index, row in df.iterrows():
        if row['Age'] == -1:
            df.loc[index, 'Age'] = medians[row['Name']]

    return df
# Name Ticket Cabin

def title_keymap_generate(target):
    titles = target.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    km = titles.unique()
    survived_rate = pd.Series(0.0, index=km)
    for title in km:
        survived_total = target.Survived[titles == title].value_counts()
        if 1 in survived_total:
            survived_rate[title] = float(survived_total[1]) / float(sum(survived_total))
        else:
            survived_rate[title] = 0
    return survived_rate

def sibsp_map_generate(source):
    km = source['SibSp'].unique()
    sib_rate = pd.Series(0.0, index=km)
    for sib in km:
        survived_total = source.Survived[source['SibSp'] == sib].value_counts()
        if 1 in survived_total:
            sib_rate[sib] = float(survived_total[1]) / float(sum(survived_total))
        else:
            sib_rate[sib] = 0
    return sib_rate



train_df = pd.read_csv('./data/train.csv', index_col=False).head(700)
# valid_df = pd.read_csv('./data/train.csv', index_col=False)[540:700]
test_df = pd.read_csv('./data/train.csv', index_col=False).tail(191)
# valid_df = pd.concat([train_df.tail(100), test_df.head(100)], axis=0)
target_df = pd.read_csv('./data/test.csv', index_col=False)

survived_rate = title_keymap_generate(train_df)
sib_rate = sibsp_map_generate(train_df)

X = processFeatures(train_df)
y = train_df['Survived']
testX = processFeatures(test_df)
testY = test_df['Survived']
# validX = processFeatures(valid_df)
# validY = valid_df['Survived']

targetX = processFeatures(target_df)

# reg = svm.SVC()
# reg = LogisticRegression()
# reg = KNeighborsClassifier(n_neighbors = 3)
# reg = GaussianNB()
# reg = Perceptron()
# reg = svm.LinearSVC()
# reg = SGDClassifier()
# reg = DecisionTreeClassifier()

# create param grid object
forrest_params = dict(
    max_depth = [n for n in range(9, 14)],
    min_samples_split = [n for n in range(4, 11)],
    min_samples_leaf = [n for n in range(2, 5)],
    n_estimators = [n for n in range(10, 60, 10)],
)

forrest = RandomForestClassifier()
forest_cv = GridSearchCV(estimator=forrest, param_grid=forrest_params, cv=5)
forest_cv.fit(X, y)

print("Best score: {}".format(forest_cv.best_score_))
print("Best params: {}".format(forest_cv.best_estimator_))
print("trainingSet:", round(forest_cv.score(X, y) * 100, 2))
# print("validset:",round(forest_cv.score(validX, validY) * 100, 2))
print("testSet:", round(forest_cv.score(testX, testY) * 100, 2))

result = forest_cv.predict(targetX)

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

knn:
trainingSet: 86.67
validset: 77.5
testSet: 78.01

gussian:
trainingSet: 78.5
validset: 77.5
testSet: 79.38

Perceptron:
trainingSet: 74.33
validset: 69.0
testSet: 74.91

LinearSVC:
trainingSet: 81.17
validset: 78.0
testSet: 79.73

SGDClassifier:
trainingSet: 79.33
validset: 76.5
testSet: 79.73

DecisionTreeClassifier:
trainingSet: 98.5
validset: 83.5
testSet: 73.2

RandomForestClassifier:
trainingSet: 98.5
validset: 84.5
testSet: 80.76

RandomForestClassifier with params:
trainingSet: 85.19
validset: 77.5
testSet: 84.82

kaggal: 3641 - 0.78468

trainingSet: 89.86
testSet: 86.39
kaggal: 0.7799


----------------------------------
refer: https://medium.com/towards-data-science/how-i-got-98-prediction-accuracy-with-kaggles-titanic-competition-ad24afed01fc





"""
