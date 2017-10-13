# coding=utf-8

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn import preprocessing
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, log_loss

def processFeatures(data):
    source = data
    source.Cabin = source.Cabin.str.extract('([A-Z])\d+', expand=False)
    source.Cabin.fillna('NULL', inplace=True)

    source.Fare.fillna(source['Fare'].dropna().median(), inplace=True)
    dummiesEmbarked = pd.get_dummies(source['Embarked'], prefix='Embarked')
    dummiesCabin = pd.get_dummies(source['Cabin'], prefix='Cabin')
    dummiesPclass = pd.get_dummies(source['Pclass'], prefix='Pclass')

    source['Title'] = source.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    source['Title'] = source['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    source['Title'] = source['Title'].replace('Mlle', 'Miss')
    source['Title'] = source['Title'].replace('Ms', 'Miss')
    source['Title'] = source['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    source['Title'] = source['Title'].map(title_mapping)

    dummiesTitle = pd.get_dummies(source['Title'], prefix='Title')

    source.loc[source['Fare'] <= 7.91, 'Fare'] = 0
    source.loc[(source['Fare'] > 7.91), 'Fare'] = 1
    source.loc[(source['Fare'] > 14.454) & (source['Fare'] <= 31), 'Fare'] = 2
    source.loc[source['Fare'] > 31, 'Fare']= 3

    dummiesFare = pd.get_dummies(source['Fare'], prefix='Fare')

    source = pd.concat([source, dummiesEmbarked, dummiesPclass, dummiesCabin, dummiesTitle, dummiesFare], axis=1)

    source['Sex'] = source['Sex'].map(lambda x: 1 if x == 'male' else 0)
    source['isChild'] = source['Age'].map(lambda x: 1 if x <= 16 else 0)
    source['isOld'] = source['Age'].map(lambda x: 1 if x > 60 else 0)
    source['isAlone'] = 0
    source['FamilySize'] = source['SibSp'] + source['Parch'] + 1

    source['SibSp'] = source['SibSp'].map(sib_rate, na_action=None)
    source['SibSp'].fillna(0.5, inplace=True)

    source.loc[source['FamilySize'] == 1, 'isAlone'] = 1
    source = set_missing_ages(source, ['Age', 'Fare', 'Parch', 'SibSp', 'Pclass'], 'Age')
    source = source.filter(regex='SibSp|isOld|isChild|isAlone|Title_.*|Age|Fare_.*|Embarked_.*|Cabin_.*|Sex|Pclass_.*')

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

train_df = pd.read_csv('./data/train.csv', index_col=False)
# valid_df = pd.read_csv('./data/train.csv', index_col=False)[540:700]
# test_df = pd.read_csv('./data/train.csv', index_col=False).tail(191)
# valid_df = pd.concat([train_df.tail(100), test_df.head(100)], axis=0)
target_df = pd.read_csv('./data/test.csv', index_col=False)

sib_rate = sibsp_map_generate(train_df)

X = processFeatures(train_df)
y = train_df['Survived']
# testX = processFeatures(test_df)
# testY = test_df['Survived']
# validX = processFeatures(valid_df)
# validY = valid_df['Survived']

targetX = processFeatures(target_df)

# create param grid object
forrest_params = dict(
    max_depth = [n for n in range(9, 14)],
    min_samples_split = [n for n in range(4, 11)],
    min_samples_leaf = [n for n in range(2, 5)],
    n_estimators = [n for n in range(10, 60, 10)]
)

classifiers = [
    KNeighborsClassifier(3),
    svm.SVC(probability=True),
    svm.LinearSVC(),
    DecisionTreeClassifier(),
    GridSearchCV(estimator=RandomForestClassifier(), param_grid=forrest_params, cv=5),
    AdaBoostClassifier(),
    SGDClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    Perceptron(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(),
    DecisionTreeClassifier()
]

maxValue = 0
targetClf = None
acc_dict = {}

log_cols = ["Classifier", "Accuracy"]
log = pd.DataFrame(columns=log_cols)
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    for clf in classifiers:
        name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        print(acc)
        if acc > maxValue:
            targetClf = clf
            maxValue = acc
        if name in acc_dict:
            acc_dict[name] += acc
        else:
            acc_dict[name] = acc

for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf] / 10.0
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
    log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

result = targetClf.predict(targetX)
submission = pd.DataFrame({
    "PassengerId": target_df["PassengerId"],
    "Survived": result
})
submission.to_csv('./submission.csv', index=False)

"""
kaggal: 0.78468
kaggal: 3587
"""
