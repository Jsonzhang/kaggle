# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import math as math
# visualization
import seaborn as sb
import matplotlib.pyplot as plt
import scipy.optimize as op
from scipy.optimize import fmin
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import re
# from sklearn import *
# import xgboost as xgb
from sklearn.svm import SVC

pd.set_option('display.max_rows', 1000)
np.set_printoptions(threshold=np.inf)

train_df = pd.read_csv('./train.csv', index_col=False).head(800)
test_df = pd.read_csv('./train.csv', index_col=False).tail(91)
# train_df_new = pd.read_csv('./train.csv', index_col=False )
gender_submission = pd.read_csv('./gender_submission.csv')
# combine = [train_df, test_df]
# print(train_df.describe())


train_df["title"] = [i[i.index(', ') + 2:i.index('.')] for i in train_df["Name"]]
np.unique(train_df["title"])
rare_title = ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']
train_df["title"] = ["Rare" if i in rare_title else i for i in train_df["title"]]

# fig, ax = plt.subplots()
# fig.set_size_inches(11.7, 10.27)
# plt.title('Title')
# sb.set(style="ticks")
# sb.countplot(x="title", hue = "Survived", data = train_df, palette="Set3", edgecolor=sb.color_palette("husl", 8))

# train_df["ticket_number"] = [i if re.search('[a-zA-Z]', i) == None else None for i in train_df["Ticket"]]
# fig, ax = plt.subplots()
# fig.set_size_inches(11.7, 10.27)
# # sb.set(style="ticks")
# sb.stripplot(x="Survived", y="ticket_number", data=train_df, palette="Set3")
# print(train_df['ticket_number'])

train_df["family_members"] =  train_df["Parch"] + train_df["SibSp"]
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 10.27)
plt.title('Number of Family Members')
sb.set(style="white")
sb.countplot(x="family_members", hue = "Survived", data = train_df, palette="Set3", edgecolor=sb.color_palette("husl", 8))

def prob_surv(dataset, group_by):
  df = pd.crosstab(index = dataset[group_by], columns = dataset.Survived).reset_index()
  df['prob_surv'] = df[1] / (df[1] + df[0])
  return df[[group_by, 'prob_surv']]

train_df['age_cat'] = pd.cut(train_df['Age'], 30, labels = np.arange(1,31))
train_df['fare_cat'] = pd.cut(train_df['Fare'], 50, labels = np.arange(1,51))
sb.lmplot(data = prob_surv(train_df, 'age_cat'), x = 'age_cat', y = 'prob_surv', fit_reg = True, palette="Set3")
plt.title('Probability of being Survived with respect to Age')
plt.show()
sb.lmplot(data = prob_surv(train_df, 'fare_cat'), x = 'fare_cat', y = 'prob_surv', fit_reg = True, palette="Set3")
plt.title('Probability of being Survived with respect to Fare')
plt.show()

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 10.27)

plt.figure(1)

plt.subplot(221)
plt.title('Ticket Class')
sb.set(style="darkgrid")
sb.countplot(x="Pclass", hue = "Survived", data = train_df,palette="Set3", edgecolor=sb.color_palette("husl", 8))

plt.subplot(222)
plt.title('Gender')
sb.set(style="darkgrid")
sb.countplot(x="Sex", hue = "Survived", data = train_df, palette="Set3", edgecolor=sb.color_palette("husl", 8))

plt.subplot(223)
plt.title('Port of Embarkation')
sb.set(style="darkgrid")
sb.countplot(x="Embarked", hue = "Survived", data = train_df,palette="Set3", edgecolor=sb.color_palette("husl", 8))

plt.subplot(224)
plt.title('Number of Siblings and Spouses')
sb.set(style="darkgrid")
sb.countplot(x="SibSp", hue = "Survived", data = train_df,palette="Set3", edgecolor=sb.color_palette("husl", 8) )

# We can see that higher fare has better odds of being survived.
plt.figure(1)
g = sb.PairGrid(
  train_df,
  y_vars=["Age", "Fare"],
  x_vars=["Survived"],
  aspect=2, 
  size=4
)
g.map(sb.violinplot, palette="Set3")

# plt.show()



























def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def computeCost(x, y, theta, m):
  h = sigmoid(np.dot(X, theta))
  lbd = 5
  return np.sum(np.dot(-y.transpose(), np.log(h)) - np.dot( (1 - y).transpose() , np.log(1 - h) )) / m + lbd * (np.sum(theta[2:] ** 2)) / (2 * m)

def preprocessingData(source, features):
  source = source.loc[:, features]
  source['Sex'] = source['Sex'].map(lambda x : 1 if x == 'male' else 0 )
  km = {
    'S': 1,
    'C': 2,
    'Q': 3
  }
  source['Embarked'] = source['Embarked'].map(lambda x : km[x] if (x in km) else 0 )
  source['Age'] = source['Age'].fillna(0)
  source['Fare'] = source['Fare'].fillna(0)
  return source
# Ticket Cabin drop off

features = ['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']

y = train_df['Survived'].values
X = preprocessingData(train_df, features)
X = preprocessing.MinMaxScaler().fit_transform(X.values.reshape(len(X), len(features)))

testY = test_df['Survived'].values
testX = preprocessingData(test_df, features)
testX = preprocessing.MinMaxScaler().fit_transform(testX.values.reshape(len(testX), len(features)))

reg = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=12, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)

reg.fit(X, y)

print(np.count_nonzero(reg.predict(X) == y) / float(len(y)))
print(np.count_nonzero(reg.predict(testX) == testY) / float(len(testY)))
# print() 
# for i in range(0, len(testX)):
  # print(len(testX[i,:].transpose()), len(reg.intercept_))
  # p[i] = 0 if sigmoid(np.dot(testX[i,:], reg.intercept_[0])) < 0.5 else 1
# print(np.count_nonzero(p == test_y) / float(len(p)))
# plt.plot(J_list)
# plt.show()
# m = len(X.values)
# [s, g] = size(theta)
# lam = 0.5

# initTheta[2:size(initTheta)] ^ 2

# J = np.matrix.sum(-y.transpose() * math.log(h) - (1 - y).transpose() * math.log(1 - h)) / m + 0.1 * (np.matrix.sum(initTheta[1:,])) / (2 * m)

# print(features)
