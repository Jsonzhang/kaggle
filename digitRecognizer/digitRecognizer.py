import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('./data/train.csv', index_col=False)
test_data = pd.read_csv('./data/test.csv', index_col=False)

y = data['label']
X = data.loc[:, ~(data == 0).all()].drop('label', axis=1)
X_target = test_data.loc[:, ~(data == 0).all()]
newX = X.applymap(lambda x: 0 if x == 0 else 1)
X_target = X_target.applymap(lambda x: 0 if x == 0 else 1)

X_train, X_test, y_train, y_test = train_test_split(newX, y, test_size=0.05, random_state=True)

logic = LogisticRegression()
logic.fit(X_train, y_train)

result = logic.predict(X_target)

submission = pd.DataFrame({
    "Label": result
})
submission.to_csv('./submission.csv', index_label='ImageId')

print(logic.score(X_train, y_train))
print(logic.score(X_test, y_test))

"""
0.929097744361
0.906666666667
"""
