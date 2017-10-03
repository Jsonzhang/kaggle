import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def computeCost(result, h, m):
  return (np.dot(-result.transpose(), np.log(h)) - np.dot((1 - result).transpose(), np.log(1 - h))) / m

X = np.matrix([[0.1, 0.1],[-0.1, 0.1],[-0.1, -0.1],[0.1, -0.1]])
result = np.matrix([[1.0],[1.0],[0.0],[0.0]])
cost = 0.0
m = len(X)
theta = np.zeros(shape=(2,1))


print(result)
print(sigmoid(np.dot(X, theta)))
print(m)
J = computeCost(result, sigmoid(np.dot(X, theta)), m)
print(J)
alpha = 0.8
# print(sigmoid(np.dot(X, theta)))
for i in range(0, 100000):
  theta = theta - (alpha / m) * np.dot(X.transpose() ,  np.subtract(sigmoid(np.dot(X, theta)),result) )
  J = computeCost(result, sigmoid(np.dot(X, theta)), m)
  print(J)
