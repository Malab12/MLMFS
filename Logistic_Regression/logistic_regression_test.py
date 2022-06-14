import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from matplotlib import pyplot as plt

from logistic_regression import LogistRegression

data = datasets.load_breast_cancer()
X, Y = data.data, data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

regressor = LogistRegression(lr=0.001, n_iters=10000)
regressor.fit(X_train, Y_train)
predictions = regressor.predict(X_test)
pred_line = regressor.predict(X)

print("Accuracy: ", accuracy(Y_test, predictions))
