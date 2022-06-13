from cProfile import label
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from matplotlib import pyplot as plt

X, Y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)


# fig = plt.figure(figsize=(8,6))
# plt.scatter(X[:, 0], Y, color='b', marker='o', s=30)
# plt.show()

# print(X_train.shape)
# print(Y_train.shape)

from linear_regression import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)
predicted_values = regressor.predict(X_test)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

print(mse(Y_test, predicted_values))

regressor_2 = LinearRegression(n_iters=10000)
regressor_2.fit(X_train, Y_train)
predicted_vals = regressor_2.predict(X_test)

pred_line1 = regressor.predict(X)
pred_line2 = regressor_2.predict(X)

print(mse(Y_test, predicted_vals))

cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, Y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, Y_test, color=cmap(0.5), s=10)
plt.plot(X, pred_line1, color='black', linewidth=2, label='n_iters=1000')
plt.plot(X, pred_line2, color='red', linewidth=2, label='n_iters=10000')
plt.show()

