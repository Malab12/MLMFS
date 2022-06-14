import numpy as np

#TODO: shhift this method into a utility class
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

class LogistRegression:

    def __init__(self, lr=0.001, n_iters=1000) -> None:
        self.lr= lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, x_train, y_train):
        #init the parameters
        n_samples, n_features = x_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        #gradient descent
        for _ in range(self.n_iters):
            linear_model = np.dot(x_train, self.weights) + self.bias
            y_predicted = sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(x_train.T, (y_predicted - y_train))
            db = (1 / n_samples) * np.sum((y_predicted - y_train))

            self.weights -= self.lr * dw
            self.bias -= self.lr*db

    def predict(self, x_test):
        linear_model = np.dot(x_test, self.weights) + self.bias
        y_predicted = sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls
