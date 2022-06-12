import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

iris = datasets.load_iris()
x, y = iris.data, iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# print(x_train.shape)
# print(x_train[0])

# print(y_train.shape)
# print(y_train)

# plt.figure()
# plt.scatter(x[:, 2], x[:, 3], c=y, cmap=cmap, edgecolors='k', s=20)
# plt.show()
from knn import KNN
classifier = KNN(k=5)
classifier.fit(x_train, y_train)
predictions = classifier.predict(x_test)


accuracy = np.sum(predictions == y_test) / len(y_test)
print(accuracy)