# Machine Learning from Scratch
My efforts to implement some basic machine learning algorithms from scratch in Python 
In this repository are the implementation of 15 basic algortihms and a neural network
1. [K Nearest Neighbours](#k-nearest-neighbours)
2. [Linear Regression](#linear-regression)
3. Logistic Regression
4. Naive Bayes
5. Perceptron
6. SVM
7. Decision Tree
8. Random Forest
9. PCA
10. K-Means Clustering
11. AdaBoost
12. LDA
13. Simple Direct Neural Network

## K Nearest Neighbours
The k-nearest neighbors algorithm, also known as KNN or k-NN, is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point.
### External Dependencies
1. Numpy
2. Matplotlib
3. SKlearn (for the iris dataset)

## Linear Regression
Linear regression is a linear model, e.g. a model that assumes a linear relationship between the input variables (x) and the single output variable (y). More specifically, that y can be calculated from a linear combination of the input variables (x). When there is a single input variable (x), the method is referred to as simple linear regression.

`y = a_0 + a_1 * x      ## Linear Equation`

### Cost Function
The cost function helps us to figure out the best possible values for a_0 and a_1 which would provide the best fit line for the data points. Since we want the best values for a_0 and a_1, we convert this search problem into a minimization problem where we would like to minimize the error between the predicted value and the actual value.

![image](https://user-images.githubusercontent.com/54464437/173301157-5a39463f-93fc-470a-b2f8-f6090b1f0f7a.png)

This cost function is also known as the Mean Squared Error(MSE) function. Now, using this MSE function we are going to change the values of a_0 and a_1 such that the MSE value settles at the minima.

### Gradient Descent
 Gradient descent is a method of updating a_0 and a_1 to reduce the cost function(MSE). The idea is that we start with some values for a_0 and a_1 and then we change these values iteratively to reduce the cost. Gradient descent helps us on how to change the values. To update a_0 and a_1, we take gradients from the cost function. To find these gradients, we take partial derivatives with respect to a_0 and a_1.
 
 ![image](https://user-images.githubusercontent.com/54464437/173301503-c5ecc3ad-1981-4ce2-9d9c-a010eb7d6c37.png)

The partial derivates are the gradients and they are used to update the values of a_0 and a_1. Alpha is the learning rate which is a hyperparameter that you must specify. A smaller learning rate could get you closer to the minima but takes more time to reach the minima, a larger learning rate converges sooner but there is a chance that you could overshoot the minima.

### External Dependencies
1. numpy
2. Matplotlib
3. SKLearn (to generate our own data to fit linear regression on)

### Resultant Graph
![Figure_1](https://user-images.githubusercontent.com/54464437/173302024-5200b162-a6d2-4ce0-8a15-213a5291af44.png)
1. black: regression with 1000 iterations
2. red: regression with 10000 iterations

