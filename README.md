# Machine Learning from Scratch
My efforts to implement some basic machine learning algorithms from scratch in Python 
In this repository are the implementation of 15 basic algortihms and a neural network
1. [K Nearest Neighbours](#k-nearest-neighbours)
2. [Linear Regression](#linear-regression)
3. [Logistic Regression](#logistic-regression)
4. [Naive Bayes](#naive-bayes-classification)
5. [Perceptron](#perceptron)
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

## Logistic Regression
This type of statistical model (also known as logit model) is often used for classification and predictive analytics. Logistic regression estimates the probability of an event occurring, such as voted or didn’t vote, based on a given dataset of independent variables. Since the outcome is a probability, the dependent variable is bounded between 0 and 1. 

$sigma(z) = \frac{1} {1 + e^{-z}}$

$z = wx + b$

![image](https://user-images.githubusercontent.com/54464437/173510500-41ec6acb-90c1-4a7b-a85a-5c7b4fd709f5.png)

_NOTE:_ The gradient descent algorithm for logistic regression is the exact same as that of the linear regression

For analysis of the performance of the model we use accuracy and other classification related statistics

## Naive Bayes Classification
A Naive Bayes classifier is a probabilistic machine learning model that’s used for classification task. The crux of the classifier is based on the Bayes theorem.

![image](https://user-images.githubusercontent.com/54464437/174065855-81ab0750-018e-4958-9d5c-a4936a987eda.png)

Using Bayes theorem, we can find the probability of A happening, given that B has occurred. Here, B is the evidence and A is the hypothesis. The assumption made here is that the predictors/features are independent. That is presence of one particular feature does not affect the other. Hence it is called naive.

### Prediction Mathematics
Bayes theorem can be rewritten as:

![image](https://user-images.githubusercontent.com/54464437/174066133-e85556fc-93df-4d12-9441-dee0979d6d35.png)

The variable y is the class variable,. Variable X represent the parameters/features.

X is given as,

![image](https://user-images.githubusercontent.com/54464437/174066262-d91f38f1-1fd6-445a-b684-9bfa7ac982d0.png)

Here x_1,x_2….x_n represent the features. By substituting for X and expanding using the chain rule we get,

![image](https://user-images.githubusercontent.com/54464437/174066434-b5d7e8fb-8176-4342-95ba-1ce8e1b07c9d.png)

Now, you can obtain the values for each by looking at the dataset and substitute them into the equation. For all entries in the dataset, the denominator does not change, it remain static. Therefore, the denominator can be removed and a proportionality can be introduced.

![image](https://user-images.githubusercontent.com/54464437/174066507-c4deda7b-3ad2-4a5c-b7df-1b413675a2ec.png)

There could be cases where the classification could be multivariate. Therefore, we need to find the class y with maximum probability.

![image](https://user-images.githubusercontent.com/54464437/174066581-3aff3f75-0650-4bc4-8381-55deadddedb1.png)

Using the above function, we can obtain the class, given the predictors.

_NOTE_: the performance metric used was accuracy

## Perceptron
A Perceptron is an algorithm used for supervised learning of binary classifiers. Binary classifiers decide whether an input, usually represented by a series of vectors, belongs to a specific class.

In short, a perceptron is a single-layer neural network. They consist of four main parts including input values, weights and bias, net sum, and an activation function.

### Working of a perceptron
The process begins by taking all the input values and multiplying them by their weights. Then, all of these multiplied values are added together to create the weighted sum. The weighted sum is then applied to the activation function, producing the perceptron's output. The activation function plays the integral role of ensuring the output is mapped between required values such as (0,1) or (-1,1). It is important to note that the weight of an input is indicative of the strength of a node. Similarly, an input's bias value gives the ability to shift the activation function curve up or down.

![image](https://user-images.githubusercontent.com/54464437/174229906-21ecb3c1-bfef-406a-97b4-6af199c63ab9.png)

### Learning rate of perceptron
Perceptron Learning Rule states that the algorithm would automatically learn the optimal weight coefficients. The input features are then multiplied with these weights to determine if a neuron fires or not.

![image](https://user-images.githubusercontent.com/54464437/174229989-e2a3bd35-cf50-4ed5-b136-62e3a36bcaef.png)

The Perceptron receives multiple input signals, and if the sum of the input signals exceeds a certain threshold, it either outputs a signal or does not return an output. In the context of supervised learning and classification, this can then be used to predict the class of a sample.

### Perceptron functions
Perceptron is a function that maps its input “x,” which is multiplied with the learned weight coefficient; an output value ”f(x)”is generated.

![image](https://user-images.githubusercontent.com/54464437/174230171-8241e618-8f36-4146-af58-9add758f79b6.png)

In the equation given above:

1. “w” = vector of real-valued weights
2. “b” = bias (an element that adjusts the boundary away from origin without any dependence on the input value)
3. “x” = vector of input x values
4. “m” = number of inputs to the Perceptron

![image](https://user-images.githubusercontent.com/54464437/174230239-688be0c5-b7dd-427d-b62b-0df727953921.png)

The output can be represented as “1” or “0.”  It can also be represented as “1” or “-1” depending on which activation function is used.

### Inputs of a Perceptron
What is Perceptron: A Beginners Guide for Perceptron
By Simplilearn
Last updated on May 18, 202290099
What is Perceptron: A Beginners Tutorial for Perceptron
Table of Contents
Biological NeuronRise of Artificial Neurons (Based on Biological Neuron)What is Artificial NeuronBiological Neuron vs. Artificial NeuronArtificial Neuron at a GlanceView More
A perceptron is a neural network unit (an artificial neuron) that does certain computations to detect features or business intelligence in the input data. And this perceptron tutorial will give you an in-depth knowledge of Perceptron and its activation functions.

Let’s begin with understanding what is artificial neuron.

Post Graduate Program in AI and Machine Learning
In Partnership with Purdue UniversityEXPLORE COURSEPost Graduate Program in AI and Machine Learning
Biological Neuron
A human brain has billions of neurons. Neurons are interconnected nerve cells in the human brain that are involved in processing and transmitting chemical and electrical signals. Dendrites are branches that receive information from other neurons.

diagram-of-a-biological-neuron_1

Cell nucleus or Soma processes the information received from dendrites. Axon is a cable that is used by neurons to send information. Synapse is the connection between an axon and other neuron dendrites.

Let us discuss the rise of artificial neurons in the next section.

What are you waiting for? Interested in taking up a Deep Learning Course? Check out our Course Preview here!
Rise of Artificial Neurons (Based on Biological Neuron)
Researchers Warren McCullock and Walter Pitts published their first concept of simplified brain cell in 1943. This was called McCullock-Pitts (MCP) neuron. They described such a nerve cell as a simple logic gate with binary outputs.

Multiple signals arrive at the dendrites and are then integrated into the cell body, and, if the accumulated signal exceeds a certain threshold, an output signal is generated that will be passed on by the axon. In the next section, let us talk about the artificial neuron.

What is Artificial Neuron
An artificial neuron is a mathematical function based on a model of biological neurons, where each neuron takes inputs, weighs them separately, sums them up and passes this sum through a nonlinear function to produce output.

diagram-for-general-view-of-artificial-neuron_2.

In the next section, let us compare the biological neuron with the artificial neuron.

Free Course: Introduction to AI
Learn the Core AI Concepts and Key Skills for FREESTART LEARNINGFree Course: Introduction to AI
Biological Neuron vs. Artificial Neuron
The biological neuron is analogous to artificial neurons in the following terms:

Biological Neuron

Artificial Neuron

Cell Nucleus (Soma)

Node

Dendrites

Input

Synapse

Weights or interconnections

Axon

Output

Artificial Neuron at a Glance
The artificial neuron has the following characteristics:

A neuron is a mathematical function modeled on the working of biological neurons
It is an elementary unit in an artificial neural network
One or more inputs are separately weighted
Inputs are summed and passed through a nonlinear function to produce output
Every neuron holds an internal state called activation signal
Each connection link carries information about the input signal
Every neuron is connected to another neuron via connection link
In the next section, let us talk about perceptrons.

Perceptron 
Perceptron was introduced by Frank Rosenblatt in 1957. He proposed a Perceptron learning rule based on the original MCP neuron. A Perceptron is an algorithm for supervised learning of binary classifiers. This algorithm enables neurons to learn and processes elements in the training set one at a time.

general-diagram-of-perceptron-for-supervised-learning_4.

There are two types of Perceptrons: Single layer and Multilayer.

Single layer - Single layer perceptrons can learn only linearly separable patterns
Multilayer - Multilayer perceptrons or feedforward neural networks with two or more layers have the greater processing power
The Perceptron algorithm learns the weights for the input signals in order to draw a linear decision boundary.

This enables you to distinguish between the two linearly separable classes +1 and -1.

Note: Supervised Learning is a type of Machine Learning used to learn models from labeled training data. It enables output prediction for future or unseen data. Let us focus on the Perceptron Learning Rule in the next section.

Deep Learning Course (with TensorFlow & Keras)
Master the Deep Learning Concepts and ModelsVIEW COURSEDeep Learning Course (with TensorFlow & Keras)
Perceptron Learning Rule
Perceptron Learning Rule states that the algorithm would automatically learn the optimal weight coefficients. The input features are then multiplied with these weights to determine if a neuron fires or not.

symbolic-representation-of-perceptron-learning-rule_5.

The Perceptron receives multiple input signals, and if the sum of the input signals exceeds a certain threshold, it either outputs a signal or does not return an output. In the context of supervised learning and classification, this can then be used to predict the class of a sample.

Next up, let us focus on the perceptron function.

Perceptron Function
Perceptron is a function that maps its input “x,” which is multiplied with the learned weight coefficient; an output value ”f(x)”is generated.

Perceptron_6.

In the equation given above:

“w” = vector of real-valued weights
“b” = bias (an element that adjusts the boundary away from origin without any dependence on the input value)
“x” = vector of input x values
Perceptron_7.

“m” = number of inputs to the Perceptron
The output can be represented as “1” or “0.”  It can also be represented as “1” or “-1” depending on which activation function is used.

Let us learn the inputs of a perceptron in the next section.

Inputs of a Perceptron
A Perceptron accepts inputs, moderates them with certain weight values, then applies the transformation function to output the final result. The image below shows a Perceptron with a Boolean output.
A Boolean output is based on inputs such as salaried, married, age, past credit profile, etc. It has only two values: Yes and No or True and False. The summation function “∑” multiplies all inputs of “x” by weights “w” and then adds them up as follows:

![image](https://user-images.githubusercontent.com/54464437/174230521-cd4fe84a-0526-4570-981b-e22a777a8fed.png)







