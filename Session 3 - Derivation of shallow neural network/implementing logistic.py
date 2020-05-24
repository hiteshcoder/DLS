# -*- coding: utf-8 -*-
"""
Created on Sun May 17 13:09:15 2020

@author: hitz_
"""


import pandas as pd
import numpy as np 
#import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

df = pd.read_csv('Z:/Vision data science/Deep learning sessions/DLS/session 1-logistic regression/diabetes.csv') 
print(df.shape)
df.describe()

target_column = ['Outcome'] 
predictors = list(set(list(df.columns))-set(target_column))
#feature scaling
df[predictors] =df[predictors]/df[predictors].max()
df.describe()

X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(X_train.shape); print(X_test.shape)

y_train.shape
y_test.shape
X_train.shape
X_test.shape
#transposing the values of X and Y so that you understand as per the derivation
y_train_new=y_train.transpose()
y_test_new=y_test.transpose()
x_train_new=X_train.transpose()
x_test_new=X_test.transpose()

x_train_new.shape
y_train_new.shape
x_test_new.shape
y_test_new.shape

#-building the sigmoid function
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

#initialize the parameters and making sure of their dimensions and value of b can be float or int

def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b

#for our case our dimensions are 8 
dim = 8
w, b = initialize_with_zeros(dim)
print ("w = " + str(w))
print ("b = " + str(b))
w.shape

#lets define functions for forward propagation and back propagation
def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads = {"dw": dw,
             "db": db}
    return grads, cost

#lets check for an example
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))

#updating the parameters
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        
        
        w = w - learning_rate * dw 
        b = b - learning_rate * db
       
        if i % 100 == 0:
            costs.append(cost)
        
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

#define the predict function
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
       
    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
        
    assert(Y_prediction.shape == (1, m))
    return Y_prediction

#now lets merge all the individual functions those are formed

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

d = model(x_train_new, y_train_new, x_test_new, y_test_new, num_iterations = 2000, learning_rate = 0.005, print_cost = True)










