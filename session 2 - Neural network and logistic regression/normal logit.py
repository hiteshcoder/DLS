# -*- coding: utf-8 -*-
"""
Created on Thu May  7 19:06:32 2020

@author: hitz_
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

df = pd.read_csv('Z:/Vision data science/session 1-logistic regression/diabetes.csv') 
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

# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

count_classes = y_test.shape[1]
print(count_classes)

#neural network
model1 = Sequential()
#1 neuron
model1.add(Dense(2, activation='sigmoid',input_dim=8))

# Compile the model
model1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# build the model
model1.fit(X_train, y_train, epochs=30)

pred_train1= model1.predict(X_train)
scores1 = model1.evaluate(X_train, y_train, verbose=0)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores1[1], 1 - scores1[1]))   
 
pred_test1= model1.predict(X_test)
scores2_1 = model1.evaluate(X_test, y_test, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2_1[1], 1 - scores2_1[1]))    

#Accuracy on test data: 0.6147186160087585% 
#Error on test data: 0.38528138399124146