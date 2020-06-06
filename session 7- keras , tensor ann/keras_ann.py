# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 14:02:24 2020

@author: hitz_
"""
# =============================================================================
# List of keras initializers that you can use 
# https://keras.io/api/layers/initializers/
# By default keras uses 'glorot_uniform'(thats good for tanh)
# For a relu implementation -'he_uniform'
# to read about great initialization techniques check the paper I posted
# https://www.linkedin.com/posts/hitesh-nayak-27415750_weight-initialization-method-for-improving-activity-6668395185665396736-r1r1

# =============================================================================

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

# Importing the dataset
dataset = pd.read_csv('Z:/Vision data science/Deep learning sessions/DLS/session 7- keras , CNN/Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)
# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = Sequential()
model.add(Dense(6, activation='relu', input_dim=12))
#model.add(Dense(6, activation='relu', input_dim=12,init ='he_normal'))
# https://faroit.com/keras-docs/1.2.2/initializations/
# for initialization details
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
help( model.compile)
# build the model
model.fit(X_train, y_train, epochs=100,batch_size=32)

# evaluate the keras model
pred_train= model.predict(X_train)
scores = model.evaluate(X_train, y_train, verbose=0)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   
 
pred_test= model.predict(X_test)
scores2 = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))    


# =============================================================================
# adding regularization 
# 1.dropout regularization
# model = Sequential()
# model.add(Dense(60, input_dim=60, activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.2))
# model.add(Dense(30, activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid'))
# =============================================================================

# =============================================================================
# 2. adding l1_l2 regularization
# layer = layers.Dense(
#     units=64,
#     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
#     bias_regularizer=regularizers.l2(1e-4),
#     activity_regularizer=regularizers.l2(1e-5)
# )
# kernel_regularizer: Regularizer to apply a penalty on the layer's kernel
# bias_regularizer: Regularizer to apply a penalty on the layer's bias
# activity_regularizer: Regularizer to apply a penalty on the layer's output
# =============================================================================
