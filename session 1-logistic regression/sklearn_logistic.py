# -*- coding: utf-8 -*-
"""
Created on Thu May  7 19:16:37 2020

@author: hitz_
"""
#scientific package to do broadcasting and linear algebra functions
import numpy as np
#package for data manupulation or massaging 
import pandas as pd
#for the normal cumulative distribution function
from scipy.stats import norm
#machine learning algorithms
from sklearn.linear_model import LogisticRegression
#dividing the data into test and train
from sklearn.model_selection import train_test_split

#https://www.kaggle.com/uciml/pima-indians-diabetes-database

df = pd.read_csv('Z:/Vision data science/session 1-logistic regression/diabetes.csv') 
print(df.shape)
#understanding the data
df.describe()
df.head(3)
df.columns

# converting the dataset into dependent and independent varaible
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
X = df[feature_cols] 

# Target variable/dependent variable
y = df.Outcome 

#converting into test and train datasets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.3, random_state=40)
logreg = LogisticRegression(penalty='none',solver='saga')
#forming the equation
logreg.fit(X_train1, y_train1)
#predictiong y values from that equation
y_pred = logreg.predict(X_test1)
#the coefficients of the equation.
logreg.coef_
#print statement to print the accuracy
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test1, y_test1)))

#printing the confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test1, y_pred)
print(confusion_matrix)

#classification report population
from sklearn.metrics import classification_report
print(classification_report(y_test1, y_pred))

#finding the p-value from Logit
def logit_pvalue(model, x):
    """ Calculate z-scores for scikit-learn LogisticRegression.
    parameters:
        model: fitted sklearn.linear_model.LogisticRegression with intercept and large C
        x:     matrix on which the model was fit
    This function uses asymtptics for maximum likelihood estimates.
    """
    p = model.predict_proba(x)
    n = len(p)
    m = len(model.coef_[0]) + 1
    coefs = np.concatenate([model.intercept_, model.coef_[0]])
    x_full = np.matrix(np.insert(np.array(x), 0, 1, axis = 1))
    ans = np.zeros((m, m))
    for i in range(n):
        ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i,1] * p[i, 0]
    vcov = np.linalg.inv(np.matrix(ans))
    se = np.sqrt(np.diag(vcov))
    t =  coefs/se  
    p = (1 - (norm.cdf(abs(t)))) * 2
    return p

print(logit_pvalue(logreg, X_train1))
