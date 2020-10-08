#!/usr/bin/env python
# coding: utf-8

## Multiple Linear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the dataset
from sklearn.datasets import load_boston
dataset = load_boston()

## Print the keys of the dataset
dataset.keys()

## Print the description of the dataset to know about the features
print(dataset.DESCR)

## Load the data into a pandas DataFrame(We will only load the columns of features)
df = pd.DataFrame(dataset.data, columns = dataset.feature_names)

## Print the first 5 rows of the dataset
print(df.head())

## Load the target column(the y variable in regression) to the dataframe
df['MEDV'] = dataset.target
print(df.head())

## Check if there are any missing values in the data
print(df.isnull().sum())

## Print basic stats of the dataset
print(df.info())
print(df.describe().T)

## Plot the target variable to observe its characteristics 
plt.style.use('seaborn')
plt.hist(df['MEDV'].values)

import seaborn as sns

## Select the features (Selecting all the feature columns does not give a good prediction)
X = df[['LSTAT', 'RM', 'PTRATIO']]
X.head()

y = df['MEDV']
print(y.head())

sns.pairplot(df[['LSTAT', 'RM', 'PTRATIO', 'MEDV']], size= 2.5)

## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

## Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

## Predicting the Test set results
y_pred = regressor.predict(X_test)

## Print the predicted and actual values
np.set_printoptions(precision=2)
print(f"The predicted values are : \n {y_pred}")
print(f"The actual values are : \n {y_test.to_numpy()}")

## Evaluating the Model Performance
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
