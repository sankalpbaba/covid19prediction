#!/usr/bin/env python
 ## Importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


pd.read_csv("C:/Users/sanka/OneDrive/Desktop/covid19-20.csv",squeeze=True)
#  Importing the dataset
dataset = pd.read_csv('covid19-20.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)


## Splitting the dataset into the Training set and Test set
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
X = ordinal_encoder.fit_transform(X)
print(X)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print(y_train)


ordinal_encoder = OrdinalEncoder()
ordinal_encoder.fit(X_train)
X_train = ordinal_encoder.transform(X_train)
X_test = ordinal_encoder.transform(X_test)


 ## Training XGBoost on the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
## Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

## Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

y_pred= classifier.predict(X_test)
print(y_pred)
