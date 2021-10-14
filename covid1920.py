#!/usr/bin/env python
# coding: utf-8

# # XGBoost

# ## Importing the libraries

# In[51]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[78]:


pd.read_csv("C:/Users/sanka/OneDrive/Desktop/covid19-20.csv",squeeze=True)


# In[ ]:





# In[ ]:





#  Importing the dataset

# In[52]:


dataset = pd.read_csv('covid19-20.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[53]:


print(X)


# In[54]:


print(y)


# ## Splitting the dataset into the Training set and Test set

# In[26]:


from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
X = ordinal_encoder.fit_transform(X)


# In[27]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough', )
X = np.array(ct.fit_transform(X))


# In[28]:


print(X)


# In[29]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# In[30]:


print(y)


# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[32]:


print(y_train)


# In[33]:


ordinal_encoder = OrdinalEncoder()
ordinal_encoder.fit(X_train)
X_train = ordinal_encoder.transform(X_train)
X_test = ordinal_encoder.transform(X_test)


# ## Training XGBoost on the Training set

# In[34]:


from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)


# ## Making the Confusion Matrix

# In[35]:


from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# ## Applying k-Fold Cross Validation

# In[37]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# In[76]:


y_pred= classifier.predict(X_test)
print(y_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




