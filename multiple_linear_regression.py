#!/usr/bin/env python
# coding: utf-8

# # Multiple Linear Regression

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[2]:


from sklearn.datasets import load_boston
dataset = load_boston()


# ## Print the keys of the dataset

# In[3]:


dataset.keys()


# ## Print the description of the dataset to know about the features

# In[4]:


print(dataset.DESCR)


# ## Load the data into a pandas DataFrame 
# (We will only load the columns of features for ease of use)

# In[5]:


df = pd.DataFrame(dataset.data, columns = dataset.feature_names)


# ## Print the first 5 rows of the dataset

# In[6]:


print(df.head())


# ## Load the target column(the y variable in regression) to the dataframe

# In[7]:


df['MEDV'] = dataset.target


# In[8]:


print(df.head())


# ## Check if there are any missing values in the data

# In[9]:


df.isnull()


# In[10]:


df.isnull().sum()


# ## Print basic stats of the dataset

# In[11]:


print(df.info())


# In[12]:


df.describe().T


# ## Plot the target variable to observe its characteristics 

# In[13]:


plt.style.use('seaborn')


# In[14]:


plt.hist(df['MEDV'].values)


# In[15]:


import seaborn as sns


# ## Select the features (Selecting all the feature columns does not give a good prediction)

# In[16]:


X = df[['LSTAT', 'RM', 'PTRATIO']]
X.head()


# In[17]:


y = df['MEDV']
print(y.head())


# In[21]:


sns.pairplot(df[['LSTAT', 'RM', 'PTRATIO', 'MEDV']], size= 2.5)


# ## Splitting the dataset into the Training set and Test set

# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ## Training the Multiple Linear Regression model on the Training set

# In[23]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# ## Predicting the Test set results

# In[24]:


y_pred = regressor.predict(X_test)


# ## Print the predicted and actual values

# In[25]:


np.set_printoptions(precision=2)
print(f"The predicted values are : \n {y_pred}")


# In[26]:


print(f"The actual values are : \n {y_test.to_numpy()}")


# ## Evaluating the Model Performance

# In[27]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

