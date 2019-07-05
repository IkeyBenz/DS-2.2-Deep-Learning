# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'Assignments'))
    print(os.getcwd())
except:
    pass
# %% [markdown]
# ## Homework 1:
#
# 1- Build a Keras Model for linear regression (check: https://keras.io/activations/). Use Boston Housing Dataset to train and test your model
#
# 2- Build a Keras Model for logistic regression. Use diabetes.csv to train and test
#
# Comments:
#
# 1- Build the **simplest model** for linear regression with Keras and compare your model performance with `from sklearn.linear_model import LinearRegression`
#
# 2- Build the **simplest model** for logistic regression with Keras and compare your model performance with `from sklearn.linear_model import LogisticRegression`
#
# 3- **Add more complexity to your models in (1) and (2)** and compare with previous results

# %%
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston

# %% Loading Data
data = load_boston()
X = pd.DataFrame(data=data.data, columns=data.feature_names)
y = pd.DataFrame(data=data.target, columns=['MEDV'])

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %% Building Model
regressor = Sequential([
    Dense(activation='linear', input_dim=13,
          units=1, kernel_initializer='uniform'),
    Dense(activation='linear', units=1, kernel_initializer='uniform')
])
regressor.compile(optimizer='adam', loss='mse', metrics=['mse'])

# %% Fitting Model
regressor.fit(X_train, y_train, batch_size=1, epochs=100)


# %% Making Predictions
y_pred = regressor.predict(X_test)

# %% Comparing to SKLearn.Linreg
linreg = LinearRegression().fit(X_train, y_train)

print(linreg.score(X_test, y_test))
print(r2_score(y_test, y_pred))
