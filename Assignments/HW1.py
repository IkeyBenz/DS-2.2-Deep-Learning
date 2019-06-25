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
# Build model for linear regression:
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_boston


# %%
df = load_boston()
X = pd.DataFrame(df.data)


# %%
