#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import money_flow_index
from ta.momentum import rsi  # Relative Strength Index
from ta.momentum import stoch  # Stochastic Oscilator
from ta.momentum import uo  # Ultimate Oscilator
from ta.momentum import wr  # William Percent Range
from ta.trend import macd  # Moving Average Convergence/Divergence


# *************************** DATA PREPROCESSING ******************************

# Load Data
df = pd.read_csv('^GSPC.csv')

X = df.drop(['Open', 'Date', 'Adj Close'], axis=1)
y = np.array(df['Open']).reshape(-1, 1)

# Data Augmentation
# df['Relative_Strength_Index'] = rsi(df['Close'])
# df['Money_Flow_Index'] = money_flow_index(
#     df['High'], df['Low'], df['Close'], df['Volume'])
# df['Stoch_Oscilator'] = stoch(df['High'], df['Low'], df['Close'])
# df['Ultimate_Oscilator'] = uo(df['High'], df['Low'], df['Close'])
# df['William_Percent'] = wr(df['High'], df['Low'], df['Close'])
# df['MACD'] = macd(df['Close'])


# Some indicators require many days in advance before they produce any
# values. So the begining rows of our df may have NaNs. Lets drop them:
df = df.dropna()

# Scaling Data
input_scaler = MinMaxScaler(feature_range=(0, 1))   # Using twp scalers becasue
output_scaler = MinMaxScaler(feature_range=(0, 1))  # the putput is a scalar,
scaled_X = input_scaler.fit_transform(X)            # thats gonna mess up the
scaled_y = output_scaler.fit_transform(y)           # inverse_transform later


# Creating a data structure with 60 timesteps and 1 output
X = np.array([scaled_X[i:i+60, :] for i in range(len(scaled_X)-60)])
y = np.array([scaled_y[i+60, 0] for i in range(len(scaled_y)-60)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)


# *************************** BUILDING RNN ************************************

# regressor = Sequential([

#     LSTM(  # Add First LSTM Layer
#         units=50,
#         return_sequences=True,  # Allow access to hidden state for next layers
#         input_shape=(X_train.shape[1], X_train.shape[2])
#     ),
#     Dropout(0.2),  # Prevent Overfitting

#     LSTM(units=50, return_sequences=True),
#     Dropout(0.2),

#     LSTM(units=50, return_sequences=True),
#     Dropout(0.2),

#     LSTM(units=50),  # Don't return sequences becasue this is last LSTM layer
#     Dropout(0.2),

#     Dense(units=1)  # Output layer

# ])
# regressor.compile(optimizer='adam', loss='mean_squared_error')

# regressor.fit(X_train, y_train, epochs=5, batch_size=32)
# regressor.save('just-close-high-low-volume.h5')

# *********************** MODEL EVALUATION ************************************
regressor = load_model('just-close-high-low-volume.h5')
y_pred = regressor.predict(X_test)

y_true = output_scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred = output_scaler.inverse_transform(y_pred)

accuracy = explained_variance_score(y_true, y_pred)  # = 0.998015
mae = mean_absolute_error(y_true, y_pred)            # = 16.31382
mse = mean_squared_error(y_true, y_pred)             # = 1096.1698
msle = mean_squared_log_error(y_true, y_pred)        # = 0.006196
r2e = r2_score(y_true, y_pred)                       # = 0.99766

# ************************ GRAPHING RESULTS ***********************************
# train_test_split makes our data out of order!!!! how tf to graph??
# okay we are going to take the last 5 months of data (arbitrary choice)
# in order to graph what we predict.


def back_test(days=100):
    '''Takes the last days from the dataframe, makes predictions using our
       regressor, and plots y_true alongside y_pred.
    '''
    past_x = np.array([scaled_X[-i-60:-i, :] for i in range(days+1, 1, -1)])
    past_y = np.array([scaled_y[-i, 0] for i in range(days+1, 1, -1)])

    predictions = regressor.predict(past_x)

    y_true = output_scaler.inverse_transform(past_y.reshape(-1, 1))
    y_pred = output_scaler.inverse_transform(predictions)

    plt.plot(y_true, label="y_true")
    plt.plot(y_pred, label="y_pred")
    plt.legend()
    plt.show()
