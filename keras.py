import pandas as pd
from sklearn.linear_model import LinearRegression
from tensorflow import keras

data = pd.read_csv('/datasets/train_data_n.csv')
features = data.drop('target', axis=1)
target = data['target']

model = keras.models.Sequential()
model.add(keras.layers.Dense(units=1, input_dim=features.shape[1]))
model.compile(loss='mean_squared_error', optimizer='sgd')

model.fit(features, target, verbose=2)
