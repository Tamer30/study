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





################################### +validation +epochs

import pandas as pd
from tensorflow import keras

data_train = pd.read_csv('/datasets/train_data_n.csv')
features_train = data_train.drop('target', axis=1)
target_train = data_train['target']

data_valid = pd.read_csv('/datasets/test_data_n.csv')
features_valid = data_valid.drop('target', axis=1)
target_valid = data_valid['target']

model = keras.models.Sequential()
model.add(keras.layers.Dense(units=1, input_dim=features_train.shape[1]))
model.compile(loss='mean_squared_error', optimizer='sgd')
model.fit(features_train, target_train, verbose=2, epochs=5 ,validation_data=(features_valid, target_valid))
