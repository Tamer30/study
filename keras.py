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


#######################################

import pandas as pd
from tensorflow import keras
from sklearn.metrics import accuracy_score


df = pd.read_csv('/datasets/train_data_n.csv')
df['target'] = (df['target'] > df['target'].median()).astype(int)
features_train = df.drop('target', axis=1)
target_train = df['target']

df_val = pd.read_csv('/datasets/test_data_n.csv')
df_val['target'] = (df_val['target'] > df['target'].median()).astype(int)
features_valid = df_val.drop('target', axis=1)
target_valid = df_val['target']

model = keras.models.Sequential()
model.add(keras.layers.Dense(units=1, input_dim=features_train.shape[1], 
                             activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd')
model.fit(features_train, target_train, epochs=5, verbose=0,
          validation_data=(features_valid, target_valid))


predictions = model.predict(features_valid) > 0.5
score = accuracy_score(target_valid, predictions)
print("Accuracy:", score) 
