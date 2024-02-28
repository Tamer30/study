
#Задача 2
#Рассчитайте Манхэттенское расстояние между наблюдением из тестовой выборки и всеми объектами в тренировочной.
#Отсортируйте расстояния между наблюдениями по возрастанию и выведите на экран 10 значений для объектов, наиболее близких к новому.

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# импортируйте из библиотеки scipy функцию для расчёта Манхэттенского расстояния
from scipy.spatial.distance import cityblock

# создание константы RANDOM_STATE
RANDOM_STATE = 42

# выполняем подготовку данных функцией, все операции в скрытом прекоде
X_train, X_test, y_train, y_test = prepare_data('music_genre_2_classes_balanced_v2_exp.csv')

# выберите из тестовой выборки одно наблюдение
sample = X_test.sample(1, random_state=RANDOM_STATE)

# вычислите Манхэттенское расстояние между тестовым наблюдением 
# и объектами в тренировочной выборке
X_train['manhatten'] = X_train.apply(lambda x: cityblock(x.values, sample.values[0]), axis=1)

# создаём датафрейм, включающий в себя столбец X_train['manhatten'] и y_train
result_df = pd.concat((X_train['manhatten'], y_train), axis=1)

# отсортируйте по возрастанию расстояния (столбец 'manhatten') 
# выведите 10 наблюдений, ближайших к новому
# выведите столбцы ['music_genre','manhatten']
print(result_df.sort_values(by='manhatten').head(10))
