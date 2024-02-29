#Задача 1
#В этом задании вы должны дополнить подготовку данных, поэтому мы сделали весь код видимым.
#Масштабируйте количественные признакипри помощи StandardScaler() и вычислите Манхэттенское и Евклидово расстояния между новым наблюдением и наблюдениями из тренировочной выборки.
#Отсортируйте расстояния между наблюдениями по возрастанию и выведите на экран 10 значений объектов, наиболее близких к новому.

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# импортируем из библиотеки scipy функцию для расчёта расстояния
from scipy.spatial.distance import euclidean, cityblock

# импортируйте из библиотеки sclearn класс для масштабирования
from sklearn.preprocessing import StandardScaler

# создание константы RANDOM_STATE
RANDOM_STATE = 42

def prepare_data(fname):
    # добавляем данные из CSV-файла в датафрейм pandas, делим датасет на выборки
    df = pd.read_csv(fname)
    X = df.drop(columns='music_genre')
    y = df.music_genre

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        random_state=RANDOM_STATE
    )

    # создайте списки с количественными и категориальными признаками
    cat_col_names = X_train.select_dtypes(exclude='number').columns.tolist()
    num_col_names = X_train.select_dtypes(include='number').columns.tolist()

    # выберите класс OneHotEncoder() для кодирования 
    # задайте 'sparse=False', избегайте появление дамми-ловушки
    ohe = OneHotEncoder(sparse=False, drop='first')

    # обучите и преобразуйте категориальные признаки из тренировочной и тестовой выборок 
    # для тренировочной выборки сделайте это одной командой
    X_train_ohe = ohe.fit_transform(X_train[cat_col_names])
    X_test_ohe = ohe.transform(X_test[cat_col_names])

    # сохраните в переменной encoder_col_names список названий новых столбцов 
    encoder_col_names = ohe.get_feature_names()

    # создайте датафрейм из закодированных данных
    # передайте названия столбцов из переменной encoder_col_names
    X_train_ohe = pd.DataFrame(X_train_ohe, columns=encoder_col_names)
    X_test_ohe = pd.DataFrame(X_test_ohe, columns=encoder_col_names)

    # выберите класс StandartScaler() для масштабирования 
    scaler = StandardScaler()

    # масштабируйте количественные признаки из тренировочной и тестовой выборок 
    # для тренировочной выборки сделайте это одной командой
    X_train[num_col_names] = scaler.fit_transform(X_train[num_col_names])
    X_test[num_col_names] = scaler.transform(X_test[num_col_names])

    # обнулите индексы строк перед объединением числовых и категориальных признаков в датафрейм
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    X_train = pd.concat((X_train[num_col_names], X_train_ohe), axis=1)
    X_test = pd.concat((X_test[num_col_names], X_test_ohe), axis=1)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = prepare_data('music_genre_2_classes_balanced_v2_exp.csv')

sample = X_test.sample(1, random_state=RANDOM_STATE)

# вычислите расстояния между тестовым наблюдением и тренировочным датасетом
euclidian_df = X_train.apply(lambda x: euclidean(x.values, sample.values[0]) , axis=1).rename('euclidean')
manhatten_df = X_train.apply(lambda x: cityblock(x.values, sample.values[0]) , axis=1).rename('manhatten')

# создайте датафрейм, включающий в себя euclidian_df и y_train
euclidian_df = pd.concat((euclidian_df, y_train), axis=1)

# создайте датафрейм, включающий в себя manhatten_df и y_train
manhatten_df = pd.concat((manhatten_df, y_train), axis=1)

# выполните сортировку по возрастанию расстояний и выведите 10 наблюдений
print(euclidian_df.sort_values(by='euclidean').head(10))
print(manhatten_df.sort_values(by='manhatten').head(10))

##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################

#Задача 2
#Выполните подготовку данных и обучите модель kNN —  одну с Евклидовым расстоянием, другую — с Манхэттенским. Оцените качество решения модели метрикой accuracy в обоих случаях.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# импортируйте функцию для оценки качества модели 
from sklearn.metrics import accuracy_score

# импортируйте класс для модели классификации kNN 
from sklearn.neighbors import KNeighborsClassifier

# создание константы RANDOM_STATE
RANDOM_STATE = 42

# выполняем подготовку данных функцией, все операции в скрытом прекоде
X_train, X_test, y_train, y_test = prepare_data('music_genre_2_classes_balanced_v2_exp.csv')


# создайте и обучите модель kNN на тренировочных данных
# при обучении модель должна использовать 300 ближайших соседей и Евклидово расстояние
model = KNeighborsClassifier(n_neighbors=300, metric='euclidean')
model.fit(X_train, y_train)
preds = model.predict(X_test)

# выполните тестирование на тестовой выборке
euclidean_accuracy = accuracy_score(y_test, preds)

# создайте и обучите модель kNN на тренировочных данных
# при обучении модель должна использовать 300 ближайших соседей и Манхэттенское расстояние
model = KNeighborsClassifier(n_neighbors=300, metric='cityblock')
model.fit(X_train, y_train)
preds = model.predict(X_test)

# выполните тестирование на тестовой выборке
cityblock_accuracy = accuracy_score(y_test, preds)

# выведите на экран обе метрики
print('Accuracy (euclidean distance) =', euclidean_accuracy)
print('Accuracy (cityblock distance) =', cityblock_accuracy)
