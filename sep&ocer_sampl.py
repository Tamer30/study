#Задача 1
#Разделите данные для бинарной классификации на тренировочную, валидационную и тестовую выборки. Сэмплируйте тренировочные и валидационные данные методом RandomOverSampler.


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Создание константы RANDOM_STATE
RANDOM_STATE = 42

# Загрузка данных из CSV-файла в датафрейм pandas
df = pd.read_csv('music_genre_2_classes_imbalanced_v2.csv')
X = df.drop(columns='music_genre')
y = df.music_genre

# Сформируйте тренировочную, валидационную и тестовую выборки в соотношении 3-1-1 
# Стратифицируйте данные и зафиксируйте random_state
X_train, X_test_valid, y_train, y_test_valid = train_test_split(
    X,
    y,
    test_size=0.4,
    stratify=y,
    random_state=RANDOM_STATE)

X_valid, X_test, y_valid, y_test = train_test_split(
    X_test_valid,
    y_test_valid,
    test_size=0.5,
    stratify=y_test_valid,
    random_state=RANDOM_STATE)

# Выполняем подготовку данных функцией, все операции в скрытом прекоде
X_train, X_valid, X_test = prepare_data(X_train, X_valid, X_test)

# Создайте экземпляр класса RandomOverSampler с фиксированным random_state
sampler = RandomOverSampler(random_state=RANDOM_STATE)

# Сэмплируйте данные одной командой
X_train_sampled, y_train_sampled = sampler.fit_resample(X_train, y_train)
X_valid_sampled, y_valid_sampled = sampler.fit_resample(X_valid, y_valid)

# Выведите на экран три столбчатых диаграммы с распределением целевого признака
# Отложите на них исходные данные, тренировочную и валидационную выборки
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
pd.Series(y).value_counts().plot(kind='bar', rot=0, ax=axes[0])
pd.Series(y_train_sampled).value_counts().plot(kind='bar', rot=0, ax=axes[1])
pd.Series(y_valid_sampled).value_counts().plot(kind='bar', rot=0, ax=axes[2])

axes[0].set_title('Исходные данные')
axes[1].set_title('train после сэмплирования')
axes[2].set_title('valid после сэмплирования')
plt.show()
