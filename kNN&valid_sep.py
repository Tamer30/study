import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score

# Создание константы RANDOM_STATE
RANDOM_STATE = 42

# Загрузка данных из CSV-файла в датафрейм pandas
df = pd.read_csv('music_genre_2_classes_imbalanced_v2.csv')
X = df.drop(columns='music_genre')
y = df.music_genre

# Формирование тренировочной, валидационной и тестовой выборок в соотношении 3-1-1 (60%-20%-20%)
X_train, X_test_valid, y_train, y_test_valid = train_test_split(X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y)

X_valid, X_test, y_valid, y_test = train_test_split(X_test_valid, y_test_valid, test_size=0.5, random_state=RANDOM_STATE, stratify=y_test_valid)

# Выполняем подготовку данных функцией, все операции в скрытом прекоде
X_train, X_valid, X_test = prepare_data(X_train, X_valid, X_test)

# Создаём экземпляр класса RandomOverSampler с фиксированным random_state
sampler = RandomOverSampler(random_state=RANDOM_STATE)

# Сэмплируем данные
X_train_sampled, y_train_sampled = sampler.fit_resample(X_train, y_train)
X_valid_sampled, y_valid_sampled = sampler.fit_resample(X_valid, y_valid)

# Создайте и обучите модель kNN на тренировочных данных
# Выполните предсказание на тестовых данных
model = KNeighborsClassifier()
model.fit(X_train_sampled, y_train_sampled)

# Посчитайте F1-меру на валидационной сэмплированной выборке
y_pred_valid_sampled = model.predict(X_valid_sampled)
f1_valid_sampled = f1_score(y_valid_sampled, y_pred_valid_sampled, pos_label='Rock')
print('Проверка модели на сэмплированной валидационной выборке: F1 =', round(f1_valid_sampled, 2))

# Посчитайте F1-меру на валидационной не сэмплированной выборке
y_pred_valid = model.predict(X_valid)
f1_valid = f1_score(y_valid, y_pred_valid, pos_label='Rock')
print('Проверка модели на НЕ сэмплированной валидационной выборке: F1 =', round(f1_valid, 2))

# Посчитайте F1-меру на тестовой выборке
y_pred_test = model.predict(X_test)
f1_test = f1_score(y_test, y_pred_test, pos_label='Rock')
print('Проверка модели на тестовой выборке: F1 =', round(f1_test, 2))
