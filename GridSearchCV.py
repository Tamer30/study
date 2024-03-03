# импортируем библиотеки и объявляем константы
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.2

# считываем данные из CSV-файла
df = pd.read_csv('progulka_extended.csv')

# делим данные на входные и целевые
X = df.drop('гулять?', axis=1)
y = df['гулять?']

# делим данные на тренировочные и тестовые
# используем константы с размером тестовой выборки и random_state
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

# инициализируем модель дерева решений
model = DecisionTreeClassifier(random_state=RANDOM_STATE)

# Создайте словарь с гиперпараметрами:
# - min_samples_split в диапазоне от 2 до 6 (не включительно);
# - min_samples_leaf в диапазоне от 1 до 3 (не включительно);
# - max_depth в диапазоне от 2 до 4 (не включительно).
parameters = {'min_samples_split': range(2, 6),
             'min_samples_leaf': range(1, 3),
             'max_depth': range(2, 4)}

# Инициализируйте класс для автоматизированного поиска:
# значение кросс-валидации 5, метрика accuracy и n_jobs=-1.
gs = GridSearchCV(
    model,
    parameters,
    n_jobs=-1,
    cv=5,
    scoring='accuracy'
)

# запустите поиск гиперпараметров
gs.fit(X_train, y_train) 

# выводим лучшие гиперпараметры
print(gs.best_params_)

#########################################

'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2}

Изучаем атрибуты GridSearchCV
Пока мы вас вскользь познакомили с одним атрибутом GridSearchCV — best_params_. Что ещё вам может пригодиться:

    best_estimator_ — лучшая обученная модель;
    best_score_ — лучшая метрика при кросс-валидации;
    cv_results_ — общие результаты поиска гиперпараметров.

###################################################################################################################

# импортируем библиотеки и объявляем константы
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import time

RANDOM_STATE = 42
TEST_SIZE = 0.25

# считываем данные из CSV-файла и задаём id в качестве индексов
df = pd.read_csv('train_satisfaction.csv')
df = df.set_index('id')

# делим данные на входные и целевые
X = df.drop(['Удовлетворён предоставленной услугой'], axis=1)
y = df['Удовлетворён предоставленной услугой']

# делим данные на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y)

# создаём список со строковыми категориями для кодирования
cols_ohe = X_train.select_dtypes(include='object').columns.tolist()

# создаём экземпляр класса OneHotEncoder для кодирования
oh_encoder = OneHotEncoder(drop='first', sparse=False)

# обучаем OneHotEncoder на категориальных признаках из тренировочной выборки
oh_encoder.fit(X_train[cols_ohe])

# сохраняем в переменной encoder_col_names список названий новых столбцов
encoder_col_names = oh_encoder.get_feature_names()

# преобразовываем категориальные переменные в тренировочной и тестовой выборках
X_train[encoder_col_names] = oh_encoder.transform(X_train[cols_ohe])
X_test[encoder_col_names] = oh_encoder.transform(X_test[cols_ohe])

# удаляем преобразованные признаки
X_train = X_train.drop(cols_ohe, axis=1)
X_test = X_test.drop(cols_ohe, axis=1)

# запускаем таймер
start = time.time()

# инициализируем модель дерева решений
model = DecisionTreeClassifier(random_state=RANDOM_STATE)

# Создайте словарь с гиперпараметрами:
# - min_samples_split в диапазоне от 2 до 6 (не включительно);
# - min_samples_leaf в диапазоне от 1 до 6 (не включительно);
# - max_depth в диапазоне от 2 до 6 (не включительно).
parameters = {'min_samples_split': range(2, 6),
             'min_samples_leaf': range(1, 6),
             'max_depth': range(2, 6)}

# Инициализируйте класс для автоматизированного поиска:
# значение кросс-валидации 5, метрика roc-auc и n_jobs=-1.
gs = GridSearchCV(
    model,
    parameters,
    n_jobs=-1,
    cv=5,
    scoring='roc_auc'
)

# запустите поиск гиперпараметров
gs.fit(X_train, y_train) 

# считаем, сколько секунд прошло с начала запуска
gs_search_time = time.time() - start
print(f'Search time:{gs_search_time}')

# выведите лучшие гиперпараметры
print('Гиперпараметры', gs.best_params_)
# выведите лучшую метрику качества
print('ROC-AUC', gs.best_score_)
