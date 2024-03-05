# импортируем нужные библиотеки и объявляем константы
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

TEST_SIZE = 0.2
RANDOM_STATE = 42

# считываем данные из CSV-файла
df = pd.read_csv('progulka.csv')

# делим данные на входные и целевые
X = df.drop('гулять?', axis=1)
y = df['гулять?']

# разделите данные на тренировочные и тестовые
# используйте константы с размером тестовой выборки и random_state
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=RANDOM_STATE,
    test_size=TEST_SIZE)

# инициализируйте модель дерева решений и обучите её на тренировочных данных
model = DecisionTreeClassifier(random_state=RANDOM_STATE)
model.fit(X_train, y_train)

# визуализируйте схему дерева решений
plot_tree(decision_tree=model, filled=True, feature_names=df.columns);


##############################################################################################

Инициализируем модель дерева решений с базовыми настройками:

DecisionTreeClassifier(
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=None,
    max_leaf_nodes=None
) 

Все указанные здесь значения — это гиперпараметры. Их можно менять при инициализации модели,
за исключением одного знакомого вам random_state — он фиксируется один раз, и больше его трогать не стоит.
Гиперпараметр max_depth ограничивает максимальную глубину дерева, а max_leaf_nodes — ограничивает число листьев.

###################################################################################################

# импортируем нужные библиотеки и объявляем константы
import time

import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.25

# считываем данные из CSV-файла в датафрейм и задаём id в качестве индексов
df = pd.read_csv('train_satisfaction.csv')
df = df.set_index('id')

# делим данные на входные и целевые
X = df.drop(['Удовлетворён предоставленной услугой'], axis=1)
y = df['Удовлетворён предоставленной услугой']

# делим данные на тренировочные и тестовые
# используем константы с размером тестовой выборки и random_state
# выполняем стратификацию по целевому признаку
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
# создаём список для фиксации результатов обучения модели
cv_results = []

# переберите значения min_samples_split в диапазоне от 2 до 6 (не включительно)
for min_samples_split in range(2, 6):
    # переберите значения min_samples_leaf в диапазоне от 1 до 6 (не включительно)
    for min_samples_leaf in range(1, 6):
        # переберите значения max_depth в диапазоне от 2 до 6 (не включительно)
        for max_depth in range(2, 6):
            # инициализируйте модель дерева решений с этими гиперпараметрами
            model = DecisionTreeClassifier(random_state=RANDOM_STATE,
                                          min_samples_split=min_samples_split,
                                          min_samples_leaf=min_samples_leaf,
                                          max_depth=max_depth)

            # посчитайте метрику roc-auc при кросс-валидации
            roc_auc_cv = cross_val_score(
                model,
                X_train,
                y_train,
                scoring='roc_auc',
                n_jobs=-1).mean()

            # добавляем метрику и гиперпараметры в список 
            cv_results.append((roc_auc_cv, {
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'max_depth': max_depth
            }))

# считаем, сколько секунд прошло с начала запуска
loop_search_time = time.time() - start
print(f'Search time:{loop_search_time}')

# сортируем результаты метрики от самых больших до самых маленьких
loop_search_cv_results = pd.DataFrame(sorted(cv_results, key=lambda x: x[0], reverse=True))
loop_search_cv_results.columns = ['Score', 'Params']
print(loop_search_cv_results[:10])


######################################################################################################

# обучите модель
model.fit(X_train, y_train)

# сформируйте таблицу важности признаков
feature_importances = pd.DataFrame(
    {
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    })
