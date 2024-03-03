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
