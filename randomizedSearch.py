# импортируем библиотеки и объявляем константы
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import time

RANDOM_STATE = 42

# подготавливаем данные заранее созданной функцией
X_train, X_test, y_train, y_test = prepare_data('train_satisfaction.csv')

# запускаем таймер
start = time.time() 

# инициализируем модель дерева решений
model = DecisionTreeClassifier(random_state=RANDOM_STATE)

# Создайте словарь с гиперпараметрами:
# - min_samples_split в диапазоне от 2 до 6 (не включительно);
# - min_samples_leaf в диапазоне от 1 до 6 (не включительно);
# - max_depth в диапазоне от 2 до 6 (не включительно).
parameters = {
    'min_samples_split': range(2, 6),
    'min_samples_leaf': range(1, 6),
    'max_depth': range(2, 6)
}

# Инициализируйте класс для автоматизированного случайного поиска:
# значение кросс-валидации 5, метрика roc-auc и n_jobs=-1.
rs = RandomizedSearchCV(
    model,
    parameters,
    n_jobs=-1,
    cv=5,
    scoring='roc_auc',
    random_state=RANDOM_STATE
)

# запустите поиск гиперпараметров
rs.fit(X_train, y_train)

# считаем, сколько секунд прошло с начала запуска
rs_search_time = time.time() - start
print(f'Search time:{rs_search_time}')

# выводим лучшие гиперпараметры
print('Гиперпараметры', rs.best_params_)
# выводим лучшую метрику качества
print('ROC-AUC', rs.best_score_)
