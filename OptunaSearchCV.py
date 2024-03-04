# импортируем библиотеки и объявляем константы
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from optuna import distributions
from optuna.integration import OptunaSearchCV
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
    'min_samples_split': distributions.IntDistribution(2, 5),
    'min_samples_leaf': distributions.IntDistribution(1, 5),
    'max_depth': distributions.IntDistribution(2, 5)
}

# Инициализируйте класс для байесовского поиска на 20 итераций:
# значение кросс-валидации 5, метрика roc-auc.
oscv = OptunaSearchCV(
    model,
    parameters,
    cv=5,
    scoring='roc_auc',
    n_trials=20,
    random_state=RANDOM_STATE
)

# запустите поиск гиперпараметров
oscv.fit(X_train, y_train)

# считаем, сколько секунд прошло с начала запуска
oscv_search_time = time.time() - start
print(f'Search time:{oscv_search_time}')

# выведите лучшие гиперпараметры
print('Гиперпараметры', oscv.best_params_)
# выведите лучшую метрику качества
print('ROC-AUC', oscv.best_score_)
