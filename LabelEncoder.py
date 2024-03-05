# импортируем библиотеки и объявляем константы
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV

RANDOM_STATE = 42

# считываем данные из CSV-файла
df_full = pd.read_csv('railway_full.csv')

# делим данные на входные и целевые
X = df_full.drop(['Общая оценка качества предоставленной услуги'], axis=1)
y = df_full['Общая оценка качества предоставленной услуги']

# подготавливаем данные заранее созданной функцией
X_train, X_test, y_train, y_test = prepare_data(X, y)

# создайте экземпляр класса LabelEncoder для кодирования целевого признака
label_encoder = LabelEncoder()

# обучите модель и трансформируйте тренировочную выборку 
y_train = label_encoder.fit_transform(y_train)

# трансформируем тестовую выборку
y_test = label_encoder.transform(y_test)

# инициализируем модель дерева решений
model = DecisionTreeClassifier(random_state=RANDOM_STATE)

# создаём словарь с гиперпараметрами
parameters = {
    'min_samples_split': range(2, 11),
    'min_samples_leaf': range(1, 11),
    'max_depth': range(2, 11),
}

# создаём экземпляр объекта для случайного поиска гиперпараметров
rs = RandomizedSearchCV(
    model,
    parameters,
    n_jobs=-1,
    scoring='roc_auc_ovo',
    random_state=RANDOM_STATE
)

# запускаем поиск гиперпараметров
rs.fit(X_train, y_train)

# выводим лучшую метрику и гиперпараметры
print(f'Best score: {rs.best_score_}, best params: {rs.best_params_}')
