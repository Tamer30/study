import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# загружаем необходимые инструменты
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# загружаем классы для преобразования данных
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder, 
    StandardScaler, 
    MinMaxScaler,
    RobustScaler
)
 
# загружаем инструмент для автоподбора гиперпараметров
from sklearn.model_selection import GridSearchCV

# загружаем класс для работы с пропусками
from sklearn.impute import SimpleImputer

# загружаем нужные метрики
from sklearn.metrics import roc_auc_score

# импортируем модель
from sklearn.tree import DecisionTreeClassifier

# загрузите нужные модели
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

RANDOM_STATE = 42
TEST_SIZE = 0.25

# загружаем данные
df_full = pd.read_csv('railway_full.csv')

X_train, X_test, y_train, y_test = train_test_split(
    df_full.drop(
        [
            'Удовлетворён предоставленной услугой',
            'Общая оценка качества предоставленной услуги'
        ], 
        axis=1
    ),
    df_full['Удовлетворён предоставленной услугой'],
    test_size = TEST_SIZE, 
    random_state = RANDOM_STATE,
    stratify = df_full['Удовлетворён предоставленной услугой']
)

# создаём списки с названиями признаков
ohe_columns = [
    'Пол', 'Путешествует с детьми', 'Путешествует по работе', 
    'Тип', 'Оценка качества питания'
]
ord_columns = [
    'Оценка комфортности покупки билета онлайн', 'Оценка качества wifi', 
    'Оценка комфортности времени отправления/прибытия'
]
num_columns = ['Возраст', 'Расстояние']

# создаём пайплайн для подготовки признаков из списка ohe_columns: заполнение пропусков и OHE-кодирование
# SimpleImputer + OHE
ohe_pipe = Pipeline(
    [
        (
            'simpleImputer_ohe', 
            SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        ),
        (
            'ohe', 
            OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False)
        )
    ]
)

# cоздаём пайплайн для подготовки признаков из списка ord_columns: заполнение пропусков и Ordinal-кодирование
# SimpleImputer + OE
ord_pipe = Pipeline(
    [
        (
            'simpleImputer_before_ord', 
            SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        ),
        (
            'ord',
            OrdinalEncoder(
                categories=[
                    ['нормально', 'хорошо', 'плохо', 'отсутствует'],
                    ['нормально', 'хорошо', 'плохо', 'отсутствует'],
                    ['нормально', 'хорошо', 'плохо'],
                ], 
                handle_unknown='use_encoded_value',
                unknown_value=np.nan
            )
        ),
        (
            'simpleImputer_after_ord', 
            SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        )
    ]
)

# создаём общий пайплайн для подготовки данных
data_preprocessor = ColumnTransformer(
    [
        ('ohe', ohe_pipe, ohe_columns),
        ('ord', ord_pipe, ord_columns),
        ('num', MinMaxScaler(), num_columns)
    ], 
    remainder='passthrough'
)

# создаём итоговый пайплайн: подготовка данных и модель
pipe_final= Pipeline(
    [
        ('preprocessor', data_preprocessor),
        ('models', DecisionTreeClassifier(random_state=RANDOM_STATE))
    ]
)

param_grid = {
    'preprocessor__num': [
        StandardScaler(), 
        MinMaxScaler(), 
        RobustScaler(), 
        'passthrough'
    ],
    # напишите название шага и укажите модели
    'models': [DecisionTreeClassifier(random_state=RANDOM_STATE),
              KNeighborsClassifier(),
              SVC(),
              LogisticRegression(random_state=RANDOM_STATE)]
}

grid = GridSearchCV(
    pipe_final, 
    param_grid=param_grid, 
    cv=5,
    # задайте метрику ROC-AUC
    scoring='roc_auc', 
    n_jobs=-1
)
grid.fit(X_train, y_train)

print('Лучшая модель и её параметры:\n\n', grid.best_estimator_)
