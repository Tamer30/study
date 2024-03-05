import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# загружаем нужные классы
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler

# загружаем класс для работы с пропусками
from sklearn.impute import SimpleImputer

# загружаем нужные метрики
from sklearn.metrics import roc_auc_score

# импортируем модель
from sklearn.tree import DecisionTreeClassifier

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

# создайте итоговый пайплайн: подготовка данных и модель
pipe_final = Pipeline(
    [
        ('preprocessor', data_preprocessor),
        ('models', DecisionTreeClassifier(random_state=RANDOM_STATE))
    ]
)

# обучите модель на тренировочной выборке с помощью пайплайна
pipe_final.fit(X_train, y_train)

# рассчитайте метрику ROC-AUC и выведите с округлением до второго знака
y_test_pred = pipe_final.predict(X_test)
print(f'Метрика ROC-AUC на тестовой выборке: {roc_auc_score(y_test, y_test_pred).round(2)}')
