simple_imputer = SimpleImputer(
    missing_values=np.nan,
    strategy='most_frequent'
) 

#missing_values — указываем, что считать пропуском, например np.nan.
#strategy — стратегия заполнения пропусков: среднее mean, медиана median,
#мода most_frequent или константа constant.
#В случае с категориальным признаком логичнее использовать моду — самое частотное значение в выборке.

# импортируем библиотеки и объявляем константы
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42

# подготавливаем данные заранее созданной функцией 
# в код добавили игнорирование неизвестных категорий при кодировании OHE
X_train, X_test, y_train, y_test = prepare_data('railway_full.csv')

# создаём список со строковыми категориями для кодирования OrdinalEncoder
cols_ordinal = [
    'Оценка комфортности покупки билета онлайн', 'Оценка качества wifi'
]

# создаём экземпляр класса OrdinalEncoder для кодирования
# кодируем неизвестные категории пропусками
ordinal_encoder = OrdinalEncoder(
    handle_unknown='use_encoded_value',
    unknown_value=np.nan
)

# обучаем OrdinalEncoder на выбранных признаках из тренировочной выборки
# преобразовываем тренировочную выборку
X_train[cols_ordinal] = ordinal_encoder.fit_transform(X_train[cols_ordinal])

# преобразовываем тестовую выборку
X_test[cols_ordinal] = ordinal_encoder.transform(X_test[cols_ordinal])

# создайте экземпляр класса для заполнения пропусков на моду
imputer = SimpleImputer(
    missing_values=np.nan,
    strategy='most_frequent'
) 

# посчитайте моду на тренировочной выборке
imputer.fit(X_train) 

# заполните пропуск в тестовых данных на моду
X_test[imputer.feature_names_in_] = imputer.transform(X_test[imputer.feature_names_in_])

# выводим пропуски в данных
print(X_test.isna().sum())
