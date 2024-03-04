from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
example_train['ordinal'] = ordinal_encoder.fit_transform(example_train)
print(example_train) 

###########################################################################

Порядок категорий можно увидеть с помощью атрибута categories_:

print(ordinal_encoder.categories_) 
[array(['нормально', 'отсутствует', 'плохо', 'хорошо'], dtype=object)] 

############################################################################

# импортируем библиотеки и объявляем константы
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

RANDOM_STATE = 42

# подготавливаем данные заранее созданной функцией 
# в код добавили игнорирование неизвестных категорий при кодировании OHE
X_train, X_test, y_train, y_test = prepare_data('railway_full.csv')

# создаём список со строковыми категориями для кодирования OrdinalEncoder
cols_ordinal = [
    'Оценка комфортности покупки билета онлайн', 'Оценка качества wifi'
]

# создайте экземпляр класса OrdinalEncoder для кодирования
# закодируйте неизвестные категории пропусками
ordinal_encoder = OrdinalEncoder(
    handle_unknown='use_encoded_value',
    unknown_value=np.nan
) 

# обучаем OrdinalEncoder на выбранных признаках из тренировочной выборки
# преобразовываем тренировочную выборку
X_train[cols_ordinal] = ordinal_encoder.fit_transform(X_train[cols_ordinal])

# преобразуйте тестовую выборку
X_test[cols_ordinal] = ordinal_encoder.transform(X_test[cols_ordinal])

# выводим пропуски в данных
print(X_test.isna().sum())
