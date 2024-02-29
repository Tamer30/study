import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# импортируйте метрики для расчёта взвешенной и обычной F1
from sklearn.metrics import f1_score, fbeta_score 

# импортируем класс для модели классификации kNN 
from sklearn.neighbors import KNeighborsClassifier

# создание константы RANDOM_STATE
RANDOM_STATE = 42

# выполняем подготовку данных функцией, все операции в скрытом прекоде
X_train, X_test, y_train, y_test = prepare_data('music_genre_2_classes_balanced_v2_exp.csv')

# создаём и обучаем модель kNN на тренировочных данных 
model = KNeighborsClassifier()
model.fit(X_train, y_train)
preds = model.predict(X_test)

# рассчитайте и выведите на экран F1
print(f"F1 = {f1_score(y_test, preds, pos_label='Jazz'):.2f}")

# рассчитайте и выведите на экран F1-бета, используйте коэффициент beta=10
print(f"F1_beta = {fbeta_score(y_test, preds, pos_label='Jazz', beta=10):.2f}")
