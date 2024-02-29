import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# импортируйте функцию для расчёта fpr, tpr, threshold
from sklearn.metrics import roc_curve

# импортируйте класс для модели классификации kNN 
from sklearn.neighbors import KNeighborsClassifier

# создание константы RANDOM_STATE
RANDOM_STATE = 42

# выполняем подготовку данных функцией, все операции в скрытом прекоде
X_train, X_test, y_train, y_test = prepare_data('music_genre_2_classes_balanced_v2_exp.csv')

# создайте и обучите модель kNN на тренировочных данных 
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# выполните предсказания на тестовых данных
preds = model.predict_proba(X_test)

# рассчитайте значения FPR и TRP
fpr, tpr, threshold = roc_curve(y_test, preds[:,1], pos_label='Jazz')

# постройте график зависимости FPR от TPR
plt.plot(fpr, tpr)
plt.title("График зависимости FPR от TPR")
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.show()
