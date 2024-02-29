import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

# импортируйте функцию ConfusionMatrixDisplay(), чтобы построить матрицу ошибок
from sklearn.metrics import ConfusionMatrixDisplay

# импортируйте класс для модели kNN
from sklearn.neighbors import KNeighborsClassifier

# создание константы RANDOM_STATE
RANDOM_STATE = 42

# выполняем подготовку данных функцией, все операции в скрытом прекоде
X_train, X_test, y_train, y_test = prepare_data('music_genre_3_classes_balanced.csv')

# обучите модель kNN на тренировочных данных
# модель должна учитывать 100 ближайших соседей
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# постройте матрицу ошибок для тестовых данных функцией ConfusionMatrixDisplay()
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.show()
