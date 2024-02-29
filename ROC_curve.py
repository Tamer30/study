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

########################################################################################################
########################################################################################################
########################################################################################################
#Задача 2
#Загрузите датасет для классификации музыкальных жанров, проведите подготовку данных и обучите модель kNN.
#Классифицируйте данные их тестовой выборки, постройте график ROC-кривой с помощью RocCurveDisplay() и вычислите ROC-AUC.

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler

# импортируйте функции для графика ROC-кривой и расчёта ROC-AUC
from sklearn.metrics import roc_auc_score, RocCurveDisplay


# импортируем класс для модели классификации kNN 
from sklearn.neighbors import KNeighborsClassifier

# создание константы RANDOM_STATE
RANDOM_STATE = 42

# выполняем подготовку данных функцией, все операции в скрытом прекоде
X_train, X_test, y_train, y_test = prepare_data('music_genre_2_classes_balanced_v2_exp.csv')

# создаём и обучаем модель kNN на тренировочных данных
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# получите оценки вероятностей на тестовой выборке
preds = model.predict_proba(X_test)

# вычислите метрику ROC-AUC и выведите её на экран командой print()
roc_auc = roc_auc_score(y_test, preds[:, 1])
print('ROC-AUC = ', roc_auc.round(2))

# выведите на экран ROC-кривую методом from_estimator()
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.plot([0,1], [0,1], linestyle='dashed', label='Random prediction')
plt.title("График ROC-AUC")
plt.legend()
plt.show()
