import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# импорт класса логистической регрессии
from sklearn.linear_model import LogisticRegression

# импортируйте класс для сэмплирования данных методом SMOTE
from imblearn.over_sampling import SMOTE

# импорт функций для вычисления ROC-AUC, F1-меры
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, f1_score

# создание константы RANDOM_STATE
RANDOM_STATE = 42

# добавляем данные из CSV-файла в датафрейм pandas, делим датасет на выборки
df = pd.read_csv('music_genre_4_classes_imbalanced.csv')
X = df.drop(columns='music_genre')
y = df.music_genre

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    random_state=RANDOM_STATE,
    stratify=y
)

# выполняем подготовку данных функцией, все операции в скрытом прекоде
X_train, X_test = prepare_data(X_train, X_test)

# создайте сэмплер, чтобы увеличить объекты минорного класса в тренировочной выборке
# не забудьте зафиксировать random_state
# используйте 50 ближайших соседей при сэмплировании

sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=50)
X_train_sampled, y_train_sampled = sampler.fit_resample(X_train, y_train) 

# создайте и обучите модель на тренировочных сэмплированных данных
# выполните предсказание на тестовых данных
# не забудьте зафиксировать random_state
model = LogisticRegression(random_state=RANDOM_STATE)
model.fit(X_train_sampled, y_train_sampled)
y_pred = model.predict(X_test)
probas = model.predict_proba(X_test)

# посчитайте и выведите ROC-AUC с усреднением 'ovo'
roc = roc_auc_score(y_test, probas, multi_class='ovo')
print('ROC-AUC =', round(roc,2))

# посчитайте и выведите F1-меру с усреднением 'macro'
f1 = f1_score(y_test, y_pred, average='macro')
print('F1-score =', round(f1,2))

# постройте матрицу ошибок методом from_estimator()
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.show()

##################################################################

from imblearn.over_sampling import SMOTETomek

sampler = SMOTETomek(random_state=42)
X_resample, y_resample = sampler.fit_resample(X, y) 

###################################################################
#Взвешивание классов

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced') 
