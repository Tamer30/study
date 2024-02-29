import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# импортируйте функции для вычисления accuracy, ROC-AUC, F1-меры
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# импортируйте класс для дамми-модели 
from sklearn.dummy import DummyClassifier

# создание константы RANDOM_STATE
RANDOM_STATE = 42

# добавляем данные из CSV-файла в датафрейм pandas, делим датасет на выборки
df = pd.read_csv('music_genre_2_classes_imbalanced_v2.csv')
X = df.drop(columns='music_genre')
y = df.music_genre

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    random_state=RANDOM_STATE, stratify=y
)

# подготовка данных (кодирование и масштабирование)
# все операции в скрытом прекоде
X_train, X_test = prepare_data(X_train, X_test)

# создайте и обучите DummyClassifier на тренировочных данных
# выполните предсказание на тестовой выборке
dummy_model = DummyClassifier(random_state=RANDOM_STATE)
dummy_model.fit(X_train, y_train)
dummy_model_preds = dummy_model.predict(X_test)
dummy_model_probas = dummy_model.predict_proba(X_test)[:,1]

# посчитайте и выведите метрику accuracy
dummy_acc = accuracy_score(y_test, dummy_model_preds)
print('Dummy Accuracy =', round(dummy_acc,2))

# посчитайте и выведите метрику ROC-AUC
dummy_roc = roc_auc_score(y_test, dummy_model_probas)
print('ROC-AUC =', round(dummy_roc,2))

# посчитайте и выведите F1-меру
dummy_f1 = f1_score(y_test, dummy_model_preds, pos_label='Rock')
print('F1-score =', round(dummy_f1,2))
