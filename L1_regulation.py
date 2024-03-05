import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest

RANDOM_STATE = 77
scaler = StandardScaler()

test_data = pd.read_csv('orders_seafood_new.csv')
data = pd.read_csv('orders_seafood_test.csv')

# объединяем исходный и новый датасеты
all_data = pd.concat([data,test_data],axis=0)
X = all_data.drop(columns=['target', 'client_id'])
y = all_data['target']

# делим данные на выборки
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)

# добавляем полиномиальные признаки
poly = PolynomialFeatures(2)

X_train = poly.fit_transform(X_train)
X_test = poly.transform(X_test)

# сохранение таблицы X_test с полиномиальными признаками до стандартизации
X_test_df = pd.DataFrame(X_test,columns = poly.get_feature_names())

# стандартизируем признаки
X_train_scalled = pd.DataFrame(scaler.fit_transform(X_train),columns = poly.get_feature_names())
X_test_scalled =  pd.DataFrame(scaler.transform(X_test),columns = poly.get_feature_names())

# запускаем цикл, который обучит несколько моделей с разной силой регуляризации
# итоговая выдача — таблица с информацией о C, точности и сэкономленных деньгах
C_ = [0.1,0.4,0.5,0.6,0.7,0.8,1,2,5,10]
for c in C_:
    model = LogisticRegression(random_state=RANDOM_STATE, penalty='l1' , solver='saga', C=c)
    model.fit(X_train_scalled, y_train)

    acc1 = accuracy_score(y_test, model.predict(X_test_scalled))

    predicts = X_test_df[['x3']].copy()
    predicts['logreg'] = model.predict(X_test_scalled)
    predicts['y_test'] = y_test.tolist()

    TP = predicts[(predicts['logreg']==1)&(predicts['y_test']==1)]['x3'].sum()*0.7*0.8
    FP = predicts[(predicts['logreg']==1)&(predicts['y_test']==0)]['x3'].sum()*0.2
    
    print(c, acc1, TP-FP)
