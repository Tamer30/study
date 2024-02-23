# импорт необходимых библиотек
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

scaler = StandardScaler()

# объявляем константу random_state
RANDOM_STATE = 77

# объявляем переменную с данными
# выделяем целевой и входные признаки
data = pd.read_csv('orders_seafood.csv')
X = data.drop(columns=['target', 'client_id'])
y = data['target']
columns = ['confirm_count','summ_']

# делим данные на тренировочные и валидационные
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
scaler = StandardScaler()

# инициализируйте модель SVM c ядром полинома степени 6
model = SVC(kernel='poly', degree = 6)

# передайте признаки в тренировочную выборку и стандартизируйте их
X_new_train = X_train[columns]
X_new_scalled = scaler.fit_transform(X_new_train)

# обучите модель
model.fit(X_new_scalled, y_train)

# постройте разделяющую границу между классами
sns.set_style(style='white')
plot_decision_regions(X_new_scalled, y_train.to_numpy(), clf=model, legend=2)
plt.xlabel(columns[1])
plt.ylabel(columns[0])
plt.show();
