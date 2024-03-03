# импортируем нужные библиотеки и объявляем константы
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

TEST_SIZE = 0.2
RANDOM_STATE = 42

# считываем данные из CSV-файла
df = pd.read_csv('progulka.csv')

# делим данные на входные и целевые
X = df.drop('гулять?', axis=1)
y = df['гулять?']

# разделите данные на тренировочные и тестовые
# используйте константы с размером тестовой выборки и random_state
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=RANDOM_STATE,
    test_size=TEST_SIZE)

# инициализируйте модель дерева решений и обучите её на тренировочных данных
model = DecisionTreeClassifier(random_state=RANDOM_STATE)
model.fit(X_train, y_train)

# визуализируйте схему дерева решений
plot_tree(decision_tree=model);
