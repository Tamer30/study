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
plot_tree(decision_tree=model, filled=True, feature_names=df.columns);


##############################################################################################

Инициализируем модель дерева решений с базовыми настройками:

DecisionTreeClassifier(
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=None,
    max_leaf_nodes=None
) 

Все указанные здесь значения — это гиперпараметры. Их можно менять при инициализации модели,
за исключением одного знакомого вам random_state — он фиксируется один раз, и больше его трогать не стоит.
Гиперпараметр max_depth ограничивает максимальную глубину дерева, а max_leaf_nodes — ограничивает число листьев.
