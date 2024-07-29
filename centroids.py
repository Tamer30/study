import pandas as pd
import warnings
from sklearn.cluster import KMeans

# Игнорируем предупреждения
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Загрузка данных
data = pd.read_csv('/datasets/segments.csv')

# Создание модели k-средних для трёх кластеров с заданным random_state
model = KMeans(n_clusters=3, random_state=12345)

# Обучение модели на данных
model.fit(data)

# Вывод значений центроидов кластеров
print("Центроиды кластеров:")
print(model.cluster_centers_)
