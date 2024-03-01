from sklearn.model_selection import train_test_split

# импортируйте из библиотеки imblearn класс RandomOverSamper
from imblearn.over_sampling import RandomOverSampler

# создание константы RANDOM_STATE
RANDOM_STATE = 42

# загрузка данных из CSV-файла в датафрейм pandas
df = pd.read_csv('music_genre_2_classes_imbalanced_v2.csv')
X = df.drop(columns='music_genre')
y = df.music_genre

# делим датасет на тренировочную и тестовую выборки со стратификацией
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    random_state=RANDOM_STATE,
    stratify=y
)

# создайте экземпляр класса RandomOverSampler с фиксированным random_state
sampler = RandomOverSampler(random_state=RANDOM_STATE)

# сэмплируйте данные методов оверсэмплинга одной командой
X_train_sampled, y_train_sampled = sampler.fit_resample(X_train, y_train) 

# выведите на экран распределения классов в тренировочном датасете до и после сэмплирования
print(f'Тренировочные данные до сэмплирования:\n{X_train.value_counts(), y_train.value_counts()}')
print(f'\nТренировочные данные после сэмплирования:\n{X_train_sampled.value_counts(), y_train_sampled.value_counts()}')
