#Можно поступить иначе и выделить данные для валидации из тех, которые вы отделили от тренировочных:
from sklearn.model_selection import train_test_split

df_train, df_test_valid = train_test_split(df, test_size=0.4, random_state=42)
df_test, df_valid = train_test_split(df_test_valid, test_size=0.5, random_state=42) 

#Результат:

#    df_train — 60% от исходного датасета,
#    df_test — 20%,
#    df_valid — 20%.

#Сэмплирование и выборки
#Возвращаемся к главной теме. Прежде чем формировать выборки, проверьте, сбалансированы ли они. 
#Если вы найдёте дисбаланс классов — не игнорируйте это и разделите данные, передав параметру stratify значение целевого признака:

df_train, df_test_valid = train_test_split(df, test_size=0.4, \
   random_state=42, \
   stratify=df['target'])

df_test, df_valid = train_test_split(df_test_valid, test_size=0.5, \ 
   random_state=42, \
   stratify=df_test_valid['target'])) 

#Вы также можете сразу отделить целевые и входные признаки во время разделения датасета:

df = pd.read_csv('music_genre_2_classes_imbalanced_v2.csv')
X = df.drop(columns='music_genre')
y = df.music_genre

X_train, X_test_valid, y_train, y_test_valid = train_test_split(X, y, \
  test_size=0.4, \
  random_state=42, \ 
  stratify=y)

X_valid, X_test, y_valid, y_test = train_test_split(X_test_valid, \
  y_test_valid, \
  test_size=0.5, \
  random_state=42, \ 
  stratify=y_test_valid) 
