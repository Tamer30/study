#Можно поступить иначе и выделить данные для валидации из тех, которые вы отделили от тренировочных:
from sklearn.model_selection import train_test_split

df_train, df_test_valid = train_test_split(df, test_size=0.4, random_state=42)
df_test, df_valid = train_test_split(df_test_valid, test_size=0.5, random_state=42) 

#Результат:

#    df_train — 60% от исходного датасета,
#    df_test — 20%,
#    df_valid — 20%.
