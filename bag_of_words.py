import pandas as pd
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer
# < напишите код здесь >

data = pd.read_csv("/datasets/tweets_lemm.csv")
corpus = data['lemm_text'].values.astype('U')
count_vect = CountVectorizer()

# создайте мешок слов без учёта стоп-слов
bow = count_vect.fit_transform(corpus) 
# < напишите код здесь >

print("Размер мешка без учёта стоп-слов:", bow.shape)

# создайте новый мешок слов с учётом стоп-слов
stop_words = set(stopwords.words('russian'))
count_vect = CountVectorizer(stop_words=stop_words) 
bow = count_vect.fit_transform(corpus) 
# < напишите код здесь >

print("Размер мешка с учётом стоп-слов:", bow.shape)
