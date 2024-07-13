import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
# импортируйте CountVectorizer

data = pd.read_csv("/datasets/tweets_lemm.csv")
corpus = list(data['lemm_text'])
count_vect = CountVectorizer(ngram_range=(2, 2))

# создайте n-грамму n_gramm, для которой n=2
n_gramm = count_vect.fit_transform(corpus) 
print("Размер:", n_gramm.shape)
