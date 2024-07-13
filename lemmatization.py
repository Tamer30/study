import pandas as pd
from pymystem3 import Mystem
import re # < напишите код здесь >

data = pd.read_csv('/datasets/tweets.csv')
corpus = list(data['text'])


def lemmatize(text):
    m = Mystem()
    lemm_list = m.lemmatize(text)
    lemm_text = "".join(lemm_list)
        
    return lemm_text


def clear_text(text):
    temp = re.sub(r'[^а-яА-ЯёЁ ]', ' ', text)
    temp = " ".join(temp.split())
    return(temp)

print("Исходный текст:", corpus[0])
print("Очищенный и лемматизированный текст:", lemmatize(clear_text(corpus[0])))
