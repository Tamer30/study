import torch
import transformers
import pandas as pd

# инициализируем токенизатор
tokenizer = transformers.BertTokenizer(
    vocab_file='/datasets/ds_bert/vocab.txt')
df_tweets = pd.read_csv('/datasets/tweets.csv')
# токенизируем текст
#vector = tokenizer.encode('Очень удобно использовать уже готовый трансформатор текста', add_special_tokens=True)
tokenized = df_tweets['text'].apply(
  lambda x: tokenizer.encode(x, add_special_tokens=True))

# применим padding к векторам
n = 280
# англ. вектор с отступами
padded = tokenized + [0]*(n - len(tokenized))

# создадим маску для важных токенов
attention_mask = np.where(padded != 0, 1, 0)
