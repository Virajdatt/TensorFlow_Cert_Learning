import json
from tensorflow import keras

with open('./Data/sarcasm.json') as f:
    datastore = json.load(f)

sentences = []
labels = []
url = []

for item in datastore:
    sentences.append(item['headline'])
    url.append(item['article_link'])
    labels.append(item['is_sarcastic'])

len(sentences) # 26709

tokenizer = keras.preprocessing.text.Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
print(len(tokenizer.word_index)) # 29657
print(sentences[0])
# former versace store clerk sues over secret 'black code' for minority shoppers
sentences_encoded = tokenizer.texts_to_sequences(sentences)
print(sentences_encoded[0])
#[308, 15115, 679, 3337, 2298, 48, 382, 2576, 15116, 6, 2577, 8434]

sequences = keras.preprocessing.sequence.pad_sequences(sentences_encoded)
print(sequences[0])
print(sequences.shape)
# [    0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0   308 15115   679  3337  2298    48   382  2576
#  15116     6  2577  8434]
# (26709, 40)
