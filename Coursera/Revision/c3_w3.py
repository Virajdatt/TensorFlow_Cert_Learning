import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

with open('./Data/sarcasm.json') as file:
    data = json.load(file)

headlines = []
labels = []
for i in data:
    headlines.append(i['headline'])
    labels.append(i['is_sarcastic'])


# Hyperparameters

vocab_size = 1000
oov_token = '<OOV>'
embedding_dim = 16
maxlen = 120
padding = 'post'
truncate = 'post'
ts = 0.8
train_size = int(ts * len(headlines))


train_headlines = headlines[:train_size]
train_rlabels = labels[:train_size]
test_headlines = headlines[train_size:]
test_rlabels = labels[train_size:]

train_labels = np.array(train_rlabels)
test_labels = np.array(test_rlabels)

tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size,
                                               oov_token=oov_token,
                                               lower=True)
tokenizer.fit_on_texts(train_headlines)

train_seq = tokenizer.texts_to_sequences(train_headlines)
train_padded = keras.preprocessing.sequence.pad_sequences(train_seq,
                                                          truncating=truncate,
                                                          padding=padding,
                                                          maxlen=maxlen)

test_seq = tokenizer.texts_to_sequences(test_headlines)
test_padded = keras.preprocessing.sequence.pad_sequences(test_seq,
                                                          truncating=truncate,
                                                          padding=padding,
                                                          maxlen=maxlen)

print(train_padded.shape, train_labels.shape)
print(test_padded.shape, test_labels.shape)


model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=maxlen),
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(32)),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer='adam')

model.fit(train_padded, train_labels,
          validation_data=(test_padded, test_labels),
          epochs=10)


model2 = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=maxlen),
    #keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
    #keras.layers.Bidirectional(keras.layers.LSTM(32)),
    keras.layers.Conv1D(128, 5, activation='relu'),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model2.summary()

model2.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer='adam')

model2.fit(train_padded, train_labels,
          validation_data=(test_padded, test_labels),
          epochs=10)


model3 = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=maxlen),
    keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.GRU(32)),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model3.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer='adam')

model3.fit(train_padded, train_labels,
          validation_data=(test_padded, test_labels),
          epochs=10)

print('LSTM evaluated', model.evaluate(test_padded, test_labels, verbose=0)[-1])
print('CNN evaluated', model2.evaluate(test_padded, test_labels, verbose=0)[-1])
print('GRU evaluated', model3.evaluate(test_padded, test_labels, verbose=0)[-1])