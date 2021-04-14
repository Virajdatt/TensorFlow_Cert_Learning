import json
from tensorflow import keras
import numpy as np
with open('./Data/sarcasm.json') as f:
    datastore = json.load(f)

sentences = []
labels = []
url = []

for item in datastore:
    sentences.append(item['headline'])
    url.append(item['article_link'])
    labels.append(item['is_sarcastic'])

labels = np.array(labels)

# Hyper parameters

VOCAB = 10000
EVD = 16
MAX_LEN = 40
EPOCHS = 10
OOV_TOKEN = '<OOV>'
TRUNC_TYPE = 'post'
TEST_SPLIT = 16709
tokenizier = keras.preprocessing.text.Tokenizer(num_words=VOCAB,
                                                oov_token=OOV_TOKEN)
tokenizier.fit_on_texts(sentences)
word_index = tokenizier.word_index
sequnce = tokenizier.texts_to_sequences(sentences)


padded_seq = keras.preprocessing.sequence.pad_sequences(sequnce,
                                                        maxlen=MAX_LEN,
                                                        truncating=TRUNC_TYPE)
print(padded_seq.shape)
# (26709, 40)

train_padded = padded_seq[:TEST_SPLIT]
testing_padded = padded_seq[TEST_SPLIT:]
# print(f'Train len: {len(train_padded)},'
#       f'Testing len: {len(testing_padded)}')
# Train len: 16709,Testing len: 10000
training_labels = labels[:TEST_SPLIT]
testing_labels = labels[TEST_SPLIT:]

# Sanity check of things
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# Model 1 : LSTM

lstm_model = keras.Sequential([
    keras.layers.Embedding(VOCAB, EVD, input_length=MAX_LEN),
    keras.layers.Bidirectional(keras.layers.LSTM(32, )),
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

lstm_model.compile(loss='binary_crossentropy',
                   optimizer='adam',
                   metrics='accuracy')

LSCB = keras.callbacks.TensorBoard('./files/sarcasm_lstm_1')
lstm_model.fit(train_padded,
               training_labels,
               epochs=EPOCHS,
               validation_data=(testing_padded, testing_labels),
               callbacks=[LSCB]
               )
# Model 2: CNN
cnn_model = keras.Sequential([
    keras.layers.Embedding(VOCAB, EVD, input_length=MAX_LEN),
    keras.layers.Conv1D(128, 5, activation='relu'),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

cnn_model.compile(loss='binary_crossentropy',
                   optimizer='adam',
                   metrics='accuracy')

CNCB = keras.callbacks.TensorBoard('./files/sarcasm_cnn_1')
cnn_model.fit(train_padded,
               training_labels,
               epochs=EPOCHS,
               validation_data=(testing_padded, testing_labels),
               callbacks=[CNCB]
               )

# Model 3: GRU
gru_model = keras.Sequential([
    keras.layers.Embedding(VOCAB, EVD, input_length=MAX_LEN),
    keras.layers.Bidirectional(keras.layers.GRU(32)),
    keras.layers.Flatten(),
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
gru_model.compile(loss='binary_crossentropy',
                   optimizer='adam',
                   metrics='accuracy')

GRUCB = keras.callbacks.TensorBoard('./files/sarcasm_gru_1')
gru_model.fit(train_padded,
               training_labels,
               epochs=EPOCHS,
               validation_data=(testing_padded, testing_labels),
               callbacks=[GRUCB]
               )

