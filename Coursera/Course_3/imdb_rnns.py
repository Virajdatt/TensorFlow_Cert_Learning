from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np

imdb, info = tfds.load('imdb_reviews', with_info=True,
                       as_supervised=True)

raw_train, raw_test = imdb['train'], imdb['test']

# Preparing the datasets for pre-processing
# Essentially converting the tf.data.Dataset objects to plain list or np.array
training_seneteces, training_labels = [], []
testing_seneteces, testing_labels = [], []

for sen, lab in raw_train:
    training_seneteces.append(str(sen.numpy()))
    training_labels.append(lab.numpy())
    # break

for sen, lab in raw_test:
    testing_seneteces.append(str(sen.numpy()))
    testing_labels.append(lab.numpy())
    # break

training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)

# Hyperparameter defining zone

VOCAB = 10000  # Used in (Tokenizer and keras embedding layer)
EVD = 16  # Embedding Vector Dimensions # Used in keras embedding layer
OOV_TOKEN = '<OOV>'  # Used in Tokenizer
MAX_LEN = 100  # Used in (pad_sequences and keras embedding layer)
TRUNC_TYPE = 'post'  # Used in pad_sequences
EPOCHS = 10  # Used in keras fit

# Preprocessing the train and test texts to convert strings to sequences

tokenizer = keras.preprocessing.text.Tokenizer(num_words=VOCAB, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(training_seneteces)
word_index = tokenizer.word_index

train_seq = tokenizer.texts_to_sequences(training_seneteces)
train_padded = keras.preprocessing.sequence.pad_sequences(train_seq, maxlen=MAX_LEN,
                                                          truncating=TRUNC_TYPE)

testing_seq = tokenizer.texts_to_sequences(testing_seneteces)
testing_padded = keras.preprocessing.sequence.pad_sequences(testing_seq, maxlen=MAX_LEN,
                                                            truncating=TRUNC_TYPE)

# Sanity check of things
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(f'Decoded Review: \n{decode_review(train_padded[0])}')
print(f'Original Review: \n{training_seneteces[0]}')
# Modeling part

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

LSCB = keras.callbacks.TensorBoard('./files/imdb_lstm_1')
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

CNCB = keras.callbacks.TensorBoard('./files/imdb_cnn_1')
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

GRUCB = keras.callbacks.TensorBoard('./files/imdb_gru_1')
gru_model.fit(train_padded,
               training_labels,
               epochs=EPOCHS,
               validation_data=(testing_padded, testing_labels),
               callbacks=[GRUCB]
               )
