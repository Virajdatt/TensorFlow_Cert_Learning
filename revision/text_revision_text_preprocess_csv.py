import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet
from nltk.corpus import stopwords
import re
import warnings
import tensorflow as tf
warnings.filterwarnings('ignore')

## Read the csv data

bbc_data = pd.read_csv('/Users/virajdatt/Desktop/TFCertPrep/data/bbc-text.csv')

#chopped_data = bbc_data[1:5]


## Cleanup Function

def clean_up(review):
    clean = re.sub("[^a-zA-Z]", " ", review)
    # replace multiple space by a single
    clean = re.sub(' +', ' ', clean)

    word_tokens = clean.lower().split()

    # 4. Remove stopwords
    le = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    stop_words.add("co")
    stop_words.add("http")
    word_tokens = [le.lemmatize(w) for w in word_tokens if not w in stop_words]

    cleaned = " ".join(word_tokens)
    # re.sub(' +', ' ',string4)
    return cleaned

## Run the clean-up


bbc_data['cleaned'] = bbc_data['text'].apply(clean_up)

bbc_data['category_encoded'] = pd.Categorical(pd.factorize(bbc_data['category'])[0] + 1)

## splitting data
train_size = int(0.9 * len(bbc_data))
train_data = bbc_data[:train_size]
test_data = bbc_data[train_size:]


train_sentences = train_data['cleaned'].values
test_sentences = test_data['cleaned'].values

train_labels = np.array(train_data['category_encoded'].values)
test_labels = np.array(test_data['category_encoded'].values)


## HYPER-PARAM:-

NUM_WORDS = 1000
TRUNCATE = 'post'  # 'pre'
PADDING = 'post'   # 'pre
MAX_LEN = 100
EVD = 16

## 1. Fit Tokenizer

bbc_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=NUM_WORDS,
                                                      oov_token='<OOV>')
bbc_tokenizer.fit_on_texts(train_sentences)

## 2. Convert text to sequence

train_seq = bbc_tokenizer.texts_to_sequences(train_sentences)
test_seq = bbc_tokenizer.texts_to_sequences(test_sentences)

## 3. Convert the sequence to padded sequences

train_padded = tf.keras.preprocessing.sequence.pad_sequences(train_seq,
                                                             truncating=TRUNCATE,
                                                             padding=PADDING,
                                                             maxlen=MAX_LEN)
test_padded = tf.keras.preprocessing.sequence.pad_sequences(test_seq,
                                                             truncating=TRUNCATE,
                                                             padding=PADDING,
                                                             maxlen=MAX_LEN)


## Modelling



lstm_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(NUM_WORDS, EVD, input_length=MAX_LEN),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, )),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

lstm_model.compile(loss='sparse_categorical_crossentropy',
                   optimizer='adam',
                   metrics='accuracy')


lstm_model.fit(train_padded,
              train_labels,
              epochs=20,
               validation_data=(test_padded, test_labels)
              )

# Epoch 20 - loss: 0.8749 - accuracy: 0.3961 - val_loss: 1.0327 - val_accuracy: 0.3857

# Model 2: CNN

cnn_model = tf.keras.Sequential([

    tf.keras.layers.Embedding(NUM_WORDS, EVD, input_length=MAX_LEN),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

cnn_model.compile(loss='sparse_categorical_crossentropy',
                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, ),
                   metrics='accuracy')

cnn_model.fit(train_padded,
              train_labels,
              epochs=10,
               validation_data=(test_padded, test_labels)
              )

# Epoch 30 loss: 155.0502 - accuracy: 0.2143 - val_loss: 244.7632 - val_accuracy: 0.2377

gru_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(NUM_WORDS, EVD, input_length=MAX_LEN),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])
gru_model.compile(loss='sparse_categorical_crossentropy',
                   optimizer='adam',
                   metrics='accuracy')


gru_model.fit(train_padded,
              train_labels,
              epochs=10,
               validation_data=(test_padded, test_labels)
              )

# Epoch 10 loss: 0.7055 - accuracy: 0.5789 - val_loss: 0.7488 - val_accuracy: 0.6009
lstm_model2 = tf.keras.Sequential([
    tf.keras.layers.Embedding(NUM_WORDS, EVD, input_length=MAX_LEN),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, )),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

lstm_model2.compile(loss='sparse_categorical_crossentropy',
                   optimizer='adam',
                   metrics='accuracy')


lstm_model2.fit(train_padded,
              train_labels,
              epochs=20,
               validation_data=(test_padded, test_labels)
              )
# Epoch 20 loss: 0.7444 - accuracy: 0.4530 - val_loss: 0.8152 - val_accuracy: 0.4036
