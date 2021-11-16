import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet
from nltk.corpus import stopwords
import numpy as np


data = pd.read_csv('/Users/virajdatt/Desktop/TFCertPrep/data/training_cleaned.csv', header=None)

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


data['cleaned'] = data[5].apply(clean_up)

data[0] = pd.to_numeric(data[0])


len(data)
## splitting data
train_size = int(0.9 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]


train_sentences = train_data['cleaned'].values
test_sentences = test_data['cleaned'].values

train_labels = np.array(train_data[0].values)
test_labels = np.array(test_data[0].values)


## HYPER-PARAM:-

#NUM_WORDS = 1000
TRUNCATE = 'post'  # 'pre'
PADDING = 'post'   # 'pre
MAX_LEN = 16
EVD = 100


## Tokenizer

tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(train_sentences)

sequences = tokenizer.texts_to_sequences(train_sentences)
test_sequences = tokenizer.texts_to_sequences(test_sentences)

padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequences,
                                                          maxlen=MAX_LEN,
                                                          padding=PADDING,
                                                          truncating=TRUNCATE)

test_padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(test_sequences,
                                                          maxlen=MAX_LEN,
                                                          padding=PADDING,
                                                          truncating=TRUNCATE)


vocab_size=len(tokenizer.word_index)


embeddings_index = {}
with open('/Users/virajdatt/Desktop/TFCertPrep/files/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split();
        word = values[0];
        coefs = np.asarray(values[1:], dtype='float32');
        embeddings_index[word] = coefs;

embeddings_matrix = np.zeros((vocab_size+1, EVD));
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word);
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector;


print(len(embeddings_matrix))


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, EVD, input_length=MAX_LEN, weights=[embeddings_matrix], trainable=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 2
history = model.fit(padded_sequence, train_labels, epochs=num_epochs, validation_data=(test_padded_sequence, test_labels), verbose=2)

print("Training Complete")