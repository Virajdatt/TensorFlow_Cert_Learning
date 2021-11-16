import tensorflow as tf
import tensorflow_datasets as tfds
import io
import numpy as np

imdb, metadata = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

train, test = imdb['train'], imdb['test']

train_sen, train_lab = [], []
test_sen, test_lab = [], []

for sen, lab in train:
    train_sen.append(str(sen.numpy()))
    train_lab.append(lab.numpy())

for sen, lab in test:
    test_sen.append(str(sen.numpy()))
    test_lab.append(lab.numpy())

train_labels = np.array(train_lab)
test_labels = np.array(test_lab)

## HYPER-PARAM:-

NUM_WORDS = 1000
TRUNCATE = 'post'  # 'pre'
PADDING = 'post'  # 'pre
MAX_LEN = 100
EVD = 16
EPOCHS = 10
## PRE-PROCESS
## 1. Fit Tokenizer

imdb_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=NUM_WORDS,
                                                       oov_token='<OOV>')
imdb_tokenizer.fit_on_texts(train_sen)

## 2. Convert text to sequence
train_seq = imdb_tokenizer.texts_to_sequences(train_sen)
test_seq = imdb_tokenizer.texts_to_sequences(test_sen)

## 3. Convert the sequence to padded sequences

train_padded = tf.keras.preprocessing.sequence.pad_sequences(train_seq,
                                                             truncating=TRUNCATE,
                                                             padding=PADDING,
                                                             maxlen=MAX_LEN)

test_padded = tf.keras.preprocessing.sequence.pad_sequences(test_seq,
                                                            truncating=TRUNCATE,
                                                            padding=PADDING,
                                                            maxlen=MAX_LEN)

##                                                      MODELLING PART

dnn_model = tf.keras.Sequential([
                                 tf.keras.layers.Embedding(NUM_WORDS, EVD, input_length=MAX_LEN),
                                 tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(6, activation='relu'),
                                 tf.keras.layers.Dense(1, activation='sigmoid')
])


dnn_model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

dnn_model.fit(train_padded,
              train_labels,
              epochs=EPOCHS,
              validation_data=(test_padded, test_labels)
              )
              #


# Steps to Visualize the embeddings

# Sanity check of things
reverse_word_index = dict([(value, key) for (key, value) in imdb_tokenizer.word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# 1. Get the weights and the embedding layer

embedding_layer = dnn_model.layers[0]
embedding_weights = embedding_layer.get_weights()[0]
print(embedding_weights.shape)  # VOCAB x EVD
# (1000, 16)

# 2. Create the files necessary for the embedding Visualisation

out_v = io.open('/Users/virajdatt/Desktop/TFCertPrep/files/vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('/Users/virajdatt/Desktop/TFCertPrep/files/meta.tsv', 'w', encoding='utf-8')

for word_num in range(1, NUM_WORDS):
    word = reverse_word_index[word_num]
    embedding = embedding_weights[word_num]

    #print(word)
    out_m.write(word+'\n')
    out_v.write('\t'.join([str(x) for x in embedding]) + "\n")
    #print('\t'.join([str(x) for x in embedding]) + "\n")
    #break
out_m.close()
out_v.close()
