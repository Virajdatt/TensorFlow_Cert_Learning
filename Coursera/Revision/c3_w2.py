import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np

# 1. Load data from tfds

imdb, info = tfds.load('imdb_reviews', as_supervised=True, with_info=True)

# 2. Prepare Data

train, test = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

for s, l in train:
    training_sentences.append(s.numpy().decode('utf8'))
    training_labels.append(l.numpy())

for s, l in test:
    testing_sentences.append(s.numpy().decode('utf8'))
    testing_labels.append(l.numpy())

training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

# 3. Tokenizing and padding

vocab_size = 1000
embedding_dim = 32
max_len = 200
trun_type = 'post'
oov_tok = '<OOV>'

tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size,
                                               oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

sequences = tokenizer.texts_to_sequences(training_sentences)
padded_seq = keras.preprocessing.sequence.pad_sequences(sequences,
                                                        maxlen=max_len,
                                                        truncating=trun_type)

test_sequences = tokenizer.texts_to_sequences(testing_sentences)
test_padded_seq = keras.preprocessing.sequence.pad_sequences(test_sequences,
                                                             maxlen=max_len,
                                                             truncating=trun_type)

tokenizer.index_word.get(12)
sequences[0]


def decode_padded(sentence):
    tokens = 0
    for i in sentence:
        tokens = tokens +1
        print(tokenizer.index_word.get(i), end=' ')
    print('\nNo of tokens =',tokens)

print(decode_padded(sequences[0]))
print(decode_padded(padded_seq[0]))

# 4. Model Building and Fitting
model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len ),
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=[keras.metrics.binary_accuracy, 'accuracy']
)

history = model.fit(
    padded_seq, training_labels_final,
    validation_data=(test_padded_seq, testing_labels_final),
    epochs=10
)

print(type(training_labels),
      type(training_labels_final)
      )

# 5.Embedding Layer Viz

embedding_layer = model.layers[0]
#dir(embedding_layer)
embedding_weights = embedding_layer.get_weights()[0]

import io

out_v = io.open('./files/vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('./files/meta_data.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
    word = tokenizer.index_word.get(word_num)
    embeddings = embedding_weights[word_num]
    #print(word)
    #print(embeddings.shape)
    #break
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()
