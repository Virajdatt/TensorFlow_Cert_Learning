import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

raw_data = pd.read_csv('./Data/bbc-text.csv')

stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]


def clean_bbc(text):

    words = text.split()
    words = [w for w in words if w not in stopwords]
    words = ' '.join(words)
    return words

raw_data['cleaned'] = raw_data['text'].apply(clean_bbc)
raw_labels = raw_data['category'].values
raw_sentences = raw_data['cleaned'].values


# Hyper-Parameteres

vocab_size = 1000
train_size = 0.85
oov_token = '<OOV>'
lower = 'True'
maxlen = 120
padding = 'post'
truncating = 'post'
embedding_dim = 16

train_raw_sentences = raw_sentences[:int(len(raw_sentences) * train_size)]
train_rlabels = raw_labels[:int(len(raw_sentences) * train_size)]
test_raw_sentences = raw_sentences[int(len(raw_sentences) * train_size):]
test_rlabels = raw_labels[int(len(raw_sentences) * train_size):]

tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size,
                                               lower=lower,
                                               oov_token=oov_token)
tokenizer.fit_on_texts(train_raw_sentences)

train_sen = tokenizer.texts_to_sequences(train_raw_sentences)
train_padded = keras.preprocessing.sequence.pad_sequences(train_sen,
                                                          maxlen=maxlen,
                                                          padding=padding,
                                                          truncating=truncating)

test_sen = tokenizer.texts_to_sequences(test_raw_sentences)
test_padded = keras.preprocessing.sequence.pad_sequences(test_sen,
                                                          maxlen=maxlen,
                                                          padding=padding,
                                                          truncating=truncating)


# Label tokenizer

label_tokenizer = keras.preprocessing.text.Tokenizer()
label_tokenizer.fit_on_texts(raw_labels)

train_labels = label_tokenizer.texts_to_sequences(train_rlabels)
test_labels = label_tokenizer.texts_to_sequences(test_rlabels)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

print(train_padded.shape, train_labels.shape)
print(test_padded.shape, test_labels.shape)


trainds = tf.data.Dataset.from_tensor_slices((train_padded, train_labels))
trainds = trainds.cache()
trainds = trainds.shuffle(1000)
#trainds = trainds.batch(32)
trainds = trainds.prefetch(1)

testds = tf.data.Dataset.from_tensor_slices((test_padded, test_labels))
testds = testds.cache()
#testds = trainds.batch(32)
testds = testds.prefetch(1)


model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=maxlen),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(6, activation='softmax'),
    ])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(trainds,
          #validation_data=(testds),
          epochs=10)

print(train_padded.shape, train_labels.shape)
print(test_padded.shape, test_labels.shape)
for x,y in trainds.take(1):
    print(x.shape, y.shape)