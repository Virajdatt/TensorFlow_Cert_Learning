import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

raw_data = pd.read_csv('./Data/bbc-text.csv')

raw_data.head()

stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

def clean_up(review):
    words = review.split()
    word_tokens = [''.join(w) for w in words if not w in stopwords]

    cleaned = " ".join(word_tokens)
    # re.sub(' +', ' ',string4)
    return cleaned


raw_data['cleaned'] = raw_data['text'].apply(clean_up)

labels = raw_data['category'].values
data = raw_data['cleaned'].values

vocab_size = 1000
embedding_dim = 16
max_len = 120
trunc_type = 'post'
padding_type = 'post'
oov_token = '<OOV>'
training_split = 0.8

print(f'labels len {len(labels)}')
print(f'data len {len(data)}')

train_size = int(len(data)*training_split)

training_data = data[:train_size]
training_lab = labels[:train_size]

val_data = data[train_size:]
val_lab = labels[train_size:]

print(f'training data size {len(training_data)}, labels size {len(training_lab)}')
print(f'validation data size {len(val_data)}, validation size {len(val_lab)}')

tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size,
                                               lower=True,
                                               oov_token=oov_token)

tokenizer.fit_on_texts(training_data)

train_sequences = tokenizer.texts_to_sequences(training_data)
train_padded = keras.preprocessing.sequence.pad_sequences(train_sequences,
                                                        maxlen=max_len,
                                                        truncating=trunc_type,
                                                        padding=padding_type)


print(len(train_sequences[0]))
print(len(train_padded[0]))

print(len(train_sequences[1]))
print(len(train_padded[1]))

print(len(train_sequences[10]))
print(len(train_padded[10]))

validation_sequences = tokenizer.texts_to_sequences(val_data)
validation_padded = keras.preprocessing.sequence.pad_sequences(validation_sequences, padding=padding_type, maxlen=max_len,
                                                               truncating=trunc_type,)

print(len(validation_sequences))
print(validation_padded.shape)

label_tokenizer = keras.preprocessing.text.Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(training_lab))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(val_lab))

print(training_label_seq[0])
print(training_label_seq[1])
print(training_label_seq[2])
print(training_label_seq.shape)

print(validation_label_seq[0])
print(validation_label_seq[1])
print(validation_label_seq[2])
print(validation_label_seq.shape)

model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(6, activation='softmax')


])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )

history = model.fit(train_padded, training_label_seq,
          validation_data=(validation_padded, validation_label_seq),
          epochs=30)

import io

embedding_layer = model.layers[0]
#dir(embedding_layer)
embedding_weights = embedding_layer.get_weights()[0]

out_v = io.open('./files/vectors_1.tsv', 'w', encoding='utf-8')
out_m = io.open('./files/meta_data_1.tsv', 'w', encoding='utf-8')
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



