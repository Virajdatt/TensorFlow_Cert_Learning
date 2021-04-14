from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
import io


imdb, info = tfds.load('imdb_reviews', with_info=True,
                       as_supervised=True)

raw_train, raw_test = imdb['train'], imdb['test']

# Taking a peek into our new dataset :)

# positive_review = raw_train.filter(lambda x,y: y==1)
# negative_review = raw_train.filter(lambda x,y: y==0)
# positive_review.element_spec
# negative_review.element_spec
# # (TensorSpec(shape=(), dtype=tf.string, name=None), TensorSpec(shape=(), dtype=tf.int64, name=None))
# len(list((raw_train.as_numpy_iterator()))) # 25000
# len(list((positive_review.as_numpy_iterator()))) # 12500
# len(list((negative_review.as_numpy_iterator()))) # 12500


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

VOCAB = 1000  # Used in (Tokenizer and keras embedding layer)
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
print(train_seq[0])
print(train_padded[0])

testing_seq = tokenizer.texts_to_sequences(testing_seneteces)
testing_padded = keras.preprocessing.sequence.pad_sequences(testing_seq, maxlen=MAX_LEN,
                                                            truncating=TRUNC_TYPE)
print(testing_seq[0])
print(testing_padded[0])

print(f'The training shape is {train_padded.shape},'
      f'The testing shape is {testing_padded.shape}')

# Sanity check of things
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(f'Decoded Review: \n{decode_review(train_padded[0])}')
print(f'Original Review: \n{training_seneteces[0]}')
# Modeling part

model = keras.Sequential([
    keras.layers.Embedding(VOCAB, EVD, input_length=MAX_LEN),
    keras.layers.Flatten(),
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )

model.fit(train_padded,
          training_labels,
          epochs=EPOCHS,
          validation_data=(testing_padded, testing_labels)

          )

# Steps to Visualize the embeddings

# 1. Get the weights and the embedding layer

embedding_layer = model.layers[0]
embedding_weights = embedding_layer.get_weights()[0]
print(embedding_weights.shape)  # VOCAB x EVD
# (1000, 16)

# 2. Create the files necessary for the embedding Visualisation

out_v = io.open('files/vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('files/meta.tsv', 'w', encoding='utf-8')

for word_num in range(1, VOCAB):
    word = reverse_word_index[word_num]
    embedding = embedding_weights[word_num]

    #print(word)
    out_m.write(word+'\n')
    out_v.write('\t'.join([str(x) for x in embedding]) + "\n")
    #print('\t'.join([str(x) for x in embedding]) + "\n")
    #break
out_m.close()
out_v.close()

# 3. Upload these files and vizualize in http://projector.tensorflow.org/