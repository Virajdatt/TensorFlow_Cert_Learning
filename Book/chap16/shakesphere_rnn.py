import tensorflow as tf
from tensorflow import keras
import numpy as np

np.random.seed(42)
tf.random.set_seed(42)

# Read in the text data
with open('./Data/tiny_shake.txt') as f:
    ssphere_txt = f.read()


print(ssphere_txt[:148])

# Create a char level tokenizer
tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(ssphere_txt)


# Test the tokenizer
tokenizer.texts_to_sequences(['First'])

# Max number of chars
max_chars = len(tokenizer.word_index)

# Total number of chars in the entire text data
dataset_size = tokenizer.document_count

# Encode the text data to number sequence

[encoded_text] = np.array(tokenizer.texts_to_sequences(([ssphere_txt]))) - 1
#len(encoded_text) #1115389

# encoded_text[:148]
# tokenizer.sequences_to_texts([encoded_text[:148]])
# tokenizer.texts_to_sequences(([ssphere_txt]))

train_size = dataset_size * 90 // 100
dataset = tf.data.Dataset.from_tensor_slices(encoded_text[:train_size])

n_steps = 100
window_size = n_steps + 1
batch_size = 32

dataset = dataset.repeat().window(window_size, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(window_size))
dataset = dataset.shuffle(10000).batch(batch_size)
dataset = dataset.map(lambda window: (window[:, :-1], window[:, 1:]))
dataset = dataset.map(
    lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_chars), Y_batch)
)
dataset = dataset.prefetch(1)
for X_batch, Y_batch in dataset.take(1):
    print(X_batch.shape, Y_batch.shape)


model = keras.Sequential([
    keras.layers.GRU(30, return_sequences=True, input_shape=[None, max_chars]),
    keras.layers.GRU(30, return_sequences=True, dropout=0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(max_chars,
                                activation='softmax'))
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
history = model.fit(dataset, steps_per_epoch=train_size // batch_size,
                    epochs=10)


