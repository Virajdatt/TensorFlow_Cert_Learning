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

loaded_m = keras.models.load_model('/Users/virajdatt.kohir/Downloads/testing_save.h5')

def preprocess(texts):
    X = np.array(tokenizer.texts_to_sequences(texts)) - 1
    return tf.one_hot(X, max_chars)

X_new = preprocess(["Ho"])
#Y_pred = model.predict_classes(X_new)
Y_pred = np.argmax(loaded_m(X_new), axis=-1)
tokenizer.sequences_to_texts(Y_pred + 1)[0][-1] # 1st sentence, last char