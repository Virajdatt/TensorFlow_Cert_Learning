import tensorflow as tf
from tensorflow import keras

sentences = [
    'I love dog',
    'I love cat',
    'You love my dog',
    'That dog is amazing',
]

tokenizer = keras.preprocessing.text.Tokenizer(num_words=10)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sentences_encoded = tokenizer.texts_to_sequences(sentences)
print(word_index)
print(sentences_encoded)

unseen_words = ['Dog is not amazing',
                'That cat is love'
                ]
unseen_words_encoded = tokenizer.texts_to_sequences(unseen_words)
print(unseen_words_encoded)
# [[2, 8, 9], [7, 4, 8, 1]]
# {'love': 1, 'dog': 2, 'i': 3, 'cat': 4, 'you': 5, 'my': 6, 'that': 7, 'is': 8, 'amazing': 9}

oov_tokenizer = keras.preprocessing.text.Tokenizer(num_words=10, oov_token='<OOV>')
oov_tokenizer.fit_on_texts(sentences)
sentences_oov = oov_tokenizer.texts_to_sequences(sentences)
print(oov_tokenizer.word_index)
print(sentences_oov)
unseen_words_oov = oov_tokenizer.texts_to_sequences(unseen_words)
print(unseen_words_oov)

# {'<OOV>': 1, 'love': 2, 'dog': 3, 'i': 4, 'cat': 5, 'you': 6, 'my': 7, 'that': 8, 'is': 9, 'amazing': 10}
# [[4, 2, 3], [4, 2, 5], [6, 2, 7, 3], [8, 3, 9, 1]]
# [[3, 9, 1, 1], [8, 5, 9, 2]]

print(unseen_words_oov)
padded_unseen_words = keras.preprocessing.sequence.pad_sequences(unseen_words_oov, maxlen=10)
print(padded_unseen_words)
padded_unseen_words_post = keras.preprocessing.sequence.pad_sequences(unseen_words_oov,
                                                                      maxlen=10,
                                                                      truncating='post',
                                                                      padding='post'
                                                                      )
print(padded_unseen_words_post)
# {'<OOV>': 1, 'love': 2, 'dog': 3, 'i': 4, 'cat': 5, 'you': 6, 'my': 7, 'that': 8, 'is': 9, 'amazing': 10}

# [[3, 9, 1, 1], [8, 5, 9, 2]]
# [[0 0 0 0 0 0 3 9 1 1]
#  [0 0 0 0 0 0 8 5 9 2]]
# [[3 9 1 1 0 0 0 0 0 0]
#  [8 5 9 2 0 0 0 0 0 0]]
