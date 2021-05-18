import tensorflow as tf
from tensorflow import keras

sentences = ['Fourth Pillar of Democracy',
             'Employees are bad',
             'Now most comics are depressed',
             'Dead Walking']

tokenzier = keras.preprocessing.text.Tokenizer(oov_token='<OOV>',
                                               #filters='depressed',
                                               lower=True,
                                               )

test_sentences = [
    'Fifth Pillar of Commentary',
    'Walking Dead'
]

tokenzier.fit_on_texts(sentences)
dir(tokenzier)
print(tokenzier.index_word)
print(tokenzier.word_index)


og_encoded = tokenzier.texts_to_sequences(sentences)
test_encoded = tokenzier.texts_to_sequences(test_sentences)

print(og_encoded,)
print(test_encoded)


og_padded_pre = keras.preprocessing.sequence.pad_sequences(og_encoded,
                                                    )
test_padded_pre = keras.preprocessing.sequence.pad_sequences(test_encoded,
                                                    )

og_padded_post = keras.preprocessing.sequence.pad_sequences(og_encoded,
                                                    padding='post')

test_padded_post = keras.preprocessing.sequence.pad_sequences(test_encoded,
                                                              padding='post'
                                                              )

og_padded_post_3 = keras.preprocessing.sequence.pad_sequences(og_encoded,
                                                              padding='post',
                                                              maxlen=3,
                                                              truncating='post')

test_padded_post_3 = keras.preprocessing.sequence.pad_sequences(test_encoded,
                                                              padding='post',
                                                              maxlen=3,
                                                              truncating='post')
