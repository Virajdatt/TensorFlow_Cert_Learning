import json
import pandas as pd
import tensorflow as tf

sentences = [
    'I love dog',
    'I love cat',
    'You love my dog',
    'That dog is amazing',
]


unseen_words = ['Dog is not amazing',
                'That cat is love'
                ]

## HYPER-PARAM:-

NUM_WORDS = 10
TRUNCATE = 'post'  # 'pre'
PADDING = 'post'   # 'pre
MAX_LEN = 10

## Fit Tokenizer
simple_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=NUM_WORDS,
                                                         oov_token='<OOV>')
simple_tokenizer.fit_on_texts(sentences)

word_index = simple_tokenizer.word_index

# Convert text to sequence

simple_sequence = simple_tokenizer.texts_to_sequences(sentences)
unseen_sequence = simple_tokenizer.texts_to_sequences(unseen_words)

# Convert the sequence to padded sequencestext_revision_text_preprocess.py
simple_padded = tf.keras.preprocessing.sequence.pad_sequences(simple_sequence,
                                                              truncating=TRUNCATE,
                                                              padding=PADDING,
                                                              maxlen=MAX_LEN)

unseen_padded = tf.keras.preprocessing.sequence.pad_sequences(unseen_sequence,
                                                              truncating=TRUNCATE,
                                                              padding=PADDING,
                                                              maxlen=MAX_LEN)



