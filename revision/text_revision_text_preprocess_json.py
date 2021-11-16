import json
import warnings
import tensorflow as tf
import pandas as pd

warnings.filterwarnings('ignore')

## Read the json data

sarcasm_data = pd.read_json('/Users/virajdatt/Desktop/TFCertPrep/data/Sarcasm_Headlines_Dataset_v2.json',
                            lines=True)



## HYPER-PARAM:-

NUM_WORDS = 100
TRUNCATE = 'post'  # 'pre'
PADDING = 'post'   # 'pre
MAX_LEN = 10

## 1. Fit Tokenizer

sarcasm_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=NUM_WORDS,
                                                          oov_token='<OOV>')
sarcasm_tokenizer.fit_on_texts(sarcasm_data['headline'].values)

## 2. Convert text to sequence
sarcasm_seq = sarcasm_tokenizer.texts_to_sequences(sarcasm_data['headline'].values)

## 3. Convert the sequence to padded sequences

sarcasm_padded = tf.keras.preprocessing.sequence.pad_sequences(sarcasm_seq,
                                                               truncating=TRUNCATE,
                                                               padding=PADDING,
                                                               maxlen=MAX_LEN)




