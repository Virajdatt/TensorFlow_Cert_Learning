from tensorflow import keras
import pandas as pd

data = pd.read_csv('Data/bbc-text.csv')

print(data.head())

labels = data['category'].values
raw_data = data['text'].values

print(len(raw_data[0]))
print(raw_data[0])
max = 0

for i,j in zip(labels, raw_data):
    if len(j) > max:
        max = len(j)
    #print(i, len(j))
    #break
print(max)

tokenizer = keras.preprocessing.text.Tokenizer(oov_token='<OOV>',
                                               )

tokenizer.fit_on_texts(raw_data)
print(len(tokenizer.word_index))

sequences = tokenizer.texts_to_sequences(raw_data)
padded_seq = keras.preprocessing.sequence.pad_sequences(sequences,
                                                        padding='post')
print(len(padded_seq))
print(padded_seq.shape)

huge_padded = []
for i in padded_seq:
    print(i.shape)

for i in huge_padded:
    print(i)
    break

huge_padded_words = []

for i in huge_padded:
    print(i)
    try:
        word = tokenizer.index_word.get(huge_padded[i])
        huge_padded_words.append(word)
    except:
        huge_padded_words.append('AARAM')
    break


