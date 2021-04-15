from tensorflow import keras
import numpy as np

tokenizer = keras.preprocessing.text.Tokenizer()
data = "In the town of Athy one Jeremy Lanigan \n Battered away til he hadnt a pound. \nHis father died and made him a man again \n Left him a farm and ten acres of ground. \nHe gave a grand party for friends and relations \nWho didnt forget him when come to the wall, \nAnd if youll but listen Ill make your eyes glisten \nOf the rows and the ructions of Lanigans Ball. \nMyself to be sure got free invitation, \nFor all the nice girls and boys I might ask, \nAnd just in a minute both friends and relations \nWere dancing round merry as bees round a cask. \nJudy ODaly, that nice little milliner, \nShe tipped me a wink for to give her a call, \nAnd I soon arrived with Peggy McGilligan \nJust in time for Lanigans Ball. \nThere were lashings of punch and wine for the ladies, \nPotatoes and cakes; there was bacon and tea, \nThere were the Nolans, Dolans, OGradys \nCourting the girls and dancing away. \nSongs they went round as plenty as water, \nThe harp that once sounded in Taras old hall,\nSweet Nelly Gray and The Rat Catchers Daughter,\nAll singing together at Lanigans Ball. \nThey were doing all kinds of nonsensical polkas \nAll round the room in a whirligig. \nJulia and I, we banished their nonsense \nAnd tipped them the twist of a reel and a jig. \nAch mavrone, how the girls got all mad at me \nDanced til youd think the ceiling would fall. \nFor I spent three weeks at Brooks Academy \nLearning new steps for Lanigans Ball. \nThree long weeks I spent up in Dublin, \nThree long weeks to learn nothing at all,\n Three long weeks I spent up in Dublin, \nLearning new steps for Lanigans Ball. \nShe stepped out and I stepped in again, \nI stepped out and she stepped in again, \nShe stepped out and I stepped in again, \nLearning new steps for Lanigans Ball. \nBoys were all merry and the girls they were hearty \nAnd danced all around in couples and groups, \nTil an accident happened, young Terrance McCarthy \nPut his right leg through miss Finnertys hoops. \nPoor creature fainted and cried Meelia murther, \nCalled for her brothers and gathered them all. \nCarmody swore that hed go no further \nTil he had satisfaction at Lanigans Ball. \nIn the midst of the row miss Kerrigan fainted, \nHer cheeks at the same time as red as a rose. \nSome of the lads declared she was painted, \nShe took a small drop too much, I suppose. \nHer sweetheart, Ned Morgan, so powerful and able, \nWhen he saw his fair colleen stretched out by the wall, \nTore the left leg from under the table \nAnd smashed all the Chaneys at Lanigans Ball. \nBoys, oh boys, twas then there were runctions. \nMyself got a lick from big Phelim McHugh. \nI soon replied to his introduction \nAnd kicked up a terrible hullabaloo. \nOld Casey, the piper, was near being strangled. \nThey squeezed up his pipes, bellows, chanters and all. \nThe girls, in their ribbons, they got all entangled \nAnd that put an end to Lanigans Ball."

# Tokenizing things from the data
corpus = data.lower().split("\n")
tokenizer.fit_on_texts(corpus)
word_index = tokenizer.word_index
total_words = len(word_index) + 1
print(total_words)

# Creating input sequence

input_sequnce = []

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    # print(token_list)

    for tokens in range(1, len(token_list)):
        n_gram_sequence = token_list[:tokens + 1]
        # print(token_list)
        # print(n_gram_sequence)
        input_sequnce.append(n_gram_sequence)
    # break

"""
Sample output with break and print statements enabled
[4, 2, 66, 8, 67, 68, 69, 70]
[4, 2]
[4, 2, 66]
[4, 2, 66, 8]
[4, 2, 66, 8, 67]
[4, 2, 66, 8, 67, 68]
[4, 2, 66, 8, 67, 68, 69]
[4, 2, 66, 8, 67, 68, 69, 70]
"""

# Padded Sequences

max_sequence_len = max(len(x) for x in input_sequnce)
padded_seq = np.array(keras.preprocessing.sequence.pad_sequences(input_sequnce,
                                                                 maxlen=max_sequence_len,
                                                                 padding='pre')
                      )

# Crete features and lables

X, label = padded_seq[:, :-1], padded_seq[:,-1]  # Features everything expect the next word
Y = keras.utils.to_categorical(label, num_classes=total_words)  # OHE
# Modeling

model = keras.Sequential([
    keras.layers.Embedding(total_words, 64, input_length=max_sequence_len - 1),
    keras.layers.Bidirectional(keras.layers.LSTM(20)),
    keras.layers.Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )
tensorboard = keras.callbacks.TensorBoard('./files/simple_textgen')

model.fit(X, Y,
          epochs=500,
          callbacks=[tensorboard],

          )

# Predictions
seed_text = "Chaitnay works a lot"
next_words = 10

for _ in range(next_words):
    seq = tokenizer.texts_to_sequences([seed_text])[0] # returns a list of lists so we use 0 index
    test_pad = keras.preprocessing.sequence.pad_sequences([seq],
                                                          maxlen=max_sequence_len-1,
                                                          padding='pre')
    predicted = model.predict_classes(test_pad,verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == int(predicted):
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)
