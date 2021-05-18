
tokenzier = keras.preprocessing.text.Tokenizer(oov_token='<OOV>',
                                               #filters='depressed',
                                               lower=True,
                                               )
tokenzier.fit_on_texts(sentences)

og_encoded = tokenzier.texts_to_sequences(sentences)
test_encoded = tokenzier.texts_to_sequences(test_sentences)

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