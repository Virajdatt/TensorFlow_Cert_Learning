import tensorflow as tf

model = tf.keras.models.Sequential([

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(30, return_sequences=True), input_shape=[None, 1]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(30, return_sequences=True)),
        #tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(20, activation="relu"),
        tf.keras.layers.Dense(N_FEATURES)
    ])

   
    # lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    #     lambda epoch: 1e-8 * 10 ** (epoch / 20)
    # )


    model.compile(
        loss=tf.keras.losses.Huber(),
        optimizer='adam',
        metrics=["mae"]
    )


    MC = tf.keras.callbacks.ModelCheckpoint(
        '/Users/virajdatt/Desktop/TFCertPrep/Exam_Questions/final_mc.h5',
        monitor='val_mae',
        save_best_only='True',
        verbose=1
    )

    ES = tf.keras.callbacks.EarlyStopping(
        monitor='val_mae',
        patience=5,
        verbose=1,
        restore_best_weights='True'
    )

    model.fit(
        train_set, validation_data=valid_set, epochs=10,
        callbacks=[MC, ES]
    )
