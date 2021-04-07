from tensorflow import keras
import numpy as np

tdata = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")

results = np.array([[0],[1],[1],[0]], "float32")

class AccStop(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy')>0.98:
            print("Stopping training BRUH!!!!")
            self.model.stop_training = True

model = keras.Sequential([keras.layers.Dense(16, input_dim=2 ,activation='relu'),
                          keras.layers.Dense(1, activation='sigmoid'),]
                         )

model.compile(loss='mean_squared_error', optimizer='adam', metrics='accuracy')

model.fit(tdata, results, epochs=160, callbacks=AccStop())

print(model.predict(tdata).round())