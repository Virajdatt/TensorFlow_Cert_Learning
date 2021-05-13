from tensorflow import keras
import numpy as np
xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype = float)
ys = np.array([100, 150, 200, 250, 300, 350], dtype = float)


model = keras.Sequential([
    keras.layers.Dense(1)
])

model.compile(loss='mse',
              optimizer='sgd')

model.fit(xs, ys/100, epochs=100)

print(model.predict([7.0]) * 100)
print(model.predict([5.0]) * 100)
