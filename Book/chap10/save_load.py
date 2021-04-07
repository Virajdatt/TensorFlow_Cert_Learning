from tensorflow import keras
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing_data = datasets.fetch_california_housing()

x_train, x_test, y_train,  y_test = train_test_split(housing_data.data, housing_data.target)

xt, xv, yt, yv = train_test_split(x_train, y_train)

scaler = StandardScaler()
xt = scaler.fit_transform(xt)
x_test = scaler.transform(x_test)
xv = scaler.transform(xv)

model = keras.Sequential([keras.layers.Dense(30, input_shape=xt[1:].shape, activation='relu'),
                          keras.layers.Dense(1)
                        ])

model.compile(loss = "mean_squared_error", optimizer="sgd")
model.fit(xt, yt, epochs=20, validation_data=(xv, yv))
model.evaluate(x_test, y_test)

model.predict(x_test)

model.save('./files/save_test.h5')

model2 = keras.models.load_model('./files/save_test.h5')
print(model.evaluate(x_test, y_test))
print(model2.evaluate(x_test, y_test))

# 162/162 [==============================] - 0s 485us/step - loss: 0.3690
# 0.36899006366729736
# 162/162 [==============================] - 0s 457us/step - loss: 0.3690
# 0.36899006366729736