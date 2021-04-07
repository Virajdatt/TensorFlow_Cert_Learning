from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import pydot
housing_data = datasets.fetch_california_housing()

x_train_f, x_test, y_train_f,  y_test = train_test_split(housing_data.data, housing_data.target)

xtrain, xvalid, ytrain, yvalid = train_test_split(x_train_f, y_train_f)

scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
x_test = scaler.transform(x_test)
xvalid = scaler.transform(xvalid)

inputA = keras.layers.Input(shape=[5], name='InputA')
inputB = keras.layers.Input(shape=[6], name='InputB')
hidden1 = keras.layers.Dense(30, activation='relu')(inputB)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
conc = keras.layers.concatenate([inputA, hidden2])
output = keras.layers.Dense(1, name="Output")(conc)
model = keras.Model(inputs=[inputA, inputB], outputs=[output])

model.compile(loss = "mean_squared_error", optimizer="sgd")

model.summary()

xtrain_A, xtrain_B = xtrain[:, :5], xtrain[:, 2:]
xvalid_A, xvalid_B = xvalid[:, :5], xvalid[:, 2:]
x_test_A, x_test_B = x_test[:, :5], x_test[:, 2:]

model.fit( (xtrain_A, xtrain_B), ytrain, epochs=20,
            validation_data=((xvalid_A, xvalid_B),yvalid ))

model.evaluate((x_test_A, x_test_B ), y_test)