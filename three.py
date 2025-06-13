import tensorflow as tf
import numpy as np
from tensorflow import keras

def guess_height(age):
    xs= np.array([5,6,7,8,9,10], dtype=float)
    ys = np.array([105,110,115,120,125,130])
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer="sgd", loss="mean_squared_error")
    model.fit(xs,ys,epochs=500)
    return model.predict(np.array([age]))[0][0]

prediction = guess_height(11)
print(prediction, " cm")
    