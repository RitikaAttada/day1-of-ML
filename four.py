import tensorflow as tf
import numpy as np
from tensorflow import keras

def to_fahrenheit(celsius):
    xs = np.array([0,10,20,30,40], dtype=float)
    ys = np.array([32,50,68,86,104], dtype=float)
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01), loss="mean_squared_error")
    model.fit(xs,ys,epochs=5000)
    return model.predict(np.array([celsius]))[0][0]

prediction = to_fahrenheit(25)
print(prediction, " F")