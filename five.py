import tensorflow as tf
import numpy as np
from tensorflow import keras
import os

def piano_skill(days):
    model_path = "five.h5"
    if (os.path.exists(model_path)):
        print("Loading saved model...")
        model = keras.models.load_model(model_path)
    else:
        xs = np.array([0,1,2,3,4], dtype=float)
        ys = np.array([1,2,3,4,5], dtype=float)
        model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01), loss="mean_squared_error")
        model.fit(xs,ys,epochs=1000)
        model.save("five.h5")
        print("model saved :-)")

    return model.predict(np.array([days]))[0][0]

prediction = piano_skill(6)
print(prediction)