import tensorflow as tf
import numpy as np
from tensorflow import keras
import os

def confidence(hours):
    model_path = "seven.h5"
    if (os.path.exists(model_path)):
        print("loading model")
        model = keras.models.load_model(model_path)
    else :
        xs = np.array([0,2,4,6,8], dtype=float)
        ys = np.array([1,2,4,6,7.5], dtype=float)
        model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01), loss="mean_squared_error")
        model.fit(xs,ys,epochs=3000)
        model.save("seven.h5")
        
    return model.predict(np.array([hours]))[0][0]

prediction = confidence(10)
print(prediction)
