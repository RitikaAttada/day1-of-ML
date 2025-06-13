import tensorflow as tf
import numpy as np
from tensorflow import keras
import os

def bicep_size(weight):

    model_path = "six.h5"
    if (os.path.exists(model_path)):
        print("Loading saved model... :-)")
        model = keras.models.load_model(model_path)
    else :
        xs = np.array([5,10,15,20,25], dtype= float)
        ys = np.array([11,12,13,14,15], dtype=float)
        model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01), loss="mean_squared_error")
        model.fit(xs,ys,epochs=1000)
        model.save("six.h5")

    return model.predict(np.array([weight]))[0][0]

prediction = bicep_size(30)
print(prediction, " inches")