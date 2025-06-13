import tensorflow as tf
import numpy as np
from tensorflow import keras
import os

def flexibility(months):

    model_path = "eight.h5"
    if(os.path.exists(model_path)):
        print("loading model ....")
        model = keras.models.load_model(model_path)
    else:
        xs = np.array([1,2,3,4,5], dtype=float)
        ys=np.array([2,3,4,5,6], dtype=float)
        model  = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mean_squared_error")
        model.fit(xs,ys,epochs=2000)
        model.save("eight.h5")
    
    return model.predict(np.array([months]))[0][0]

print(flexibility(6))