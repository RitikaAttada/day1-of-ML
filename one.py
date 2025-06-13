import tensorflow as tf
import numpy as np
from tensorflow import keras

def salary_model (y_new):
    xs = np.array([0,1,2,3,4,5,6], dtype=float) #number of years
    ys = np.array([0.5,0.6,0.7,0.8,0.9,1.0,1.1], dtype=float) #salary
    model = keras.Sequential([keras.layers.Dense(units =1, input_shape=[1])])
    model.compile(optimizer="sgd", loss='mean_squared_error')
    model.fit(xs,ys,epochs=500)
    return model.predict(np.array([y_new]))[0][0]

prediction = salary_model(8.0)

print(prediction*100, "thosand dollars")