import tensorflow as tf
import numpy as np
from tensorflow import keras

def ExamScore(study_hours):
    xs = np.array([1,2,3,4,5,6], dtype=float)
    ys = np.array([50,60,65,70,75,80], dtype=float)
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer = "sgd", loss="mean_squared_error")
    model.fit(xs,ys,epochs=500)
    return model.predict(np.array([study_hours]))[0][0]

prediction = ExamScore(7)
print(prediction , "marks")