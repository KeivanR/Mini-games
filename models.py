from tensorflow.keras import layers
import tensorflow as tf

model1 = tf.keras.Sequential()
model1.add(tf.keras.Input(shape=(2,2,)))

model1.add(layers.LSTM(8))
model1.add(layers.Dense(1))

model1.summary()
#print(model1.trainable_weights[1])