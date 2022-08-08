# https://dschloe.github.io/python/tensorflow2.0/ch9_1_auto_encoder/
#

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


(train_X, train_Y), (test_X, test_Y) = tf.keras.datasets.mnist.load_data()
print( train_X.shape, train_Y.shape )

train_X = train_X / 255
test_X = test_X / 255


plt.imshow( train_X[0].reshape(28,28), cmap='gray' )
plt.colorbar()
plt.savefig( 'test07.png' )


train_X = train_X.reshape( -1, 28*28 )
test_X = test_X.reshape( -1, 28*28 )
print( train_X.shape, test_X.shape )


model = tf.keras.Sequential([
    tf.keras.layers.Dense( 784, activation='relu', input_shape=(784,) ),
    tf.keras.layers.Dense( 4096, activation='relu' ),
    tf.keras.layers.Dense( 40960, activation='relu' ),
    tf.keras.layers.Dense( 4096, activation='relu' ),
    tf.keras.layers.Dense( 784, activation='sigmoid' )
])
model.compile( optimizer=tf.optimizers.Adam(), loss='mse' )
model.summary()


model.fit( train_X, train_X, epochs=10000, batch_size=256 )



