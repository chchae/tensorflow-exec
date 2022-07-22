#https://dschloe.github.io/python/tensorflow2.0/ch9_1_auto_encoder/
# 

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random



X = [0.3, -0.78, 1.26, 0.03, 1.11, 0.24, -0.24, -0.47, -0.77, -0.37, -0.85, -0.41, -0.27, 0.02, -0.76, 2.66]
Y = [12.27, 14.44, 11.87, 18.75, 17.52, 16.37, 19.78, 19.51, 12.65, 14.74, 10.72, 21.94, 12.83, 15.51, 17.14, 14.42]


if False:
    a = tf.Variable( random.random() )
    b = tf.Variable( random.random() )
    c = tf.Variable( random.random() )


    def compute_loss() :
        y_pred = a * X * X + b * X + c
        loss = tf.reduce_mean( ( Y - y_pred ) ** 2 )
        return loss

    optimizer = tf.keras.optimizers.Adam( lr=0.07 )

    for i in range(1000) :
        optimizer.minimize( compute_loss, var_list=[ a, b, c ] )
        if i % 100 == 99 :
            print( i, 'a:', a.numpy(), 'b:', b.numpy(), 'c:', c.numpy(), 'loss:', compute_loss().numpy() )


    line_x = np.arange( min(X), max(X), 0.01 )
    line_y = a * line_x * line_x + b * line_x + c
    plt.plot( line_x, line_y, 'r-' )
    plt.plot( X, Y, 'bo' )
    plt.savefig( 'test06-A.png' )



if True:
    model = tf.keras.Sequential( [
        tf.keras.layers.Dense( units=6, activation='tanh', input_shape=(1,) ),
        tf.keras.layers.Dense( units=1 )
    ])
    model.compile( optimizer=tf.keras.optimizers.SGD(lr=0.1), loss='mse' )
    model.summary()
    hist = model.fit( X, Y, epochs=100 )
    
    line_x = np.arange( min(X), max(X), 0.1 )
    line_y = model.predict(line_x)
    plt.plot( line_x, line_y, 'r-' )
    plt.plot( X, Y, 'bo' )
    plt.savefig( 'test06-B.png' )
    
    
    
