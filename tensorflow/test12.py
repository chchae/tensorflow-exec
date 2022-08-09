from inspect import ArgSpec
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.datasets.mnist


batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 500
epsilon_std = 1.0




def sampling( args: tuple ) :
    z_mean, z_log_var = args
    epsilon = tf.keras.backend.random_normal( shape=(tf.keras.backend.shape(z_mean)[0], latent_dim ), mean=0.0, stddev=epsilon_std )
    return z_mean + tf.keras.backend.exp( z_log_var / 2 ) * epsilon


#def make_encoder() :
x = tf.keras.layers.Input( shape=(original_dim,), name="input" )
h = tf.keras.layers.Dense( intermediate_dim, activation='relu', name="encoding" )(x)
z_mean = tf.keras.layers.Dense( latent_dim, name="mean" )(h)
z_log_var = tf.keras.layers.Dense( latent_dim, name="log-variance" )(h)
z = tf.keras.layers.Lambda( sampling, output_shape=(latent_dim,)) ( [ z_mean, z_log_var ] )
encoder = tf.keras.models.Model( x, [ z_mean, z_log_var, z ], name="encoder" )
#return encoder

#encoder = make_encoder()



#def make_decoder() :
input_decoder = tf.keras.layers.Input( shape=(latent_dim,), name="decoder_input" )
decoder_h = tf.keras.layers.Dense( intermediate_dim, activation='relu', name="decoder_h" )(input_decoder)
x_decodeded = tf.keras.layers.Dense( original_dim, activation='sigmoid', name="flat_decoded" )(decoder_h)
decoder = tf.keras.models.Model( input_decoder, x_decodeded, name="decoder" )
#return decoder

#decoder = make_decoder()


output_combined = decoder( encoder(x)[2] )
vae = tf.keras.models.Model( x, output_combined )
vae.summary()



kl_loss = -0.5 * tf.keras.backend.sum( 1 + z_log_var - tf.keras.backend.exp(z_log_var) - tf.keras.backend.square(z_mean), axis=1 )
vae.add_loss( tf.keras.backend.mean(kl_loss) / 784.0 )
vae.compile( optimizer='rmsprop', loss='binary_crossentropy' )


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train = tf.convert_to_tensor( x_train, dtype='float', shape=(28*28) )
#x_test  = tf.convert_to_tensor( x_test, dtype='float' )
#y_train = tf.one_hot( y_train, depth=10 )
#y_test  = tf.one_hot( y_test, depth=10 )

x_train = x_train / 255.
x_test = x_test / 255.
x_train = x_train.reshape( -1, 28*28 )
x_test = x_test.reshape( -1, 28*28 )


vae.fit( x_train, x_train, shuffle=True, epochs=epochs, batch_size=batch_size )

