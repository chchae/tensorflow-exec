# https://www.tensorflow.org/tutorials/generative/autoencoder


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


(x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print( x_train.shape, x_test.shape )




class Autoencoder(Model) :
    def __init__( self, latent_dim ) :
        super( Autoencoder, self ).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([ 
            layers.Flatten(),
            layers.Dense( latent_dim, activation='relu' )
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense( 784, activation='sigmoid' ),
            layers.Reshape( ( 28, 28 ) )
        ])

    def call( self, x ) :
        encoded = self.encoder(x)
        decoded = self.decoder( encoded )
        return decoded



latent_dim = 64
autoencoder = Autoencoder( latent_dim )
autoencoder.compile( optimizer='adam', loss=losses.MeanSquaredError() )
autoencoder.fit( x_train, x_train, epochs=20, shuffle=True, validation_data=( x_test, x_test ) )
  

encoded_imgs = autoencoder.encoder( x_test ) .numpy()
decoded_imgs = autoencoder.decoder( encoded_imgs ) .numpy()




n = 20
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
# plt.show()
plt.savefig( 'test10.png' )



