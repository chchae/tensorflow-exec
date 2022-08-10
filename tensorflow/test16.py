import tensorflow.keras as K
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


img_rows = 28
img_cols = 28
channels = 1
img_shape = ( img_rows, img_cols, channels )
z_dim = 100


class Generator( K.Model ) :
    def __init__( self, img_shape, z_dim ) :
        super().__init__( name = "generator" )
        self.model = K.models.Sequential()
        self.model.add( K.layers.Dense( 128, input_dim=z_dim ) )
        self.model.add( K.layers.LeakyReLU( alpha=0.01 ) )
        self.model.add( K.layers.Dense( img_shape[0] * img_shape[1] * img_shape[2], activation='tanh' ) )
        self.model.add( K.layers.Reshape( img_shape ) )

    def call( self, x ) :
        return self.model(x)

generator = Generator( img_shape, z_dim )
generator.compile( loss=K.losses.BinaryCrossentropy(), optimizer=K.optimizers.Adam() )


class Discriminator( K.Model ) :
    def __init__( self, img_shape ) :
        super().__init__( name = "discriminator" )
        self.model = K.models.Sequential()
        self.model.add( K.layers.Flatten( input_shape=img_shape ) )
        self.model.add( K.layers.Dense( 128 ) )
        self.model.add( K.layers.LeakyReLU( alpha=0.01 ) )
        self.model.add( K.layers.Dense( 1, activation='sigmoid' ) )

    def call( self, x ) :
        return self.model(x)

discriminator = Discriminator( img_shape )
discriminator.compile( loss=K.losses.BinaryCrossentropy(), optimizer=K.optimizers.Adam(), metrics=['accuracy'] )
discriminator.trainable = False


class GAN( K.Model ) :
    def __init__( self, generator, discriminator ) :
        super().__init__( name = "gan" )
        self.model = K.Sequential()
        self.model.add( generator )
        self.model.add( discriminator )

    def call( self, x ) :
        return self.model(x)

gan = GAN( generator, discriminator )
gan.compile( loss=K.losses.BinaryCrossentropy(), optimizer=K.optimizers.Adam() )



def train( iterations, batch_size, sample_interval ) :
    (X_train, _), (_,_) = K.datasets.mnist.load_data()      # (60000, 28, 28)
    X_train = X_train / 127.5 - 1.0
    X_train = np.expand_dims( X_train, axis=3 )             # (60000, 28, 28, 1)
    real = np.ones( ( batch_size, 1 ) )
    fake = np.zeros( ( batch_size, 1 ) )

    for iteration in range(iterations) :
        idx = np.random.randint( 0, X_train.shape[0], batch_size )
        imgs = X_train[idx]
        z = np.random.normal( 0, 1, (batch_size, 100) )
        gen_imgs = generator.predict(z,verbose=3)           # (128, 28, 28, 1)

        d_loss_real = discriminator.train_on_batch( imgs, real )
        d_loss_fake = discriminator.train_on_batch( gen_imgs, fake )
        d_loss, accuracy = 0.5 * np.add( d_loss_real, d_loss_fake )
        accuracy = 0

        #z = np.random.normal( 0, 1, (batch_size, 100) )
        #gen_imgs = generator.predict(z,verbose=3)
        gan_loss = gan.train_on_batch( z, real, )

        if( iteration + 1 ) % sample_interval == 0 :
            print( "%d [D loss: %f, accuracy : %.2f%%] [G loss: %f]" % 
                (iteration + 1, d_loss, 100.0*accuracy, gan_loss) )
            sample_images( generator )


def sample_images( generator, image_grid_rows=4, image_grid_cols=6 ) :
    z = np.random.normal( 0, 1, (image_grid_rows * image_grid_cols, z_dim) )
    gen_imgs = generator.predict(z,verbose=3)
    
    plt.clf()
    fig, axs = plt.subplots( image_grid_rows, image_grid_cols, figsize=(4,4), sharey=True, sharex=True )
    #axs = np.expand_dims( axs, axis=0 )

    cnt = 0
    for i in range( image_grid_rows ) :
        for j in range( image_grid_cols ) :
            axs[i,j].imshow( gen_imgs[ cnt, :, :, 0 ], cmap='gray' )
            axs[i,j].axis('off')
            cnt += 1
    plt.savefig( 'test16.png' )


iterations = 40000
batch_size = 128
sample_interval = 100
train( iterations, batch_size, sample_interval )



