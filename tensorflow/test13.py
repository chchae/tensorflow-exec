import tensorflow.keras as K
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



img_rows = 28
img_cols = 28
channels = 1
img_shape = ( img_rows, img_cols, channels )
z_dim = 100


def build_generator( img_shape, z_dim ) :
    model = K.models.Sequential()
    model.add( K.layers.Dense( 128, input_dim=z_dim ) )
    model.add( K.layers.LeakyReLU( alpha=0.01 ) )
    model.add( K.layers.Dense( 28 * 28 * 1, activation='tanh' ) )
    model.add( K.layers.Reshape( img_shape ) )
    return model

generator = build_generator( img_shape, z_dim )
generator.compile( loss='binary_crossentropy', optimizer=K.optimizers.Adam() )



def build_descriminator( img_shape, z_dim ) :
    model = K.models.Sequential()
    model.add( K.layers.Flatten( input_shape=img_shape ) )
    model.add( K.layers.Dense( 128 ) )
    model.add( K.layers.LeakyReLU( alpha=0.01 ) )
    model.add( K.layers.Dense( 1, activation='sigmoid' ) )
    return model

discriminator = build_descriminator( img_shape, z_dim )
discriminator.compile( loss='binary_crossentropy', optimizer=K.optimizers.Adam(), metrics=['accuracy'] )
discriminator.trainable = False


def build_gan( generator, discriminator ) :
    model = K.Sequential()
    model.add( generator )
    model.add( discriminator )
    return model

gan = build_gan( generator, discriminator )
gan.compile( loss='binary_crossentropy', optimizer=K.optimizers.Adam() )


if False  :
    batch_size = 128
    z = np.random.normal( 0, 1, (batch_size, 100) )
    gen_imgs = generator.predict(z)
    print( gen_imgs.shape )    # (128, 28, 28, 1)
    fig, axs = plt.subplots( 1, 4, figsize=(4,4), sharey=True, sharex=True )
    print( axs.shape )
    axs = axs.reshape( 1, 4 )
    img = gen_imgs[ 0, :, :, 0 ]
    axs[0,0].imshow( img, cmap='gray' )
    exit(0)


losses = []
accuracies = []
iteraction_checkpoints = []

def train( iterations, batch_size, sample_interval ) :
    (X_train, _), (_,_) = K.datasets.mnist.load_data()   # (60000, 28, 28)
    X_train = X_train / 127.5 - 1.0
    X_train = np.expand_dims( X_train, axis=3 )          # (60000, 28, 28, 1)
    real = np.ones( (batch_size, 1) )
    fake = np.zeros( (batch_size, 1) )

    for iteration in range(iterations) :
        idx = np.random.randint( 0, X_train.shape[0], batch_size )
        imgs = X_train[idx]
        z = np.random.normal( 0, 1, (batch_size, 100) )
        gen_imgs = generator.predict(z,verbose=3)

        d_loss_real = discriminator.train_on_batch( imgs, real )
        d_loss_fake = discriminator.train_on_batch( gen_imgs, fake )
        d_loss, accuracy = 0.5 * np.add( d_loss_real, d_loss_fake )
        accuracy = 0

        z = np.random.normal( 0, 1, (batch_size, 100) )
        gen_imgs = generator.predict(z,verbose=3)
        g_loss = gan.train_on_batch( z, real, )

        if( iteration + 1 ) % sample_interval == 0 :
            losses.append( (d_loss, g_loss) )
            accuracies.append( 100.0 * accuracy )
            iteraction_checkpoints.append( iteration + 1 )

            print( "%d [D loss: %f, accuracy : %.2f%%] [G loss: %f]" % (iteration + 1, d_loss, 100.0*accuracy, g_loss) )
            sample_images( generator )


def sample_images( generator, image_grid_rows=1, image_grid_cols=4 ) :
    z = np.random.normal( 0, 1, (image_grid_rows * image_grid_cols, z_dim) )
    gen_imgs = generator.predict(z,verbose=3)
    fig, axs = plt.subplots( image_grid_rows, image_grid_cols, figsize=(4,4), sharey=True, sharex=True )
    axs = np.expand_dims( axs, axis=0 )

    cnt = 0
    for i in range( image_grid_rows ) :
        for j in range( image_grid_cols ) :
            axs[i,j].imshow( gen_imgs[ cnt, :, :, 0 ], cmap='gray' )
            axs[i,j].axis('off')
            plt.imsave( 'test13.png', gen_imgs[ cnt, :, :, 0 ], cmap='gray' )
            cnt += 1


iterations = 60000
batch_size = 128
sample_interval = 1000
train( iterations, batch_size, sample_interval )



