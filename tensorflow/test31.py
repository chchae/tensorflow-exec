# https://github.com/rickiepark/gans-in-action/blob/master/chapter-4/Chapter_4_DCGAN.ipynb
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as K


class Generator( K.Model ) :
    def __init__( self, z_dim ) :
        super().__init__( name = 'generator' )
        self.model = K.models.Sequential()
        self.model.add( K.layers.Dense( 256 * 7 * 7, input_dim=z_dim ) )
        self.model.add( K.layers.Reshape( ( 7, 7, 256 ) ) )
        self.model.add( K.layers.Conv2DTranspose( 128, kernel_size=3, strides=2, padding='same' ) )
        self.model.add( K.layers.BatchNormalization() )
        self.model.add( K.layers.LeakyReLU( alpha=0.01 ) )
        self.model.add( K.layers.Conv2DTranspose( 64, kernel_size=3, strides=1, padding='same' ) )
        self.model.add( K.layers.BatchNormalization() )
        self.model.add( K.layers.LeakyReLU(alpha=0.01) )
        self.model.add( K.layers.Conv2DTranspose( 1, kernel_size=3, strides=2, padding='same' ) )
        self.model.add( K.layers.Activation('tanh') )

    def call( self, x ) :
        return self.model( x )


class Discriminator( K.Model ) :
    def __init__( self, img_shape ) :
        super().__init__( name = 'discriminator' )
        self.model = K.models.Sequential()
        self.model.add( K.layers.Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding='same'))
        self.model.add(K.layers.LeakyReLU(alpha=0.01))
        self.model.add( K.layers.Conv2D(64, kernel_size=3, strides=2, padding='same'))
        self.model.add(K.layers.LeakyReLU(alpha=0.01))
        self.model.add( K.layers.Conv2D(128, kernel_size=3, strides=2, padding='same'))
        self.model.add(K.layers.LeakyReLU(alpha=0.01))
        self.model.add(K.layers.Flatten())
        self.model.add(K.layers.Dense(1, activation='sigmoid'))

    def call( self, x ) :
        return self.model(x)



class GAN( K.Model ) :
    def __init__( self, img_shape, z_dim ) :
        super().__init__( name='gan' )

        self.generator = Generator( z_dim )        
        self.generator.compile( loss=K.losses.BinaryCrossentropy(), optimizer=K.optimizers.Adam() )

        self.discriminator = Discriminator( img_shape )
        self.discriminator.compile( loss=K.losses.BinaryCrossentropy(), optimizer=K.optimizers.Adam(), metrics=['accuracy'] )
        self.discriminator.trainable = False

        self.model = K.Sequential()
        self.model.add(self.generator)
        self.model.add(self.discriminator)

        self.lossess = []
        self.accuracies = []
        self.iteration_checkpoints = []


    def call( self, x ) :
        return self.model(x)

 
    def exec_train( self, iterations, batch_size, sample_interval ):
        (X_train, _), (_, _) = K.datasets.mnist.load_data()
        X_train = X_train / 127.5 - 1.0
        X_train = np.expand_dims(X_train, axis=3)

        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        image_grid_rows = 20
        image_grid_columns = 16
        fig, axs = plt.subplots( image_grid_rows, image_grid_columns, figsize=(8, 8), sharey=True, sharex=True)
        numrow = 0

        for iteration in range(iterations):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            z = np.random.normal(0, 1, (batch_size, 100))
            gen_imgs = self.generator.predict(z, verbose=3)

            d_loss_real = self.discriminator.train_on_batch(imgs, real)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

            z = np.random.normal(0, 1, (batch_size, 100))
            gen_imgs = self.generator.predict(z, verbose=3)

            g_loss = gan.train_on_batch(z, real)

            if (iteration + 1) % sample_interval == 0:
                self.lossess.append( ( d_loss, g_loss ) )
                self.accuracies.append( 100.0 * accuracy )
                self.iteration_checkpoints.append( iteration + 1 )
                print( "%d [D 손실: %f, 정확도: %.2f%%] [G 손실: %f]" % ( iteration + 1, d_loss, 100.0 * accuracy, g_loss ) )
                
                filename = 'test31-' + "%05d" % (iteration+1) + ".png"
                self.sample_images( axs, numrow, image_grid_columns )
                numrow += 1

        filename = "test31-image.png"
        plt.savefig( filename )

    def execute( self ) :
        return


    def sample_images( self, axs, numrow, image_grid_columns=16 ) :

        z = np.random.normal( 0, 1, (image_grid_columns, cfg_z_dim) )
        gen_imgs = self.generator.predict(z, verbose=3)
        gen_imgs = 0.5 * gen_imgs + 0.5

        for i in range(image_grid_columns):
            axs[numrow, i].imshow(gen_imgs[i, :, :, 0], cmap='gray')
            axs[numrow, i].axis('off')
        # plt.close(fig)



    def plot_training_history( self, filename="test31-histo.png") :
        losses = np.array( self.lossess )
        accuracies = np.array( self.accuracies ) / 100

        plt.figure(figsize=( 15, 5 ) )
        plt.plot( self.iteration_checkpoints, losses.T[0], label="Discriminator loss" )
        plt.plot( self.iteration_checkpoints, losses.T[1], label="Generator loss" )
        plt.plot( self.iteration_checkpoints, accuracies, label="Discriminator accuracy" )

        plt.xticks( self.iteration_checkpoints, rotation=90 )

        plt.title("Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.yscale( "log" )
        plt.legend()

        plt.savefig( filename )




cfg_img_rows = 28
cfg_img_cols = 28
cfg_channels = 1
cfg_img_shape = (cfg_img_rows, cfg_img_cols, cfg_channels)
cfg_z_dim = 100
gan = GAN( cfg_img_shape, cfg_z_dim )
gan.compile( loss=K.losses.BinaryCrossentropy(), optimizer=K.optimizers.Adam() )

iterations = 20000
batch_size = 128
sample_interval = 1000
gan.exec_train( iterations, batch_size, sample_interval )
gan.plot_training_history()
