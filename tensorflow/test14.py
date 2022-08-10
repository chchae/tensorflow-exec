# https://realpython.com/generative-adversarial-networks/
# https://towardsdatascience.com/demystifying-gans-in-tensorflow-2-0-9890834ab3d9


import math
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt



train_data_length = 1024
train_data = np.zeros( ( train_data_length, 2 ) )
train_data[:, 0] = 2 * math.pi * np.random.random(train_data_length)
train_data[:, 1] = np.sin(train_data[:, 0])
train_labels = np.ones(train_data_length)
#train_set = [ (train_data[i], train_labels[i]) for i in range(train_data_length) ]
#print( train_set  )



def plot_graph( data ):
    plt.clf()
    #plt.axis( [ 0, 6, -1, 2 ] )
    plt.plot( data[:,0], data[:,1], "." )
    plt.savefig( "test14.png" )

plot_graph( train_data )



def train_loader( data, batch_size ) :
    idx = np.random.randint( 0, len(data), batch_size )
    # print( idx )
    dat = np.array(data)[idx]
    return dat

#data = train_loader( train_set, 5 )
#print( data )




img_rows = 1
img_cols = 1
channels = 1
img_shape = ( img_rows, img_cols, channels )
img_shape = ( 2 )
z_dim = 100


class Generator( K.Model ) :
    def __init__( self, img_shape, z_dim ) :
        super().__init__( name = "generator" )
        self.model = K.models.Sequential()
        self.model.add( K.layers.Dense( 128, input_dim=z_dim ) )
        self.model.add( K.layers.LeakyReLU( alpha=0.01 ) )
        self.model.add( K.layers.Dense( 2, activation='tanh' ) )
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
    #data = train_loader( train_data, 5 )
    #print( 'data : ', data )
    #X_train = data
    #print( 'x_train : ', X_train )

    real = np.ones( (batch_size, 1) )
    fake = np.zeros( (batch_size, 1) )

    for iteration in range(iterations) :
        imgs = train_loader( train_data, batch_size )
        print('imgs=', imgs.shape, imgs)

        z = np.random.normal( 0, 1, (batch_size, 100) )
        gen_imgs = generator.predict(z,verbose=3)
        print( 'gen_imgs=', gen_imgs.shape, gen_imgs )
        exit(0)

        d_loss_real = discriminator.train_on_batch( imgs, real )
        d_loss_fake = discriminator.train_on_batch( gen_imgs, fake )
        d_loss, accuracy = 0.5 * np.add( d_loss_real, d_loss_fake )
        accuracy = 0

        z = np.random.normal( 0, 1, (batch_size, 100) )
        gen_imgs = generator.predict(z,verbose=3)
        g_loss = gan.train_on_batch( z, real, )

        if( iteration + 1 ) % sample_interval == 0 :
            print( "%d [D loss: %f, accuracy : %.2f%%] [G loss: %f]" % (iteration + 1, d_loss, 100.0*accuracy, g_loss) )
            #sample_images( generator )
            print( np.reshape( gen_imgs, (128)) , gen_imgs.shape )
            myplot( gen_imgs, 'test14-'+str(iteration)+'.png' )


def sample_images( generator ) :
    #z = np.random.normal( 0, 1, (1, 100) )
    z = 2 * math.pi * np.random.random((1, 100))
    print(z.shape)
    print(z)
    gen_imgs = generator.predict(z,verbose=3)
    print(gen_imgs)

    plt.clf()
    # plt.axis( [ 0, 6, -1, 2 ] )
    plt.plot( z, gen_imgs, "." )
    plt.savefig( "test14a.png" )


def myplot( data, fname ) :
    plt.clf()
    plt.axis( [ 0, 6, -1, 2 ] )
    plt.plot( data,"." )
    plt.savefig( fname )





iterations = 2000
batch_size = 128
sample_interval = 100
train( iterations, batch_size, sample_interval )



