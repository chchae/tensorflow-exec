import numpy as np
import tensorflow as tf
import tensorflow.keras as K


class MyModel( K.Model ) :
    def __init__( self, z_dim ) :
        super().__init__( name='first' )
        self.model = K.models.Sequential()
        self.model.add( K.layers.Dense( 128, input_dim=z_dim ) )
        self.model.add( K.layers.LeakyReLU( alpha=0.01 ) )
        self.model.add( K.layers.Dense( 128 ) )

    def call( self, x ) :
        return self.model( x )



def training() :
    z_dim = 128000
    X_train = np.random.random( z_dim )
    real = np.ones( z_dim )
    print( X_train.shape, X_train )
    print( real.shape, real )

    mod = MyModel( z_dim )
    mod.compile( loss=K.losses.BinaryCrossentropy(), optimizer=K.optimizers.Adam() )

    for iter in range( z_dim ) :
        mod.train_on_batch( X_train, real )




# training()



def test1() :
    x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    y = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)
    model = tf.keras.Sequential([ K.layers.Dense(units=1, input_shape=[1] )])
    model.add( K.layers.Dense(units=1, input_shape=[12800] ) )
    model.add( K.layers.LeakyReLU( alpha=0.01 ) )
    model.add( K.layers.Dense(units=1, input_shape=[12800] ) )
    model.add( K.layers.LeakyReLU( alpha=0.01 ) )
    model.add( K.layers.Dense(units=1, activation='sigmoid' ) )

    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(x, y, epochs=50000,verbose=0)
    print(model.predict([10.0]))



def test2() :
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(12800, activation='relu'),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(12800, activation='relu'),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(12800, activation='relu'),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=500)

    model.evaluate(x_test,  y_test, verbose=0)

test2()
