import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn.model_selection as sk
import matplotlib.pyplot as plt



def prepare_delaney_data() :
    path = "../data/delaney-processed.csv"
    df = pd.read_csv( path )
    x, y = df.values[:, 2:8].astype(np.float32), df.values[:, 8].astype(np.float32)
    x_train, x_test, y_train, y_test = sk.train_test_split( x, y, test_size=0.33 )
    print( x_train.shape, x_test.shape, y_train.shape, y_test.shape )
    return x_train, x_test, y_train, y_test


def plot_history( hist, fname ) :
    plt.subplot( 2, 1, 1 )
    #plt.clf()
    plt.title( 'Learning curves')
    plt.xlabel( 'Epoch' )
    plt.ylabel( 'RMSE' )
    plt.plot( hist.history['loss'], label='train' )
    plt.plot( hist.history['val_loss'], label='validation' )
    plt.axis( [ 0, len( hist.history['loss'] ), 0, 5 ] )
    plt.legend()
    #plt.savefig( fname )



def plot_scatter( y, yhat, axis, fname ) :
    plt.subplot( 2, 1, 2 )
    #plt.clf()
    #plt.figure( figsize=(10,10) )
    plt.axis( np.append( axis, axis ) )
    plt.plot( axis, axis, color="red" )
    plt.title( 'Scatter plot')
    plt.scatter( y, yhat, color="blue", label="prediction" )
    plt.legend()
    #plt.savefig( fname )



def make_sequential_model( n_features ) :
    model = tf.keras.Sequential()
    model.add( tf.keras.layers.Dense( 10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,) ) )
    model.add( tf.keras.layers.Dense( 8, activation='relu', kernel_initializer='he_normal' ) )
    model.add( tf.keras.layers.Dense( 1 ) )
    model.compile( optimizer='adam', loss='mse' )
    return model
    

def make_sequential_model2( n_features ) :
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input( shape=[n_features] ),
        tf.keras.layers.Dense( 8, activation='relu' ),
        tf.keras.layers.Dense( 1 )
    ])
    return model



def simple_linear_ligression() :
    x_train, x_test, y_train, y_test = prepare_delaney_data()

    n_features = x_train.shape[1]
    model = make_sequential_model( n_features )
    model.compile( loss='mse', optimizer='adam', metrics='mean_squared_error' )
    hist = model.fit( x_train, y_train, epochs=100, batch_size=16, verbose=1, validation_split=0.3 )
    
    error = model.evaluate( x_test, y_test, verbose=0 )
    print( 'MSE: %.3f, RMSE: %.3f' % ( error, np.sqrt(error) ) )
    plot_history( hist, "test02-hist.png" )

    yhat = model.predict( x_test )
    plot_scatter( y_test, yhat, [-10, 2], "test02-scatter.png" )




def simple_linear_ligression2() :
    x_train, x_test, y_train, y_test = prepare_delaney_data()
    
    n_features = x_train.shape[1]
    model = make_sequential_model2( n_features )
    model.compile( loss='mse', optimizer='adam', metrics='mean_squared_error' )
    hist = model.fit( x_train, y_train, epochs=100, verbose=1, validation_split=0.3 )
    plot_history( hist, "test02-hist.png" )

    yhat = model.predict( x_test )
    plot_scatter( y_test, yhat, [-10, 2], "test02-scatter2.png" )

    plt.tight_layout()
    plt.savefig( "test02-scatter2.png" )


if __name__ == "__main__" :
    simple_linear_ligression2()
    
    



