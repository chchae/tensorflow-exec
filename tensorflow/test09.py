import sys
import numpy as np
import tensorflow.keras
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
import matplotlib.pyplot as plt


np.set_printoptions( precision=2, suppress=True, threshold=sys.maxsize,linewidth=160 )

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
# print( np.column_stack( ( x_train, y_train ) ) )


x_train_scaled = preprocessing.scale( x_train )
scaler = preprocessing.StandardScaler().fit( x_train )
x_test_scaled = scaler.transform( x_test )



def make_model() :
    model = Sequential()
    model.add( Dense( 64, activation='relu', input_shape=(13,) ) )
    model.add( Dense( 64, activation='relu' ) )
    model.add( Dense( 1, activation='relu' ) )
    return model


model = make_model()
model.compile( loss='mse', optimizer=tensorflow.keras.optimizers.RMSprop(), metrics=['mean_absolute_error'] )
model.fit( x_train_scaled, y_train, epochs=2000, batch_size=128, verbose=0, validation_split=0.2 )

score = model.evaluate( x_test_scaled, y_test, verbose=0 )
print( 'Test loss: ', score[0], ' Test accuracy: ', score[1] )

prediction = model.predict( x_test_scaled )
# print( np.column_stack( ( prediction.flatten(), y_test ) ) )


def plot_scatter( y, yhat, fname ) :
    # plt.subplot( 2, 1, 2 )
    plt.clf()
    #plt.figure( figsize=(10,10) )
    # plt.axis( np.append( axis, axis ) )
    #plt.plot( axis, axis, color="red" )
    plt.title( 'Scatter plot')
    plt.scatter( y, yhat, color="blue", label="prediction" )
    plt.legend()
    plt.savefig( fname )
    # plt.show()

plot_scatter( y_test, prediction.flatten(), 'test09.png' )



