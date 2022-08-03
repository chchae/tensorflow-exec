import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense


dataset = np.loadtxt( "../data/pima-indians-diabetes.csv", delimiter=',' )
X = dataset[:, 0:8]
y = dataset[:,8]
print( dataset, X, y )


def make_model() :
    model = Sequential()
    model.add( Dense( 12, activation='relu', input_shape=(8,) ) )
    model.add( Dense( 8, activation='relu' ) )
    model.add( Dense( 1, activation='sigmoid' ) )
    return model


model = make_model()
model.compile( loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'] )
model.fit( X, y, epochs=150, batch_size=10, verbose=1 )




