import tensorflow as tf

print( "Tensorflow version : ", tf.__version__ )

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print( x_train, x_test )

if True :
    x_train, x_test = x_train / 255.0, x_test / 255.0
else :
    x_train = tf.convert_to_tensor( x_train, dtype='float' )
    x_test  = tf.convert_to_tensor( x_test, dtype='float' )
    y_train = tf.one_hot( y_train, depth=10 )
    y_test  = tf.one_hot( y_test, depth=10 )


model = tf.keras.models.Sequential( [
    tf.keras.layers.Flatten( input_shape=(28,28) ),
    tf.keras.layers.Dense( 128, activation='relu' ),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

predictions = model( x_train[:1]).numpy()
print( predictions )
print( tf.nn.softmax( predictions ).numpy() )

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy( from_logits=True )
print( loss_fn( y_train[:1], predictions ).numpy() )

model.compile( optimizer='adam', loss=loss_fn, metrics=['accuracy'] )
hist=model.fit( x_train, y_train, epochs=20, validation_split=0.3 )
model.evaluate( x_test, y_test, verbose=2 )



import matplotlib.pyplot as plt
plt.title( 'Learning curves')
plt.xlabel( 'Epoch' )
plt.ylabel( 'RMSE' )
plt.plot( hist.history['loss'], label='train' )
plt.plot( hist.history['val_loss'], label='validation' )
plt.legend()
plt.show()




