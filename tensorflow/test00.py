import numpy as np
# import tensorflow as tf




if True:
    a = np.random.normal( 0, 1, (1, 3) )
    print( a.shape, a )
















if False :
    d0 = tf.ones( (1,) )
    #print( d0 )

    d1 = tf.ones( (2,) )
    #print( d1 )

    d2 = tf.ones( (2, 2) )
    #print( d2 )

    d3 = tf.ones( (2,2,2) )
    #print( d3 )

if False:
    a = np.zeros( [2, 3] )
    print(a)

# print( np.empty((2,3)) )

if False:
    # b = tf.placeholder( tf.float32, [None, 128] )
    b = tf.Variable( tf.zeros([16,2]), shape=(None,2), validate_shape=False )
    print( b.numpy() )


if False:
    A = tf.constant( [ [3,2], [5,3] ], dtype=tf.float32 )
    print(A.numpy())
    B = tf.Variable( [ [3,2], [5,3] ], dtype=tf.float32 )
    print(B.numpy())

    AB = tf.concat( values=[A,B], axis=1 )
    print( AB.numpy() )


    A = tf.zeros( shape=[3,4], dtype=tf.float32 )
    print(A.numpy())
    print( tf.reshape( A, shape=[2,6] ).numpy() )
    print( tf.cast( A, dtype=tf.int32 ) )
    
    
    #tf.transpose( ma )
    #tf.matmul( ma, mb )
    #tf.multiply( m, f )
    #tf.determinant( m )
    
    
if False:
    model = tf.keras.models.Sequential()
    model.add( tf.keras.layers.Dense( units=2, input_shape=(2,) ) )
    
    
    
####################################### test01






