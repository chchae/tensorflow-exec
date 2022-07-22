import numpy as np
import tensorflow as tf


class SimpleLinearRegression:
    def __init__( self, initializer='random' ) :
        if initializer == 'ones' :
            self.var = 1.
        elif initializer == 'zeros' :
            self.var = 0.
        elif initializer == 'random' :
            self.var = tf.random.uniform( shape=[], ninval=0., maxval=1. )
            
        self.m = tf.Variable( 1., shape=tf.TensorShape(None) )
        self.b = tf.Variable( self.var )
        
        
    def mse( self, true, predicted ) :
        return tf.reduce_mean( tf.square( true - predicted ) )
            

    def predict( self, x ) :
        return tf.reduce_sum( self.m * x, 1 ) + self.b
    
    
    def update( self, X, y, learning_rate ) :
        with tf.GradientTape( persistent=True ) as g :
            loss = self.mse( y, self.preeict(X) )
    


    
    
    
    
    