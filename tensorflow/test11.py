import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


data = pd.read_csv( '../data/Admission_Predict_Ver1.1.csv' )
print( data.head() )

continuous_features = data[ ['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA'] ].values / 100 
categorical_research_features = data[ [ 'Research' ] ].values 
X = np.concatenate( [ continuous_features , categorical_research_features ] , axis=1 )
Y = data[ [ 'Chance of Admit ' ] ].values

train_features , test_features ,train_labels, test_labels = train_test_split( X , Y , test_size=0.2 )

X = tf.constant( train_features , dtype=tf.float32 )
Y = tf.constant( train_labels , dtype=tf.float32 ) 
test_X = tf.constant( test_features , dtype=tf.float32 ) 
test_Y = tf.constant( test_labels , dtype=tf.float32 ) 


def mean_squared_error( Y, y_pred ) :
    return tf.reduce_mean( tf.square( y_pred - Y ) )

def mean_squared_error_deriv( Y, y_pred ) :
    return tf.reshape( tf.reduce_mean( 2 * ( y_pred - Y ) ), [ 1, 1 ] )

def h( X, weights, bias ) :
    return tf.tensordot( X, weights, axis = 1 ) + bias


