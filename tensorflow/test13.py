from inspect import ArgSpec
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.datasets.mnist



img_rows = 28
img_cols = 28
channels = 1
img_shape = ( img_rows, img_cols, channels )
z_dim = 100


def build_generator( img_shape, z_dim ) :
    model = tensorflow.keras.models.Sequential()
    model.add( tensorflow.keras.layers.Dense( 128, input_dim=z_dim ) )
    model.add( tensorflow.keras.layers.LeakyReLU( alpha=0.01 ) )
    model.add( tensorflow.keras.layers.Dense( 28 * 28 * 1, activation='tanh' ) )
    model.add( tensorflow.keras.layers.Reshape( img_shape ) )
    return model

generator = build_generator( img_shape, z_dim )



def build_descriminator( img_shape, z_dim ) :
    model = tensorflow.keras.models.Sequential()
    model.add( tensorflow.keras.layers.Flatten( input_shape=img_shape ) )
    model.add( tensorflow.keras.layers.Dense( 128 ) )
    model.add( tensorflow.keras.layers.LeakyReLU( alpha=0.01 ) )
    model.add( tensorflow.keras.layers.Dense( 1, activation='sigmoid' ) )
    return model

discriminator = build_descriminator( img_shape, z_dim )






