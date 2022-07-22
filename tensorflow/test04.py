import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns



url = '~/work/ML/data/auto-mpg.data.txt'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv( url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True )
dataset = raw_dataset.copy()
print( dataset.tail() )
print( dataset.isna().sum() )
dataset = dataset.dropna()

dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
print( dataset.tail() )

dataset_train = dataset.sample( frac=0.8, random_state=0 )
dataset_test = dataset.drop( dataset_train.index )


features_train = dataset_train.copy()
features_test = dataset_test.copy()
labels_train = features_train.pop('MPG')
labels_test = features_test.pop('MPG')

print( dataset_train.describe().transpose()[['mean', 'std']] )



normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(features_train))



linear_model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense( units=1 )
])

linear_model.predict( features_train[:10] )
