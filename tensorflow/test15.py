# https://wiserloner.tistory.com/1160


import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt


(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
print( train_images.shape )
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5
print( train_images.shape )

BUFFER_SIZE = 60000
BATCH_SIZE = 256
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print( train_dataset )



def make_generator_model():
    model = tf.keras.Sequential()
    model.add(K.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.LeakyReLU())

    model.add(K.layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # 주목: 배치사이즈로 None이 주어집니다.

    model.add(K.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.LeakyReLU())

    model.add(K.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.LeakyReLU())

    model.add(K.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')





def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(K.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(K.layers.LeakyReLU())
    model.add(K.layers.Dropout(0.3))

    model.add(K.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(K.layers.LeakyReLU())
    model.add(K.layers.Dropout(0.3))

    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(1))

    return model


discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)





cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

