from ast import increment_lineno
import os, sys
import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.compat.v1.disable_v2_behavior()


def reset_graph( seed=42 ) :
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)

# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
matplotlib.rc('font', family='NanumBarunGothic') # Linux
plt.rcParams['axes.unicode_minus'] = False



def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")
    
def plot_multiple_images(images, n_rows, n_cols, pad=2):
    images = images - images.min()  # 최소값을 0으로 만들어 패딩이 하얗게 보이도록 합니다.
    w,h = images.shape[1:]
    image = np.zeros(((w+pad)*n_rows+pad, (h+pad)*n_cols+pad))
    for y in range(n_rows):
        for x in range(n_cols):
            image[(y*(h+pad)+pad):(y*(h+pad)+pad+h),(x*(w+pad)+pad):(x*(w+pad)+pad+w)] = images[y*n_cols+x]
    plt.imshow(image, cmap="Greys", interpolation="nearest")
    plt.axis("off")






import numpy.random as rnd


def make_data() :
    rnd.seed(4)
    m = 200
    w1, w2 = 0.1, 0.3
    noise = 0.1

    angles = rnd.rand(m) * 3 * np.pi / 2 - 0.5
    data = np.empty( ( m, 3 ) )
    data[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * rnd.randn(m)/2
    data[:, 1] = np.sin(angles) * 0.7 + noise * rnd.randn(m)/2
    data[:, 2] = data[:, 0 ] * w1 + data[:, 1] * w2 + noise * rnd.randn(m)/2
    return data


from sklearn.preprocessing import StandardScaler
data = make_data()
scaler = StandardScaler()
X_train = scaler.fit_transform( data[:100] )
X_test = scaler.transform( data[100:] )
# print( X_train, X_test )



reset_graph()
n_input = 3
n_hidden = 2
n_output = n_input


#tf.compat.v1.disable_v2_behavior()
#X = tf.compat.v1.placeholder( shape=[None, n_input], dtype=tf.float32  )
#hidden = tf.compat.v1.layers( X, n_hidden )
#output = tf.compat.v1.layers.dense( hidden, n_output )

#X = tf.Variable( tf.ones( shape=[None, n_input] ), dtype=np.float32  )
#X = tf.Variable( ( np.empty((0,3), dtype=np.float32)), shape=[None, n_input] )
#X = tf.keras.Input( shape=[None, n_input], dtype=np.float32  )
#hidden = tf.keras.layers.Dense(units=n_hidden)( X )
#output = tf.keras.layers.Dense(units=n_output)( hidden )


learning_rate = 0.01
n_iteraction = 1000
#pca = hidden

if False:
    #reconstruction_loss = tf.math.reduce_mean( tf.square( output - X ) )
    reconstruction_loss = tf.keras.losses.mse( output, X )
    train_op = tf.keras.optimizers.Adam(learning_rate).minimize( reconstruction_loss, var_list=[hidden] )



model = tf.keras.Sequential([
tf.keras.layers.Dense( n_input, input_shape=(None,n_input) ),
tf.keras.layers.Dense( n_hidden ),
tf.keras.layers.Dense( n_output )
])
model.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse' )
print( model.summary() )

model.fit( X_train, )







