import numpy as np
import matplotlib.pyplot as plt


def sample_images( generator, z_dim, image_grid_rows=4, image_grid_cols=6, id=00 ) :
    z = np.random.normal( 0, 1, (image_grid_rows * image_grid_cols, z_dim) )
    gen_imgs = generator.predict(z,verbose=3)
    
    plt.clf()
    fig, axs = plt.subplots( image_grid_rows, image_grid_cols, figsize=(4,4), sharey=True, sharex=True )
    #axs = np.expand_dims( axs, axis=0 )

    cnt = 0
    for i in range( image_grid_rows ) :
        for j in range( image_grid_cols ) :
            axs[i,j].imshow( gen_imgs[ cnt, :, :, 0 ], cmap='gray' )
            axs[i,j].axis('off')
            cnt += 1
    plt.savefig( 'test' + str(id).zfill(3) + '.png' )
