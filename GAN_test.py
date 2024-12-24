import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os

current_dir = os.chdir("C:/Users/knkas/Desktop/GAN")

#=================================================================================

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# convert pixel values to -1 to 1
x_train = x_train/255.0*2-1
x_test = x_test/255.0*2-1

print("x_train.shape", np.shape(x_train) )

N, H, W = np.shape(x_train)  # (# of sample, image height, image width)  
D = H*W 

x_train = x_train.reshape( -1, D )
x_test = x_test.reshape( -1, D )

# dimension of the latent space (you can choose this to be any)
# input dimensionality of the image generator.
latent_dim = 100

def build_generator(latent_dim):
    i = Input( shape=(latent_dim, ))
    x = Dense( 256, activation=LeakyReLU(alpha=0.2))(i)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense( 512, activation=LeakyReLU(alpha=0.2))(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense( 1024, activation=LeakyReLU(alpha=0.2))(x)
    x = BatchNormalization(momentum=0.8)(x)
    out = Dense( D, activation='tanh')(x)   # we use tanh because the image pixels are -1 to 1.
    model = Model( i, out )  # output shape is (batch, D). Shape needs to be the same as the traing data.
    return model
    
def build_discriminator(img_size):
    i = Input( shape=(img_size, ) )
    x = Dense( 512, activation=LeakyReLU(alpha=0.2))(i)
    x = Dense( 256, activation=LeakyReLU(alpha=0.2))(x)
    out = Dense( 1, activation='sigmoid' )(x)
    model = Model( i, out )
    return model

discriminator_model = build_discriminator(D)
discriminator_model.compile( loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
generator_model = build_generator( latent_dim )

#---- below create combined model ----------------------
# create image from sample noise
z = Input( shape=(latent_dim, ) )

# the true output is fake, but we treat them as real!
img = generator_model( z )
fake_pred = discriminator_model(img)

# make sure only generator is trained
discriminator_model.trainable = False

# create the combined model
combined_model = Model( inputs=z, outputs=fake_pred )
combined_model.compile( loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5) )
#--------------------------------------------------------

batch_size = 32
epochs = 100  #default=30000
sample_period = 10 # every sample_period, generate and save some data. default=200

# create folder to store images
if not os.path.exists('gan_images'):
    os.makedirs('gan_images')
    
# generate 5x5 grid images of graph
def sample_images(epoch):
    rows, cols = 5, 5
    noise = np.random.randn(rows*cols, latent_dim)
    imgs = generator_model.predict(noise)
    
    # rescale images to 0 - 1 pixel values. (originally, it was between -1 to 1)
    imgs = 0.5*imgs + 0.5
    
    fig, axs = plt.subplots( rows, cols )
    idx = 0
    for i in range(rows):
        for j in range(cols):
            axs[i,j].imshow(imgs[idx].reshape(H,W), cmap='gray' )
            axs[i,j].axis('off')
            idx += 1
    fig.savefig("gan_images/%d.png" % epoch )
    plt.close()
    
    
# create batch labels to use when calling train_on_batch
ones = np.ones( batch_size )
zeros = np.zeros( batch_size )

# train the model
d_losses = []
g_losses = []    
for epoch in range(epochs):
    
    # first train discriminator
    idx = np.random.randint( 0, x_train.shape[0], batch_size )
    real_imgs = x_train[idx]  # select random batch of images for training.
    noise = np.random.randn( batch_size, latent_dim) # create fake images from random numbers
    fake_imgs = generator_model.predict(noise)
    
    d_loss_real, d_acc_real = discriminator_model.train_on_batch( real_imgs, ones )
    d_loss_fake, d_acc_fake = discriminator_model.train_on_batch( fake_imgs, zeros )
    d_loss = 0.5*(d_loss_real + d_loss_fake)
    d_acc = 0.5*(d_acc_real + d_acc_fake) 


    # second, train generator
    noise = np.random.randn(batch_size, latent_dim)
    g_loss = combined_model.train_on_batch( noise, ones )
    
    d_losses.append(d_loss)
    g_losses.append(g_loss)
    
    if epoch % 10==0:  # default=100
        print(f"epoch:{epoch+1}/{epochs}, d_loss:{d_loss:.2f}, d_acc:{d_acc:.2f}, g_loss:{g_loss:.2f}")
        
    if epoch % sample_period == 0:
        sample_images(epoch)
        


import pdb; pdb.set_trace()  

