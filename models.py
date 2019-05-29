import numpy as np
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dense, Dropout, Input,GaussianDropout, GaussianNoise, Activation
from keras.layers import Reshape, Conv2D
from keras.layers.advanced_activations import LeakyReLU, Softmax, ELU
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.layers.core import Flatten
import tensorflow as tf

from keras.models import Model
import numpy as np

from keras import backend as K


def l1Loss(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))    

def shape(depth, row, col):
    return (depth, row, col)

def decoder(h, img_dim, channels, n = 128):
    '''
    The decoder model is used as both half of the discriminator and as the generator.

    Keyword Arguments:
    h -- Integer size of the 1 dimensional input vector
    img_dim -- Integer size of the square output image
    channels -- 1 or 3 depending on whether the images have color channels or not.
    n -- Number of convolution filters, paper value is 128
    '''
    init_dim = 8 #Starting size from the paper
    layers = int(np.log2(img_dim) - 3)
    
    mod_input = Input(shape=(h,))
    x = Dense(n*init_dim**2)(mod_input)
    x = Reshape(shape(n, init_dim, init_dim))(x)

    x = Convolution2D(n, (3, 3), strides=1, border_mode="same", data_format="channels_first" )(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Convolution2D(n, (3, 3), strides=1, border_mode="same", data_format="channels_first" )(x)
    x = LeakyReLU(alpha=0.1)(x)
    print('x shape ', np.shape(x))
    for i in range(layers):
        x = UpSampling2D(size=(2, 2), data_format="channels_first")(x)
        x = Convolution2D(n, (3, 3), strides=1, border_mode="same", data_format="channels_first",
                          kernel_initializer='glorot_uniform')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Convolution2D(n, (3, 3), strides=1, border_mode="same", data_format="channels_first",
                          kernel_initializer='glorot_uniform')(x)
        x = LeakyReLU(alpha=0.1)(x)
        print('x shape in loop ', np.shape(x))
        
    x = Convolution2D(channels, (3, 3), strides=1, border_mode="same", data_format="channels_first" )(x)
    x = ELU()(x)
    
    return Model(mod_input, x)


def encoder(h, img_dim, channels, n = 128):
    '''
    The encoder model is the inverse of the decoder used in the autoencoder.

    Keyword Arguments:
    h -- Integer size of the 1 dimensional input vector
    img_dim -- Integer size of the square output image
    channels -- 1 or 3 depending on whether the images have color channels or not.
    n -- Number of convolution filters, paper value is 128
    '''
    init_dim = 8
    layers = int(np.log2(img_dim) - 2)
    
    mod_input = Input(shape=shape(channels, img_dim, img_dim))
    x = Convolution2D(channels, (3, 3), strides=1, border_mode="same", data_format="channels_first" )(mod_input)
    x = LeakyReLU(alpha=0.1)(x)
    
    for i in range(1, layers):
        x = Convolution2D(i*n, (3, 3), strides=1, border_mode="same", data_format="channels_first",
                          kernel_initializer='glorot_uniform')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Convolution2D(i*n, (3, 3), border_mode="same", subsample=(2,2), data_format="channels_first",
                          kernel_initializer='glorot_uniform')(x)
        x = LeakyReLU(alpha=0.1)(x)
    
    x = Convolution2D(layers*n, (3, 3), strides=1, border_mode="same", data_format="channels_first" )(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Convolution2D(layers*n, (3, 3), strides=1, border_mode="same", data_format="channels_first" )(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Reshape((layers * n * init_dim ** 2,))(x)
    x = Dense(h)(x)
    
    return Model(mod_input,x)


def autoencoder1(h, img_dim, channels, n = 128):
    '''
    The autoencoder is used as the discriminator

    Keyword Arguments:
    h -- Integer size of the 1 dimensional input vector
    img_dim -- Integer size of the square output image
    channels -- 1 or 3 depending on whether the images have color channels or not.
    n -- Number of convolution filters, paper value is 128
    '''
    mod_input = Input(shape=shape(channels, img_dim, img_dim))
    x = encoder(h, img_dim, channels, n)(mod_input)
    x = decoder(h, img_dim, channels, n)(x)
    
    return Model(mod_input, x)


def b_generator(h, img_dim, channels, n=64):
    '''
    The autoencoder is used as the discriminator

    Keyword Arguments:
    h -- Integer size of the 1 dimensional input vector
    img_dim -- Integer size of the square output image
    channels -- 1 or 3 depending on whether the images have color channels or not.
    n -- Number of convolution filters, paper value is 128
    '''
    mod_input = Input(shape=shape(channels, img_dim, img_dim))
    x = encoder(h, img_dim, channels, n)(mod_input)
    g1 = decoder(h, img_dim, channels, n)(x)
    g2 = decoder(h, img_dim, channels, n)(x)
    g3 = decoder(h, img_dim, channels, n)(x)
    g4 = decoder(h, img_dim, channels, n)(x)
    g5 = decoder(h, img_dim, channels, n)(x)
    g6 = decoder(h, img_dim, channels, n)(x)
    g7 = decoder(h, img_dim, channels, n)(x)
    g8 = decoder(h, img_dim, channels, n)(x)
    g9 = decoder(h, img_dim, channels, n)(x)
    # shp = K.int_shape(g1)
    # d1 = discriminator_model(shp)
    # d2 = discriminator_model(shp)
    # d3 = discriminator_model(shp)
    # d4 = discriminator_model(shp)
    # d5 = discriminator_model(shp)
    # d6 = discriminator_model(shp)
    # d7 = discriminator_model(shp)
    # d8 = discriminator_model(shp)
    # d9 = discriminator_model(shp)
    return Model(mod_input, [g1, g2, g3, g4, g5, g6, g7, g8, g9])


def b_discriminator(img_dim, channels):
    '''
    The autoencoder is used as the discriminator

    Keyword Arguments:
    h -- Integer size of the 1 dimensional input vector
    img_dim -- Integer size of the square output image
    channels -- 1 or 3 depending on whether the images have color channels or not.
    n -- Number of convolution filters, paper value is 128
    '''
    mod_input1 = Input(shape=shape(channels, img_dim, img_dim))
    mod_input2 = Input(shape=shape(channels, img_dim, img_dim))
    mod_input3 = Input(shape=shape(channels, img_dim, img_dim))
    mod_input4 = Input(shape=shape(channels, img_dim, img_dim))
    mod_input5 = Input(shape=shape(channels, img_dim, img_dim))
    mod_input6 = Input(shape=shape(channels, img_dim, img_dim))
    mod_input7 = Input(shape=shape(channels, img_dim, img_dim))
    mod_input8 = Input(shape=shape(channels, img_dim, img_dim))
    mod_input9 = Input(shape=shape(channels, img_dim, img_dim))
    shp = [channels, img_dim, img_dim]

    [d1, c1] = discriminator_model(shp)(mod_input1)
    [d2, c2] = discriminator_model(shp)(mod_input2)
    [d3, c3] = discriminator_model(shp)(mod_input3)
    [d4, c4] = discriminator_model(shp)(mod_input4)
    [d5, c5] = discriminator_model(shp)(mod_input5)
    [d6, c6] = discriminator_model(shp)(mod_input6)
    [d7, c7] = discriminator_model(shp)(mod_input7)
    [d8, c8] = discriminator_model(shp)(mod_input8)
    [d9, c9] = discriminator_model(shp)(mod_input9)
    # d1 = discriminator_model(shp)
    # outputs_d1 = d1(mod_input1)
    # d2 = discriminator_model(shp)
    # outputs_d2 = d2(mod_input1)
    # d3 = discriminator_model(shp)
    # outputs_d3 = d3(mod_input1)
    # d4 = discriminator_model(shp)
    # outputs_d4 = d4(mod_input1)
    # d5 = discriminator_model(shp)
    # outputs_d5 = d5(mod_input1)
    # d6 = discriminator_model(shp)
    # outputs_d6 = d6(mod_input1)
    # d7 = discriminator_model(shp)
    # outputs_d7 = d7(mod_input1)
    # d8 = discriminator_model(shp)
    # outputs_d8 = d8(mod_input1)
    # d9 = discriminator_model(shp)
    # outputs_d9 = d9(mod_input1)
    # [d1, c1] = x1.outputs
    # [d2, c2] = x2.outputs
    # [d3, c3] = x3.outputs
    # [d4, c4] = x4.outputs
    # [d5, c5] = x5.outputs
    # [d6, c6] = x6.outputs
    # [d7, c7] = x7.outputs
    # [d8, c8] = x8.outputs
    # [d9, c9] = x9.outputs

    # [d1, c1] = x1
    # [d2, c2] = x2
    # [d3, c3] = x3
    # [d4, c4] = x4
    # [d5, c5] = x5
    # [d6, c6] = x6
    # [d7, c7] = x7
    # [d8, c8] = x8
    # [d9, c9] = x9

    print(type(d1))
    return Model([mod_input1, mod_input2, mod_input3, mod_input4, mod_input5, mod_input6, mod_input7, mod_input8,
                  mod_input9], [d1, c1, d2, c2, d3, c3, d4, c4, d5, c5, d6, c6, d7, c7, d8, c8, d9, c9])


def generator_containing_discriminator(g, d):
    # model = Sequential()
    # input = Input(g.inputs)
    # x = g()(input)
    d.trainable = False
    # x = d()(x)
    print('g inputs ', np.shape(g.inputs[0]))
    model = Model(inputs=g.inputs, outputs=d(g.outputs))
    return model


def gan(generator, discriminator):
    '''
    Combined generator and discriminator

    Keyword arguments:
    generator -- The instantiated generator model
    discriminator -- The instantiated discriminator model
    '''
    mod_input = generator.input
    x = generator(mod_input)
    x = discriminator(x)

    return Model(mod_input, x)


def discriminator_model(shp):
    # model = Graph()
    input = Input(shape=shp)
    x = Convolution2D(64, (5, 5), border_mode='same', data_format="channels_first" )(input)
    x = Activation('tanh')(x)
    x = MaxPooling2D((2, 2), data_format="channels_first")(x)
    x = Convolution2D(128, (5, 5), border_mode='same', data_format="channels_first" )(x)
    x = Activation('tanh')(x)
    # x = MaxPooling2D(2, 2)(x)
    x = Flatten()(x)
    x = Dense(1024, activation='tanh')(x)
    # x = Activation('tanh')(x)
    d = Dense(1, activation='sigmoid')(x)
    # d = Activation('sigmoid')(d)
    cls = Dense(15)(x)
    x = Softmax()(x)
    model = Model(inputs=input, outputs=[d, cls])
    return model
