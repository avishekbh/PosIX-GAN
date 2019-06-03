from keras import backend as k
from scipy import misc
from keras.preprocessing import image
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
# from keras.models import load_weights
import keras
import models
import train
import utils
import os
from keras.utils import plot_model

img_size = 64  # Size of square image
channels = 3  # 1 for grayscale
z = 256  # Generator input


def create_inputs_headpose():
    ts_x, ts_y = load_headpose()
    ts_x = ts_x.astype(np.float32) / 255.
    ts_x = np.rollaxis(ts_x, 3, 1)
    ts_y = ts_y.astype(np.float32)
    return (ts_x, ts_y)


def load_headpose():
    tsImages = np.load('tsImages.npy')
    tsLabel = np.load('tsLabel.npy')
    return tsImages, tsLabel


generator = models.b_generator(z, img_size, channels)
print('generator summary')
print (generator.summary())
discriminator = models.b_discriminator(img_size, channels)
print('discriminator summary')
print (discriminator.summary())
gan = models.generator_containing_discriminator(generator, discriminator)

print (gan.summary())

generator.load_weights('gen_epoch800.h5') # load weights of the generator model
discriminator.load_weights('disc_epoch800.h5') # load weights of the discriminator model
gan = models.generator_containing_discriminator(generator, discriminator)
tsImages, y_true = create_inputs_headpose()
print("reached here")
[_, gen_0, _, gen_1, _, gen_2, _, gen_3, _, gen_4, _, gen_5, _, gen_6, _, gen_7, _, gen_8] = gan.predict(tsImages)
y_pred = np.mean([gen_0, gen_1, gen_2, gen_3, gen_4, gen_5, gen_6, gen_7, gen_8], axis=0) # You can also use max-voting, gives equivalent results
print(np.equal(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1)).mean()) # Accuracy

