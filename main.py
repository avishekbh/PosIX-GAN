from keras import backend as k
from scipy import misc
from keras.preprocessing import image
from keras.losses import mean_squared_error as loss_pmse
from keras.datasets import cifar10
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import keras
import models
import train
import utils
import os
from keras.utils import plot_model

# Base Code (BEGAN) link - https://github.com/pbontrager/BEGAN-keras
# As random initialization has been used for all models, and owing to size of the model itself, 
# stop and run code again if you do not see promising results in 250 epochs. 
# Generally, results are achieved in 50000 epochs.

# Training parameters
epochs = 1000000
batches_per_epoch = 3 # Change as much as required
batch_size = 15 # Do not change
gamma = .5  # between 0 and 1
# make images in form of (channels, col, row) as base code supoorted Theano backend
def create_inputs_headpose():
    tr_x, tr_y, ts_x, ts_y = load_headpose()
    tr_x = tr_x.astype(np.float32) / 255.
    tr_x = np.rollaxis(tr_x, 3, 1)
    tr_y = tr_y.astype(np.float32)
    ts_x = ts_x.astype(np.float32) / 255.
    ts_x = np.rollaxis(ts_x, 3, 1)
    ts_y = ts_y.astype(np.float32)
    print(np.shape(tr_x))
    return (tr_x, tr_y), (ts_x, ts_y)


# Load datasets, default image size 64 x 64
def load_headpose():
    trImages = np.load('trImages.npy')
    trLabel = np.load('trLabel.npy')
    tsImages = np.load('tsImages.npy')
    tsLabel = np.load('tsLabel.npy')
    return trImages, trLabel, tsImages, tsLabel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# image parameters
img_size = 64  # Size of square image
channels = 3  # 1 for grayscale
(X_train, y_train), (X_test, y_test) = create_inputs_headpose()
print(np.shape(X_train))
# Model parameters
z = 256  # Generator input
h = 256  # Autoencoder hidden representation
adam = Adam(lr=0.00005)  # lr: between 0.0001 and 0.00005

def combo_Loss(y_true, y_pred):
    return 0.7 * k.mean(k.square(y_pred - y_true), axis=-1) + 0.3 * k.mean(k.abs(y_true - y_pred))

# Loss slightly modified for faster convergence
def mse(y_true, y_pred):
    return 0.1 * k.mean(k.square(y_pred - y_true), axis=-1)
    
# Loss slightly modified for faster convergence
def patchwise_mse(y_true, y_pred):
    y_true_cl = k.permute_dimensions(y_true, (0, 2, 3, 1))
    y_pred_cl = k.permute_dimensions(y_pred, (0, 2, 3, 1))
    ch1_t, ch2_t, ch3_t = tf.split(y_true_cl, [1, 1, 1], axis=3)
    ch1_p, ch2_p, ch3_p = tf.split(y_pred_cl, [1, 1, 1], axis=3)
    chkp = loss_pmse(y_true, y_pred)
    kernel = [1, 11, 11, 1]
    strides = [1, 5, 5, 1]
    padding = 0
    patches_true_1 = tf.extract_image_patches(ch1_t, kernel, strides, [1, 1, 1, 1], padding='SAME')
    patches_true_2 = tf.extract_image_patches(ch2_t, kernel, strides, [1, 1, 1, 1], padding='SAME')
    patches_true_3 = tf.extract_image_patches(ch3_t, kernel, strides, [1, 1, 1, 1], padding='SAME')
    patches_pred_1 = tf.extract_image_patches(ch1_p, kernel, strides, [1, 1, 1, 1], padding='SAME')
    patches_pred_2 = tf.extract_image_patches(ch2_p, kernel, strides, [1, 1, 1, 1], padding='SAME')
    patches_pred_3 = tf.extract_image_patches(ch3_p, kernel, strides, [1, 1, 1, 1], padding='SAME')
    loss_1 = 0.0
    loss_2 = 0.0
    loss_3 = 0.0
    # Use value 22 instead of 12 for setting stride to 3, 3. -> leads to slower convergence but marginally better results.
    for i in range(12):
        for j in range(12):
            loss_1 = loss_1 + 0.02989 * mse(patches_true_1[0,i,j,], patches_pred_1[0,i,j])
            loss_2 = loss_2 + 0.05870 * mse(patches_true_2[0,i,j,], patches_pred_2[0,i,j])
            loss_3 = loss_3 + 0.01141 * mse(patches_true_3[0,i,j,], patches_pred_3[0,i,j])
    total_loss = chkp + ((loss_1 + loss_2 + loss_3)/(12*12))
    return total_loss

# Build models
generator = models.b_generator(z, img_size, channels)
print('generator summary')
print (generator.summary())
discriminator = models.b_discriminator(img_size, channels)
print('discriminator summary')
print (discriminator.summary())
gan = models.generator_containing_discriminator(generator, discriminator)

print (gan.summary())

generator.compile(loss=[patchwise_mse, patchwise_mse, patchwise_mse, patchwise_mse, patchwise_mse, patchwise_mse, patchwise_mse, patchwise_mse, patchwise_mse], optimizer=adam)
discriminator.compile(loss=['binary_crossentropy',
                  'categorical_crossentropy', 'binary_crossentropy',
                  'categorical_crossentropy', 'binary_crossentropy',
                  'categorical_crossentropy', 'binary_crossentropy',
                  'categorical_crossentropy', 'binary_crossentropy',
                  'categorical_crossentropy', 'binary_crossentropy',
                  'categorical_crossentropy', 'binary_crossentropy',
                  'categorical_crossentropy', 'binary_crossentropy',
                  'categorical_crossentropy', 'binary_crossentropy',
                    'categorical_crossentropy'], optimizer=adam)
gan.compile(loss=['binary_crossentropy',
                  'categorical_crossentropy', 'binary_crossentropy',
                  'categorical_crossentropy', 'binary_crossentropy',
                  'categorical_crossentropy', 'binary_crossentropy',
                  'categorical_crossentropy', 'binary_crossentropy',
                  'categorical_crossentropy', 'binary_crossentropy',
                  'categorical_crossentropy', 'binary_crossentropy',
                  'categorical_crossentropy', 'binary_crossentropy',
                  'categorical_crossentropy', 'binary_crossentropy',
                  'categorical_crossentropy'], optimizer=adam)



# Load data
(X_train, y_train), (X_test, y_test) = create_inputs_headpose()
# print(X_train)
# X_train = (X_train.astype(np.float32)-127.5) / 127.5
# X_test = (X_test.astype(np.float32)-127.5) / 127.5
dataGenerator = image.ImageDataGenerator()
batchIterator = dataGenerator.flow(X_train, y_train, batch_size=batch_size)
#
# plot_model(generator, to_file='genmodel.png')
# plot_model(discriminator, to_file='discmodel.png')
# plot_model(gan, to_file='ganmodel.png')
trainer = train.GANTrainer(generator, discriminator, gan, batchIterator, batch_size)

trainer.train(epochs, batches_per_epoch, batch_size, gamma)
