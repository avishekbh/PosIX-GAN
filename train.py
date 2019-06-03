import os
import time
import utils
import models
import keras
from keras.preprocessing import image
import numpy as np
from keras.utils import generic_utils
from keras.losses import mean_squared_error as loss_pmse
from keras import backend as k
import numpy.linalg as LA
from PIL import Image
import tensorflow as tf
import math
import os

class GANTrainer:
    def __init__(self, gen, disc, gan, data, batch_size, kLambda=.001, logEpochOutput=True, saveModelFrequency=200,
                 sampleSwatch=True, saveSampleSwatch=False):
        # '''
        # Class contains all the default values for training a particular GAN
        #
        # Keyword Arguments:
        # gen -- Generator Model to be trained
        # disc -- Discriminator Model to be trained
        # gan -- Combined Generator and Discriminator Model
        # data -- DataGenerator that outputs real data to train with each batch
        #
        # Optional Arguments:
        # kLambda -- k learning rate, value is set from the paper
        # logEpochOutput -- Whether to save each output's values
        # saveModelFrequency -- How many epochs to wait between saveing a model
        # sampleSwatch -- Whether to output an example of the data. Top 8 represent training data, bottom 8 represent real data.
        # saveSampleSwatch -- Whether to keep each swatch or overwrite it with the next one.
        # '''
        self.generator = gen
        self.discriminator = disc
        self.gan = gan
        self.dataGenerator = data
        self.datagentr = image.ImageDataGenerator()
        # Separated data for evaluation of PMSE/Combo loss
        self.x1 = np.load('41_imgs.npy') / 255.
        self.y1 = np.load('41_lbls.npy')
        self.x2 = np.load('50_imgs.npy') / 255.
        self.y2 = np.load('50_lbls.npy')
        self.x3 = np.load('51_imgs.npy') / 255.
        self.y3 = np.load('51_lbls.npy')
        self.x4 = np.load('80_imgs.npy') / 255.
        self.y4 = np.load('80_lbls.npy')
        self.x5 = np.load('90_imgs.npy') / 255.
        self.y5 = np.load('90_lbls.npy')
        self.x6 = np.load('130_imgs.npy') / 255.
        self.y6 = np.load('130_lbls.npy')
        self.x7 = np.load('140_imgs.npy') / 255.
        self.y7 = np.load('140_lbls.npy')
        self.x8 = np.load('190_imgs.npy') / 255.
        self.y8 = np.load('190_lbls.npy')
        self.x9 = np.load('200_imgs.npy') / 255.
        self.y9 = np.load('200_lbls.npy')
        # # Separated data for evaluation of PMSE/Combo loss
        # self.x1 = np.load('-90_imgs.npy') / 255.
        # self.y1 = np.load('-90_lbls.npy')
        # self.x2 = np.load('-60_imgs.npy') / 255.
        # self.y2 = np.load('-60_lbls.npy')
        # self.x3 = np.load('-30_imgs.npy') / 255.
        # self.y3 = np.load('-30_lbls.npy')
        # self.x9 = np.load('-15_imgs.npy') / 255.
        # self.y9 = np.load('-15_lbls.npy')
        # self.x4 = np.load('+00_imgs.npy') / 255.
        # self.y4 = np.load('+00_lbls.npy')
        # self.x5 = np.load('+15_imgs.npy') / 255.
        # self.y5 = np.load('+15_lbls.npy')
        # self.x6 = np.load('+30_imgs.npy') / 255.
        # self.y6 = np.load('+30_lbls.npy')
        # self.x7 = np.load('+60_imgs.npy') / 255.
        # self.y7 = np.load('+60_lbls.npy')
        # self.x8 = np.load('+90_imgs.npy') / 255.
        # self.y8 = np.load('+90_lbls.npy')
        # Real-time data augmentation
        self.b1 = self.datagentr.flow(self.x1, self.y1, batch_size=batch_size)
        self.b2 = self.datagentr.flow(self.x2, self.y2, batch_size=batch_size)
        self.b3 = self.datagentr.flow(self.x3, self.y3, batch_size=batch_size)
        self.b4 = self.datagentr.flow(self.x4, self.y4, batch_size=batch_size)
        self.b5 = self.datagentr.flow(self.x5, self.y5, batch_size=batch_size)
        self.b6 = self.datagentr.flow(self.x6, self.y6, batch_size=batch_size)
        self.b7 = self.datagentr.flow(self.x7, self.y7, batch_size=batch_size)
        self.b8 = self.datagentr.flow(self.x8, self.y8, batch_size=batch_size)
        self.b9 = self.datagentr.flow(self.x9, self.y9, batch_size=batch_size)
        try:
            self.dataGenerator.next()
        except:
            raise Exception('Data is expected to be a DataGenerator')

        self.z = self.generator.input_shape[-1]
        self.epsilon = k.epsilon()
        self.kLambda = kLambda
        self.logEpochOutput = logEpochOutput
        self.saveModelFrequency = saveModelFrequency
        self.sampleSwatch = sampleSwatch
        self.saveSampleSwatch = saveSampleSwatch

        self.k = self.epsilon  # If k = 0, like in the paper, Keras returns nan values
        self.firstEpoch = 1
    
    
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

    def train(self, nb_epoch, nb_batch_per_epoch, batch_size, gamma, path=""):

        # Train a Generator network and Discriminator Method using the BEGAN method. The networks are updated sequentially unlike what's done in the paper.
        #
        # Keyword Arguments:
        # nb_epoch -- Number of training epochs
        # batch_size -- Size of a single batch of real data.
        # nb_batch_per_epoch -- Number of training batches to run each epoch.
        # gamma -- Hyperparameter from BEGAN paper to regulate proportion of Generator Error over Discriminator Error. Defined from 0 to 1.
        # path -- Optional parameter specifying location to save output file locations. Starts from the working directory.
        gen_loss = 0.
        disc_loss = 0.00001
        g_x = []
        d_x = []
        for e in range(self.firstEpoch, self.firstEpoch + nb_epoch):
            progbar = generic_utils.Progbar(nb_batch_per_epoch * batch_size)
            start = time.time()
            for b in range(nb_batch_per_epoch):

                #
                # if e > 6:
                #     g_trend = sum(g_x[e - 6:e]) / 5.
                #     d_trend = sum(d_x[e - 6:e]) / 5.
                #     diff = g_trend - d_trend
                #     ratio = gen_loss / disc_loss
                #     ratio = gen_loss / disc_loss
                #     diff = g_trend - d_trend
                #     print(ratio, '   ', diff)
                #     if diff > 1 or ratio > 2:
                #         print('disc train check--------------------------------------------------------')
                #         self.discriminator.trainable = False
                #     if diff < -1 or ratio < 0.5:
                #         print('gen train check--------------------------------------------------------')
                #         self.gan.trainable = False
                # zD = np.random.uniform(-1, 1, (batch_size, self.z))
                # zG = np.random.uniform(-1, 1, (batch_size * 2, self.z))  #

                # Train D
                x_batch, y_batch = self.dataGenerator.next()
                x_batch_to_print = np.rollaxis(x_batch, 3, 1)
                x_batch_to_print = np.rollaxis(x_batch_to_print, 3, 1)

                # print('x batch shape', np.shape(x_batch))

                weights = np.transpose(np.ones(batch_size))
                y_batch_l = [np.where(r == 1)[0][0] for r in y_batch]
                y_1l = [np.where(r == 1)[0][0] for r in self.y1]
                y_2l = [np.where(r == 1)[0][0] for r in self.y2]
                y_3l = [np.where(r == 1)[0][0] for r in self.y3]
                y_4l = [np.where(r == 1)[0][0] for r in self.y4]
                y_5l = [np.where(r == 1)[0][0] for r in self.y5]
                y_6l = [np.where(r == 1)[0][0] for r in self.y6]
                y_7l = [np.where(r == 1)[0][0] for r in self.y7]
                y_8l = [np.where(r == 1)[0][0] for r in self.y8]
                y_9l = [np.where(r == 1)[0][0] for r in self.y9]
                # y_1 = [np.where(r == 1)[0][0] for r in self.y9]
                # counter = 0
                y1b = []
                y2b = []
                y3b = []
                y4b = []
                y5b = []
                y6b = []
                y7b = []
                y8b = []
                y9b = []

                for el in y_batch_l:
                    y1b.append(np.where(y_1l == el)[0][0])
                    y2b.append(np.where(y_2l == el)[0][0])
                    y3b.append(np.where(y_3l == el)[0][0])
                    y4b.append(np.where(y_4l == el)[0][0])
                    y5b.append(np.where(y_5l == el)[0][0])
                    y6b.append(np.where(y_6l == el)[0][0])
                    y7b.append(np.where(y_7l == el)[0][0])
                    y8b.append(np.where(y_8l == el)[0][0])
                    y9b.append(np.where(y_9l == el)[0][0])


                x_gen_b1 = np.array(self.x1)[y1b]
                x_gen_b2 = np.array(self.x2)[y2b]
                x_gen_b3 = np.array(self.x3)[y3b]
                x_gen_b4 = np.array(self.x4)[y4b]
                x_gen_b5 = np.array(self.x5)[y5b]
                x_gen_b6 = np.array(self.x6)[y6b]
                x_gen_b7 = np.array(self.x7)[y7b]
                x_gen_b8 = np.array(self.x8)[y8b]
                x_gen_b9 = np.array(self.x9)[y9b]

                x_batch1, y_batch1 = self.b1.next()
                x_batch2, y_batch2 = self.b2.next()
                x_batch3, y_batch3 = self.b3.next()
                x_batch4, y_batch4 = self.b4.next()
                x_batch5, y_batch5 = self.b5.next()
                x_batch6, y_batch6 = self.b6.next()
                x_batch7, y_batch7 = self.b7.next()
                x_batch8, y_batch8 = self.b8.next()
                x_batch9, y_batch9 = self.b9.next()
                print("PRINTING LABELS")
                b = [np.where(r == 1)[0][0] for r in y_batch1]
                print(b)
                b = [np.where(r == 1)[0][0] for r in y_batch2]
                print(b)
                # print(np.shape((self.discriminator.outputs)[1]))
                # print(np.shape(y_batch), ' ', np.shape(y_batch1))

                d_loss_real = self.discriminator.train_on_batch([x_batch1, x_batch2, x_batch3, x_batch4, x_batch5, x_batch6,
                                                                 x_batch7, x_batch8, x_batch9],
                                                                [weights, y_batch1, weights, y_batch2, weights,  y_batch3,
                                                                 weights,  y_batch4, weights, y_batch5, weights, y_batch6,
                                                                 weights, y_batch7, weights,  y_batch8, weights, y_batch9])

                gen = self.generator.predict(x_batch)

                weights = np.transpose(np.zeros(batch_size))
                print(np.shape(weights))
                d_loss_gen = self.discriminator.train_on_batch(gen,
                                                               [weights, y_batch, weights, y_batch, weights,  y_batch,
                                                                 weights,  y_batch, weights, y_batch, weights, y_batch,
                                                                 weights, y_batch, weights,  y_batch, weights, y_batch])

                d_loss = d_loss_real + d_loss_gen
                # Train G with heuristics
                if e > 6:
                    g_trend = sum(g_x[e - 6:e]) / 5.
                    d_trend = sum(d_x[e - 6:e]) / 5.
                    diff = g_trend - d_trend
                    ratio = gen_loss / disc_loss
                    ratio = gen_loss / disc_loss
                    diff = g_trend - d_trend
                    gen_lloss = self.generator.train_on_batch(x_batch,
                                                              [x_gen_b1, x_gen_b2, x_gen_b3, x_gen_b4, x_gen_b5,
                                                               x_gen_b6, x_gen_b7, x_gen_b8, x_gen_b9])
                    # print(ratio, '   ', diff)
                    if diff > 1 or ratio > 2:
                        self.discriminator.trainable = False
                    if diff < -1 or ratio < 0.5:
                        self.generator.trainable = False
                else:
                    self.discriminator.trainable = False
                    target = self.generator.predict(x_batch)
                    gen_lloss = self.generator.train_on_batch(x_batch, [x_gen_b1, x_gen_b2, x_gen_b3, x_gen_b4, x_gen_b5,
                                                                        x_gen_b6, x_gen_b7, x_gen_b8, x_gen_b9])
                    g_loss = self.gan.train_on_batch(x_batch, [weights, y_batch, weights, y_batch, weights, y_batch,
                                                               weights, y_batch, weights, y_batch, weights, y_batch,
                                                               weights, y_batch, weights, y_batch, weights,
                                                               y_batch])

                disc_loss = LA.norm(d_loss, 1)
                gen_loss = LA.norm(g_loss, 1)
                gen_lloss = LA.norm(gen_lloss,1)
                # print(np.shape(self.discriminator.outputs[0]), np.shape(self.discriminator.outputs[1]))
                # print(y_batch_l)
                # ratio = LA.norm(g_loss, 1) / LA.norm(d_loss, 1)

                # if ratio > 2:
                #     self.generator.trainable = False
                # if ratio < 2:
                #     self.discriminator.trainable = False


                # Update k

                self.k = self.k + self.kLambda * (gamma * LA.norm(np.array(d_loss_real) - np.array(g_loss), 1))
                self.k = min(max(self.k, self.epsilon), 1)
                # print('x_batch shape ', np.shape(x_batch))
                # Report Results
                m_global = d_loss + np.abs(gamma * LA.norm(np.array(d_loss_real) - np.array(g_loss), 1))
                progbar.add(batch_size, values=[("M", LA.norm(m_global, 1)), ("Loss_D", disc_loss),
                                                ("Loss_G", gen_loss), ("k", self.k),
                                                ("Loss Gen", gen_lloss)])

                if (self.logEpochOutput and b == 0):
                    with open(os.getcwd() + path + '/output.txt', 'a') as f:
                        f.write("{}, M: {}, Loss_D: {}, LossG: {}, k: {}, "
                                "gen loss: {}\n".format(e, m_global, d_loss, g_loss, self.k, gen_lloss))
                print('gen  b', np.shape(gen))
                if e % 50 == 0 or e < 50:
                    # Save generated outputs from generator 1
                    for i in range(9):
                        gen0 = np.rollaxis(gen[i], 3, 1)
                        gen0 = np.rollaxis(gen0, 3, 1)
                        # print('gen  b', np.shape(gen))
                        img0, img1, img2, img3, img4, img5, img6, img7, img8, img9 = utils.combine_images_separate(gen0)
                        img0 = img0 * 255
                        img1 = img1 * 255
                        img2 = img2 * 255
                        img3 = img3 * 255
                        img4 = img4 * 255
                        img5 = img5 * 255
                        img6 = img6 * 255
                        img7 = img7 * 255
                        img8 = img8 * 255
                        img9 = img9 * 255
                        # print('gen img b', np.shape(gen_img0))

                        # gen_img0 = gen_img0 * 255
                        # # print(gen_img)
                        # 
                        self.generator.trainable = True
                        self.discriminator.trainable = True
                        Image.fromarray(img0.astype(np.uint8)).save("New_Images/" +
                                                                    str(e + 1) + "_" + str(nb_epoch) + str(i) + "0.png")
                        Image.fromarray(img1.astype(np.uint8)).save("New_Images/" +
                                                                    str(e + 1) + "_" + str(nb_epoch) + str(i) + "1.png")
                        Image.fromarray(img2.astype(np.uint8)).save("New_Images/" +
                                                                    str(e + 1) + "_" + str(nb_epoch) + str(i) + "2.png")
                        Image.fromarray(img3.astype(np.uint8)).save("New_Images/" +
                                                                    str(e + 1) + "_" + str(nb_epoch) + str(i) + "3.png")
                        Image.fromarray(img4.astype(np.uint8)).save("New_Images/" +
                                                                    str(e + 1) + "_" + str(nb_epoch) + str(i) + "4.png")
                        Image.fromarray(img5.astype(np.uint8)).save("New_Images/" +
                                                                    str(e + 1) + "_" + str(nb_epoch) + str(i) + "5.png")
                        Image.fromarray(img6.astype(np.uint8)).save("New_Images/" +
                                                                    str(e + 1) + "_" + str(nb_epoch) + str(i) + "6.png")
                        Image.fromarray(img7.astype(np.uint8)).save("New_Images/" +
                                                                    str(e + 1) + "_" + str(nb_epoch) + str(i) + "7.png")
                        Image.fromarray(img8.astype(np.uint8)).save("New_Images/" +
                                                                    str(e + 1) + "_" + str(nb_epoch) + str(i) + "8.png")
                        Image.fromarray(img9.astype(np.uint8)).save("New_Images/" +
                                                                    str(e + 1) + "_" + str(nb_epoch) + str(i) + "9.png")

                    img0, img1, img2, img3, img4, img5, img6, img7, img8, img9 = utils.combine_images_separate(x_batch_to_print)
                    img0 = img0 * 255
                    img1 = img1 * 255
                    img2 = img2 * 255
                    img3 = img3 * 255
                    img4 = img4 * 255
                    img5 = img5 * 255
                    img6 = img6 * 255
                    img7 = img7 * 255
                    img8 = img8 * 255
                    img9 = img9 * 255
                    # print('gen img b', np.shape(gen_img0))

                    # gen_img0 = gen_img0 * 255
                    # # print(gen_img)
                    self.generator.trainable = True
                    self.discriminator.trainable = True
                    # Image.fromarray(img0.astype(np.uint8)).save("Images/" +
                    #                                             str(e + 1) + "_" + str(nb_epoch) + "0r.png")
                    # Image.fromarray(img1.astype(np.uint8)).save("Images/" +
                    #                                             str(e + 1) + "_" + str(nb_epoch) + "1r.png")
                    # Image.fromarray(img2.astype(np.uint8)).save("Images/" +
                    #                                             str(e + 1) + "_" + str(nb_epoch) + "2r.png")
                    # Image.fromarray(img3.astype(np.uint8)).save("Images/" +
                    #                                             str(e + 1) + "_" + str(nb_epoch) + "3r.png")
                    # Image.fromarray(img4.astype(np.uint8)).save("Images/" +
                    #                                             str(e + 1) + "_" + str(nb_epoch) + "4r.png")
                    # Image.fromarray(img5.astype(np.uint8)).save("Images/" +
                    #                                             str(e + 1) + "_" + str(nb_epoch) + "5r.png")
                    # Image.fromarray(img6.astype(np.uint8)).save("Images/" +
                    #                                             str(e + 1) + "_" + str(nb_epoch) + "6r.png")
                    # Image.fromarray(img7.astype(np.uint8)).save("Images/" +
                    #                                             str(e + 1) + "_" + str(nb_epoch) + "7r.png")
                    # Image.fromarray(img8.astype(np.uint8)).save("Images/" +
                    #                                             str(e + 1) + "_" + str(nb_epoch) + "8r.png")
                    # Image.fromarray(img9.astype(np.uint8)).save("Images/" +
                    #                                             str(e + 1) + "_" + str(nb_epoch) + "9r.png")

                    # Save generated outputs from generators 2-9

                    # gen0 = np.rollaxis(gen[1], 3, 1)
                    # gen0 = np.rollaxis(gen0, 3, 1)
                    # # print('gen  b', np.shape(gen))
                    # gen_img0 = utils.combine_images(gen0)
                    # # print('gen img b', np.shape(gen_img))
                    #
                    # gen_img0 = gen_img0 * 255
                    # # print(gen_img)
                    # self.generator.trainable = True
                    # self.discriminator.trainable = True
                    # Image.fromarray(gen_img0.astype(np.uint8)).save("Images/" +
                    #                                              str(e+1) + "_" + str(nb_epoch) + "1.png")
                    #
                    # gen0 = np.rollaxis(gen[2], 3, 1)
                    # gen0 = np.rollaxis(gen0, 3, 1)
                    # # print('gen  b', np.shape(gen))
                    # gen_img0 = utils.combine_images(gen0)
                    # # print('gen img b', np.shape(gen_img))
                    #
                    # gen_img0 = gen_img0 * 255
                    # # print(gen_img)
                    # self.generator.trainable = True
                    # self.discriminator.trainable = True
                    # Image.fromarray(gen_img0.astype(np.uint8)).save("Images/" +
                    #                                              str(e+1) + "_" + str(nb_epoch) + "2.png")
                    #
                    # gen0 = np.rollaxis(gen[3], 3, 1)
                    # gen0 = np.rollaxis(gen0, 3, 1)
                    # # print('gen  b', np.shape(gen))
                    # gen_img0 = utils.combine_images(gen0)
                    # # print('gen img b', np.shape(gen_img))
                    #
                    # gen_img0 = gen_img0 * 255
                    # # print(gen_img)
                    # self.generator.trainable = True
                    # self.discriminator.trainable = True
                    # Image.fromarray(gen_img0.astype(np.uint8)).save("Images/" +
                    #                                              str(e+1) + "_" + str(nb_epoch) + "3.png")
                    #
                    # gen0 = np.rollaxis(gen[4], 3, 1)
                    # gen0 = np.rollaxis(gen0, 3, 1)
                    # # print('gen  b', np.shape(gen))
                    # gen_img0 = utils.combine_images(gen0)
                    # # print('gen img b', np.shape(gen_img))
                    #
                    # gen_img0 = gen_img0 * 255
                    # # print(gen_img)
                    # self.generator.trainable = True
                    # self.discriminator.trainable = True
                    # Image.fromarray(gen_img0.astype(np.uint8)).save("Images/" +
                    #                                              str(e+1) + "_" + str(nb_epoch) + "4.png")
                    #
                    # gen0 = np.rollaxis(gen[5], 3, 1)
                    # gen0 = np.rollaxis(gen0, 3, 1)
                    # # print('gen  b', np.shape(gen))
                    # gen_img0 = utils.combine_images(gen0)
                    # # print('gen img b', np.shape(gen_img))
                    #
                    # gen_img0 = gen_img0 * 255
                    # # print(gen_img)
                    # self.generator.trainable = True
                    # self.discriminator.trainable = True
                    # Image.fromarray(gen_img0.astype(np.uint8)).save("Images/" +
                    #                                              str(e+1) + "_" + str(nb_epoch) + "5.png")
                    #
                    # gen0 = np.rollaxis(gen[6], 3, 1)
                    # gen0 = np.rollaxis(gen0, 3, 1)
                    # # print('gen  b', np.shape(gen))
                    # gen_img0 = utils.combine_images(gen0)
                    # # print('gen img b', np.shape(gen_img))
                    #
                    # gen_img0 = gen_img0 * 255
                    # # print(gen_img)
                    # self.generator.trainable = True
                    # self.discriminator.trainable = True
                    # Image.fromarray(gen_img0.astype(np.uint8)).save("Images/" +
                    #                                              str(e+1) + "_" + str(nb_epoch) + "6.png")
                    #
                    # gen0 = np.rollaxis(gen[7], 3, 1)
                    # gen0 = np.rollaxis(gen0, 3, 1)
                    # # print('gen  b', np.shape(gen))
                    # gen_img0 = utils.combine_images(gen0)
                    # # print('gen img b', np.shape(gen_img))
                    #
                    # gen_img0 = gen_img0 * 255
                    # # print(gen_img)
                    # self.generator.trainable = True
                    # self.discriminator.trainable = True
                    # Image.fromarray(gen_img0.astype(np.uint8)).save("Images/" +
                    #                                              str(e+1) + "_" + str(nb_epoch) + "7.png")
                    #
                    # gen0 = np.rollaxis(gen[8], 3, 1)
                    # gen0 = np.rollaxis(gen0, 3, 1)
                    # # print('gen  b', np.shape(gen))
                    # gen_img0 = utils.combine_images(gen0)
                    # # print('gen img b', np.shape(gen_img))
                    #
                    # gen_img0 = gen_img0 * 255
                    # # print(gen_img)
                    # self.generator.trainable = True
                    # self.discriminator.trainable = True
                    # Image.fromarray(gen_img0.astype(np.uint8)).save("Images/" +
                    #                                              str(e+1) + "_" + str(nb_epoch) + "8.png")
                    #
                    #
                    #
                    #





                # ratio = LA.norm(g_loss[0], 1) / LA.norm(d_loss[0], 1)
                # if ratio > 200:
                #     g.trainable = False
                # if ratio < 200:
                #     d.trainable = False
                # #
                # if (self.sampleSwatch and b % (nb_batch_per_epoch / 2) == 0):
                #     if (self.saveSampleSwatch):
                #         genName = '/generatorSample_{}_{}.png'.format(e, int(b / nb_batch_per_epoch / 2))
                #         discName = '/discriminatorSample_{}_{}.png'.format(e, int(b / nb_batch_per_epoch / 2))
                #     else:
                #         genName = '/currentGeneratorSample.png'
                #         discName = '/currentDiscriminatorSample.png'
                #     gen = gen[0]
                #     print('gen shape ', np.shape(gen), np.shape(x_batch))
                #     utils.plotGeneratedBatch(x_batch, gen, path + genName)
                #     utils.plotGeneratedBatch(self.discriminator.predict([x_batch1, x_batch2, x_batch3, x_batch4, x_batch5, x_batch6,
                #                                                  x_batch7, x_batch8, x_batch9]), target, path + discName)

            print('\nEpoch {}/{}, Time: {}'.format(e + 1, nb_epoch, time.time() - start))
            g_x.append(gen_loss)
            d_x.append(disc_loss)
            if (e % self.saveModelFrequency == 0):
                utils.saveModelWeights(self.generator, self.discriminator, e, path)
