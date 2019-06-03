import os
import numpy as np
import keras.backend as K
import matplotlib.pylab as plt
import math
import os
import tensorflow as tf


def dataRescale(x):
    return x*2/255 - 1


def inverseRescale(X):
    return (np.array(X) * 0.5 + 0.5) * 255.


def saveModelWeights(generator_model, discriminator_model, e, localPath):
    
    if(localPath.endswith('/')):
        raise Exception('Path must not end with /')

    model_path = os.getcwd() + localPath

    gen_weights_path = os.path.join(model_path, 'gen_epoch'+ str(e) +'.h5')
    generator_model.save_weights(gen_weights_path, overwrite=True)

    disc_weights_path = os.path.join(model_path, 'disc_epoch'+ str(e) +'.h5')
    discriminator_model.save_weights(disc_weights_path, overwrite=True)


def plotGeneratedBatch(X_real, X_gen, localPath):

    if(not localPath.endswith('.png')):
        raise Exception('Must be .png file')
    # X_real = inverseRescale(X_real)
    # X_gen = inverseRescale(X_gen)
    # Xg = np.array(X_gen)
    # print(np.shape(Xg))
    # Xg = Xg[0, :, :, :, :]
    # print(np.shape(Xg))
    Xg = X_gen[:8]
    Xr = X_real[:8]
    # print(np.shape(Xg), np.shape(Xr))
    ax = 0 if K.image_dim_ordering() == "tf" else 1
    # print (np.shape(Xg), np.shape(Xr))
    X = np.concatenate((Xg, Xr), axis=0)
    print(np.shape(Xg), np.shape(Xr), np.shape(X), X.shape[0])
    list_rows = []
    for i in range(int(X.shape[0] / 4)):
        Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=ax+1)
        list_rows.append(Xr)

    Xr = np.concatenate(list_rows, axis=ax)
    if(ax == 1):
        Xr = Xr.transpose(1,2,0)

    if Xr.shape[-1] == 1:
        plt.imshow(Xr[:, :, 0], cmap="gray")
    else:
        plt.imshow(Xr)
    plt.savefig(os.getcwd() + localPath)
    plt.clf()
    plt.close()


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    # channel = 3
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:4]
    # print shape[2]
    image = np.zeros((height*shape[0], width*shape[1], shape[2]),
                     dtype=generated_images.dtype)
    # image1 = np.zeros((height * shape[0], width * shape[1], shape[2]),
    #                  dtype=generated_images.dtype)
    # c_image = np.zeros((height*shape[0], 2*width*shape[1], shape[2]),
    #                  dtype=generated_images.dtype)
    for index, rgb in enumerate(generated_images):
        # r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        # img1 = 0.2989 * r + 0.5870 * g + 0.1140 * b
        print('index', index)
        img = rgb
        print (img.shape)
        # print img[:,:,0]
        i = int(index/width)
        # print i
        j = index % width
        # print j
        # image1[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1],0] = img1[:,:]
        # image1[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 0] = img1[:,:]
        # image1[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 0] = img1[:,:]
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 0] = img[:, :, 0]
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 1] = img[:, :, 1]
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 2] = img[:, :, 2]
    # c_image[0:height*shape[0], 0:width*shape[1], 0:shape[2]] = image
    # c_image[0:height*shape[0], width*shape[1]:2*width*shape[1], 0:shape[2]] = image1
    return image


def combine_images_separate(generated_images):
    img0 = generated_images[0, :, :, :]
    # print('generated ', np.shape(img0))
    img1 = generated_images[1, :, :, :]
    img2 = generated_images[2, :, :, :]
    img3 = generated_images[3, :, :, :]
    img4 = generated_images[4, :, :, :]
    img5 = generated_images[5, :, :, :]
    img6 = generated_images[6, :, :, :]
    img7 = generated_images[7, :, :, :]
    img8 = generated_images[8, :, :, :]
    img9 = generated_images[9, :, :, :]
    return img0, img1, img2, img3, img4, img5, img6, img7, img8, img9


def save_images(generated_images, epoch, batch):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    # channel = 3
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:4]
    # print shape[2]
    image = np.zeros((height*shape[0], width*shape[1], shape[2]),
                     dtype=generated_images.dtype)
    # image1 = np.zeros((height * shape[0], width * shape[1], shape[2]),
    #                  dtype=generated_images.dtype)
    # c_image = np.zeros((height*shape[0], 2*width*shape[1], shape[2]),
    #                  dtype=generated_images.dtype)
    for index, rgb in enumerate(generated_images):
        # r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        # img1 = 0.2989 * r + 0.5870 * g + 0.1140 * b
        img = rgb
        # img = img * 127.5 + 127.5
        Image.fromarray(img.astype(np.uint8)).save("New_Images/" + str(epoch) +"_" + str(index) + "_" + str(batch) + ".png")
        # print img.shape
        # print img[:,:,0]
        # i = int(index/width)
        # # print i
        # j = index % width
        # # print j
        # # image1[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1],0] = img1[:,:]
        # # image1[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 0] = img1[:,:]
        # # image1[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 0] = img1[:,:]
        # image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1],0] = img[:, :,0]
        # image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 1] = img[:, :, 1]
        # image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 2] = img[:, :, 2]
    # c_image[0:height*shape[0], 0:width*shape[1], 0:shape[2]] = image
    # c_image[0:height*shape[0], width*shape[1]:2*width*shape[1], 0:shape[2]] = image1
    return image