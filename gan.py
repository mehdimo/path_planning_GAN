from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from pandas import read_csv
import pandas
from pandas import DataFrame
import os
from sklearn.preprocessing import LabelEncoder
from pathClassifier import PathClassifier
import time
from keras.utils import np_utils
import matplotlib.pyplot as plt
import sys
import numpy as np
from pylab import *


class GAN():
    def __init__(self):
        self.img_rows = 19 
        self.img_cols = 13 
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity 
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = (100,)
        
        model = Sequential()

        model.add(Dense(256, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)
        
        model = Sequential()

        model.add(Flatten(input_shape=img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def loadData(self, path='dataset/'):
	files = os.listdir(path)
	data = []
	labels = []
	for fn in files:
	    ffn = os.path.join(path, fn)
	    df = read_csv(ffn, index_col=None, header=None)
	    df[df==2]=0
	    data.append(df.values)
	    label = int(fn[0:2])
	    labels.append(label)

	data = np.array(data) 
	return data, labels


    def train(self, epochs, batch_size=2, save_interval=10):

        # Load the dataset
	X_train, y_ = self.loadData()

        X_train = 2 * (X_train.astype(np.float32)) - 1
        X_train = np.expand_dims(X_train, axis=3)
        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

	    if epoch % save_interval == 0:
                self.save_imgs(epoch, X_train, y_)

    def save_imgs(self, epoch, X_train, y_):
        r, c = 4, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

	gen_imgs2 = gen_imgs.reshape(gen_imgs.shape[0], gen_imgs.shape[1]*gen_imgs.shape[2]).astype('float32')
	####################
	'''perform the classification of generated paths'''
	cl = PathClassifier()

	lblEnc = LabelEncoder()
        labels = lblEnc.fit_transform(y_)
	data = X_train
	num_pixels = data.shape[1] * data.shape[2]
	data = data.reshape(data.shape[0], num_pixels).astype('float32')
	labels = np_utils.to_categorical(labels)
	
	num_classes = labels.shape[1]
	model = cl.train_model(data, labels)
	classes = model.predict(gen_imgs2)
	classes = np.argmax(classes, axis=1)
	######################

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
	for i in range(r*c):
	    gm = gen_imgs[i].reshape(gen_imgs[i].shape[0], gen_imgs[i].shape[1])	
	    df = DataFrame(gm) 
	    df.to_csv("gen/paths_%d_%d.csv" % (epoch, i), header=False, index=False)

        fig, axs = plt.subplots(r, c, figsize=(4.5,5))
        cnt = 0
	fig.subplots_adjust(hspace=0.5)
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap=plt.cm.gray)
		axs[i,j].set_title('class ' + str(classes[cnt]))
                axs[i,j].axis('off')
		autoAxis = axs[i,j].axis()
		rec = Rectangle((autoAxis[0]-0.1,autoAxis[2]-0.2),(autoAxis[1]-autoAxis[0])+.2,(autoAxis[3]-autoAxis[2])+0.1,fill=False, lw=0.5)
		rec = axs[i,j].add_patch(rec)
		rec.set_clip_on(False)
		cnt += 1
        fig.savefig("images/paths_%d.png" % epoch, dpi=300)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    start_time = time.time()
    gan.train(epochs=10001, batch_size=64, save_interval=200)
    print("total time: %s seconds ---" % (time.time() - start_time))





