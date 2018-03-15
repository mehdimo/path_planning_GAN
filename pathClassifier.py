import os
import numpy as np
from pandas import read_csv
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import time
import matplotlib.pyplot as plt

class PathClassifier:

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

    def build_classifier_base(self, num_pixels, num_classes):	
	# fix random seed for reproducibility
	seed = 7
	np.random.seed(seed)
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

    def train_model(self, X_train, y_train, eps=10, batch_size=20):
	num_classes = y_train.shape[1]
	num_pixels = X_train.shape[1]
	model = self.build_classifier_base(num_pixels, num_classes)
	model.fit(X_train, y_train, epochs=eps, batch_size=batch_size, verbose=2)
	return model

    def predict_path_class(self, trainedModel, X_test, y_test):
	start_time = time.time()
	scores = trainedModel.evaluate(X_test, y_test, verbose=0)
	t = time.time() - start_time
	print 'prediction time:', t / len(X_test)
	performance = 100-scores[1]*100
	print("Baseline Error: %.2f%%" % performance)
	preds = trainedModel.predict(X_test)
	preds = np.argmax(preds, axis=1)
	yt = np.argmax(y_test, axis=1)
	t = (preds == yt)
	acc = [c for c in t if c==True]

	return 100 - scores[1]*100


if __name__ == '__main__':
	cl = PathClassifier()
	data, labels = cl.loadData()
	lblEnc = LabelEncoder()
        labels = lblEnc.fit_transform(labels)

	num_pixels = data.shape[1] * data.shape[2]
	data = data.reshape(data.shape[0], num_pixels).astype('float32')
	labels = np_utils.to_categorical(labels)
	
	num_classes = labels.shape[1]

	lnx = int(len(data) * 0.7)
	X_train, X_test = data[:lnx], data[lnx:]
	y_train, y_test = labels[:lnx], labels[lnx:]
	accuracy_batch = []
	for i in range(1, 10):
	    model = cl.train_model(X_train, y_train, eps=10, batch_size=i*10)
	    p = cl.predict_path_class(model, X_test, y_test)
	    accuracy_batch.append(np.array([i*10, p]))
	accuracy_batch = np.array(accuracy_batch)

	#### plot accuracy ##########
	plt.plot(accuracy_batch[:,0],accuracy_batch[:,1], marker='o', markerfacecolor='None')
	plt.xlabel('Batch size')
	plt.ylabel('Error rate %')
	plt.savefig('classifier_error.png', dpi=300)
	plt.show()
	plt.close()


	    
	


