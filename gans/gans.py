import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import keras
from keras.layers import Dense, Dropout, Input
from keras.models import Model, Sequential
from keras.datasets import mnist
from tqdm import tqdm 
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import adam, RMSprop

def load_data():
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	#normalizing training samples
	x_train = (x_train.astype(np.float32) - 127.5) / 127.5
	x_train = x_train.reshape(60000, 784)
	return (x_train, y_train, x_test, y_test)

def adam_opt():
	return adam(lr = 0.0002, beta_1 = 0.5)

(X_train, y_train, X_test, y_test) = load_data()

#---------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------#
# generator network with tanh activation function with maximum likelihood and simple dense layers
def make_generator():
	gen = Sequential()
	#input layer
	gen.add(Dense(units = 256, input_dim = 100))
	gen.add(LeakyReLU(0.2))
	#first dense layer
	gen.add(Dense(units = 512))
	gen.add(LeakyReLU(0.2))
	#second dense layer
	gen.add(Dense(units = 1024))
	gen.add(LeakyReLU(0.2))
	#output layer
	gen.add(Dense(units = 784, activation = 'tanh'))
	gen.compile(loss = 'binary_crossentropy', optimizer = adam_opt())
	return gen 

g = make_generator()
g.summary()
#---------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------#

# discriminator network with 784 inputs of generator and real inputs
# kind of semi supervised discriminator 

def make_discriminator():
	dis = Sequential()
	#input layer
	dis.add(Dense(units = 1024, input_dim = 784))
	dis.add(LeakyReLU(0.2))
	dis.add(Dropout(0.3))
	#first layer
	dis.add(Dense(units = 512))
	dis.add(LeakyReLU(0.2))
	dis.add(Dropout(0.3))
	#second layer
	dis.add(Dense(units = 256))
	dis.add(LeakyReLU(0.2))
	#output layer
	dis.add(Dense(units = 1, activation = 'sigmoid'))
	dis.compile(loss = 'binary_crossentropy', optimizer = adam_opt())
	return dis 

d = make_discriminator()
d.summary()
#---------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------#

# make a gan with inputs of generator and discriminator
# i used RMS Prop optimizer, adam can also be used
def make_gan(discriminator, generator):
	discriminator.trainable = False
	optimizer = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)
	gan_input = Input(shape = (100,))
	x = generator(gan_input)
	gan_output = discriminator(x)
	gan = Model(inputs = gan_input, outputs = gan_output)
	gan.compile(loss = 'binary_crossentropy', optimizer = optimizer)
	return gan 

gan = make_gan(d, g)
gan.summary()
#---------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------#

# plot the images and save them to view later
def plot_image(epoch, generator, examples = 100, dim = (10,10), figsize = (10,10)):
		noise = np.random.normal(loc = 0, scale = 1, size = [examples, 100])
		gen_images = generator.predict(noise)
		gen_images = gen_images.reshape(100, 28, 28)
		plt.figure(figsize = figsize)
		for i in range(gen_images.shape[0]):
			plt.subplot(dim[0], dim[1], i+1)
			plt.imshow(gen_images[i], interpolation = 'nearest')
			plt.axis('off')
		plt.tight_layout()
		plt.savefig('gan_img %d.png' %epoch)
#---------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------#

def train(epochs = 1, batch_size = 128):
	(X_train, y_train, X_test, y_test) = load_data()
	batch_count = X_train.shape[0] / batch_size

	generator = make_generator()
	discriminator = make_discriminator()
	gan = make_gan(discriminator, generator)

	for e in range(1, epochs+1):
		print("Epoch: %d" %e)
		for _ in tqdm(range(batch_size)):
			noise = np.random.normal(0, 1, [batch_size, 100])
			#fake mnist images
			gen_images = generator.predict(noise)
			#real mnist images
			image_batch = X_train[np.random.randint(low=0,high=X_train.shape[0],size=batch_size)]
			# batch of real and fake images
			X= np.concatenate([image_batch, gen_images])
			# labels for images
			y_dis = np.zeros(2 * batch_size)
			y_dis[:batch_size] = 0.9
			# training the discriminator before running the gan
			discriminator.trainable = True
			discriminator.train_on_batch(X, y_dis)
			# tricking the discriminator as the input from generator being real data
			noise = np.random.normal(0, 1, [batch_size, 100])
			y_gen = np.ones(batch_size)
			#fixing the discriminator weights
			discriminator.trainable = False
			#training the gan
			gan.train_on_batch(noise, y_gen)
		if e == 1 or e % 20 == 0:
			plot_image(e, generator)
train(400, 128)