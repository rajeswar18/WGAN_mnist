#!/usr/bin/env python
# coding: utf-8
import pickle
import gzip
import numpy as np
import lasagne
from lasagne.nonlinearities import (sigmoid,linear, )
import PIL.Image as Image
import theano.tensor as T
import theano
import os
import sys
import PIL
import math
from PIL import Image
#from Crypto.Cipher import AES
import hashlib
import binascii
from sklearn import preprocessing
from sklearn.decomposition import PCA
# In[11]:
sys.path.append('/u/mudumbas/encryption')
from paillier.paillier import *


data = pickle.load(gzip.open('mnist.pkl.gz', 'r'))
train, valid, test=data
xtrain, ytrain = train
xtest, ytest=test
xtrain=xtrain.reshape(-1,1,28,28)
xtest=xtest.reshape(-1,1,28,28)
ytrain=ytrain.astype(np.int32)
ytest=ytest.astype(np.int32)
batch_size=64
num_epochs=100
#xtrain=xtrain*255
#xtrain=xtrain.astype(np.int32)
#xtest=xtest*255
#xtest=xtest.astype(np.int32)


dsize=(None,1,28,28)
out_size=10
x=T.tensor4('inputs')
y=T.ivector('labels')
z=T.tensor4('samples')

    # In[19]:
def discriminator(input_var):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=16, filter_size=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network =lasagne.layers.Conv2DLayer(network, num_filters=1, filter_size=1, stride=1,nonlinearity=sigmoid)


    return network
NLAT=100
def generator(input_var):
    network = lasagne.layers.InputLayer(shape=(None, NLAT,1,1),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(network, num_filters=16*4*4, filter_size=(1, 1))

    network = lasagne.layers.ReshapeLayer(network, (-1, 16, 4, 4))
    network = lasagne.layers.TransposedConv2DLayer(network, num_filters=32, filter_size=(3, 3))
    network = lasagne.layers.TransposedConv2DLayer(network, num_filters=32, filter_size=(3, 3),stride=2)
    network = lasagne.layers.TransposedConv2DLayer(network, num_filters=32, filter_size=(4, 4),stride=2)


    network =lasagne.layers.Conv2DLayer(network, num_filters=1, filter_size=1, stride=1, nonlinearity=linear)

    return network

# In[23]:
def generator_input( batch_size=128,nlat=NLAT):
    samples=np.zeros((batch_size,nlat,1,1))
    for i in range(batch_size):
        sample=np.float32(np.random.randn(nlat))
        #print samples[i].shape
        samples[i]=sample.reshape((nlat,1,1))
    return samples



generator_network = generator(z)
discriminator_network=discriminator(x)
D1=lasagne.layers.get_output(discriminator_network)
D2=lasagne.layers.get_output(discriminator_network,lasagne.layers.get_output(generator_network))

discriminator_loss=-0.5*((T.log(D1)+T.log(1-D2)).mean())
generator_loss=-0.5*((T.log(D2)).mean())
weight_decay = 1e-5

generator_params=lasagne.layers.get_all_params(generator_network, trainable=True)
discriminator_params=lasagne.layers.get_all_params(discriminator_network, trainable=True)
base_lr = 2e-4
G_lr = theano.shared(lasagne.utils.floatX(base_lr))
generator_updates = lasagne.updates.adam(generator_loss, generator_params, learning_rate=G_lr,  beta1=0.5)
generator_train_fun=theano.function([z],generator_loss,updates=generator_updates,allow_input_downcast=True)
D_lr = theano.shared(lasagne.utils.floatX(base_lr))

discriminator_updates = lasagne.updates.adam(discriminator_loss, discriminator_params, learning_rate=D_lr,  beta1=0.5)
discriminator_train_fun=theano.function([x,z],discriminator_loss,updates=discriminator_updates,allow_input_downcast=True)
gen_fn = theano.function([z],lasagne.layers.get_output(generator_network,deterministic=True),allow_input_downcast=True)

lr_decay = 0.01


# In[25]:


def iterate_minibatches(inputs, targets, batchsize, shuffle=False,flag=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


num_batches=xtrain.shape[0]/batch_size
for i in range(num_epochs):
    print "training epoch",i
    for batch in iterate_minibatches(xtrain, ytrain, batchsize=128, shuffle=True):
        xtrain_batch, ytrain_batch=batch
        z_batch=generator_input(128, NLAT)
        discriminator_train_fun(xtrain_batch,z_batch)
        z_batch=generator_input(128,NLAT)
        generator_train_fun(z_batch)
    if i >= num_epochs // 2:
        progress = float(i) / num_epochs
        G_lr.set_value(lasagne.utils.floatX(base_lr*2*(1 - progress)))
        D_lr.set_value(lasagne.utils.floatX(base_lr*2*(1 - progress)))

    sampless = gen_fn(lasagne.utils.floatX(np.random.rand(100, NLAT,1,1)))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pass
    else:
        plt.imsave('mnist_samples'+str(i)+'.png',
                    (sampless.reshape(10, 10, 28, 28)
                               .transpose(0, 2, 1, 3)
                               .reshape(10*28, 10*28)),
                    cmap='gray')

    #recon_samples,revon_taget = iterate_minibatches(xtrain, ytrain, batchsize=100, shuffle=True, flag=False)
