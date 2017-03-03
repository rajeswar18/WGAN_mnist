#!/usr/bin/env python
# coding: utf-8
import pickle
import gzip
import numpy as np
import lasagne
from lasagne.nonlinearities import (sigmoid,linear, )
from lasagne.layers import dnn
import lasagne.layers as ll
from lasagne.init import Normal
import PIL.Image as Image
import theano.tensor as T
import theano
import os
import sys
import PIL
import math
import nn
from PIL import Image
#from Crypto.Cipher import AES
import hashlib
import binascii
import h5py
from sklearn import preprocessing
from sklearn.decomposition import PCA
# In[11]:
sys.path.append('/u/mudumbas/encryption')
from paillier.paillier import *


print 'Loading data ...'
mnist_data = h5py.File('/data/lisa/data/binarized_mnist/binarized_mnist.hdf5')
xtrain, xvalid, xtest = mnist_data['features'][:50000], mnist_data['features'][50000:60000], mnist_data['features'][60000:70000]
xtrain = xtrain.astype(np.float32)
xvalid = xvalid.astype(np.float32)
xtest = xtest.astype(np.float32)
ytrain = np.ones(xtrain.shape[0]).astype(np.int32)
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

    network = ll.DropoutLayer(network, p=0.5)

    network = nn.weight_norm(dnn.Conv2DDNNLayer(network, 32, (4,4), pad='valid', W=Normal(0.05), nonlinearity=nn.lrelu))

    network = nn.weight_norm(dnn.Conv2DDNNLayer(network, 64, (5,5), stride=2, pad='valid', W=Normal(0.05), nonlinearity=nn.lrelu))
    network = nn.weight_norm(dnn.Conv2DDNNLayer(network, 128, (5,5), pad='valid', W=Normal(0.05), nonlinearity=nn.lrelu))

    network = nn.weight_norm(dnn.Conv2DDNNLayer(network, 256, (5,5), pad='valid', W=Normal(0.05), nonlinearity=nn.lrelu))

    network = nn.weight_norm(dnn.Conv2DDNNLayer(network, 256, (3,3), pad='valid', W=Normal(0.05), nonlinearity=nn.lrelu))

    network =nn.weight_norm(ll.DenseLayer(network, num_units=1, W=Normal(0.05), nonlinearity=None), train_g=True, init_stdv=0.1)




    return network
NLAT=256
def generator(input_var):
    network = lasagne.layers.InputLayer(shape=(None, NLAT,1,1),
                                        input_var=input_var)

    network = ll.DenseLayer(network, num_units=4*4*256, W=Normal(0.05), nonlinearity=nn.relu)

    network = ll.ReshapeLayer(network, (batch_size,256,4,4))
    network = nn.batch_norm(nn.Deconv2DLayer(network, (batch_size,128,7,7), (4,4), stride=(1,1), pad='valid', W=Normal(0.05), nonlinearity=nn.relu))
    network = nn.batch_norm(nn.Deconv2DLayer(network, (batch_size,64,11,11), (5,5), stride=(1,1), pad='valid', W=Normal(0.05), nonlinearity=nn.relu))
    network = nn.batch_norm(nn.Deconv2DLayer(network, (batch_size,32,25,25), (5,5), stride=(2,2), pad='valid', W=Normal(0.05), nonlinearity=nn.relu))
    network = nn.batch_norm(nn.Deconv2DLayer(network, (batch_size,1,28,28), (4,4), stride=(1,1), pad='valid', W=Normal(0.05), nonlinearity=sigmoid))

    #network =lasagne.layers.Conv2DLayer(network, num_filters=1, filter_size=1, stride=1, nonlinearity=sigmoid)
    return network

# In[23]:
def generator_input( batch_size=128,nlat=NLAT):
    samples=np.zeros((batch_size,nlat,1,1))
    for i in range(batch_size):
        sample=np.float32(np.random.randn(nlat))
        #print samples[i].shape
        samples[i]=sample.reshape((nlat,1,1))
    return samples


print 'Compiling functions ...'
generator_network = generator(z)
discriminator_network=discriminator(x)
D1=lasagne.layers.get_output(discriminator_network)
D2=lasagne.layers.get_output(discriminator_network,lasagne.layers.get_output(generator_network))

discriminator_loss=-0.5*((D1-D2).mean())
generator_loss=-0.5*(D2.mean())
weight_decay = 1e-5

generator_params=lasagne.layers.get_all_params(generator_network, trainable=True)
discriminator_params=lasagne.layers.get_all_params(discriminator_network, trainable=True)
base_lr = 0.00003
G_lr = theano.shared(lasagne.utils.floatX(base_lr))
generator_updates = lasagne.updates.adam(generator_loss, generator_params, learning_rate=G_lr,  beta1=0.5)
generator_train_fun=theano.function([z],generator_loss,updates=generator_updates,allow_input_downcast=True)
D_lr = theano.shared(lasagne.utils.floatX(base_lr))



clamp_discriminator=theano

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

clamp_lower=-0.05
clamp_upper=0.05
num_batches=xtrain.shape[0]/batch_size
print 'Starting training ...'
for i in range(num_epochs):
    print "training epoch",i
    for batch in iterate_minibatches(xtrain, ytrain, batchsize=128, shuffle=True):
        xtrain_batch, ytrain_batch=batch
        z_batch=generator_input(128, NLAT)
        discriminator_train_fun(xtrain_batch,z_batch)
        #print "disc_params", discriminator_params

        #print "clipped_params"
        #print np.clip(discriminator_params[1], 0, 1)
        discriminator_params_values=lasagne.layers.get_all_param_values(discriminator_network, trainable=True)
        clamped_weights= [np.clip(w, clamp_lower, clamp_upper) for  w in discriminator_params_values]
        lasagne.layers.set_all_param_values(discriminator_network,clamped_weights, trainable=True)
        z_batch=generator_input(128,NLAT)
        generator_train_fun(z_batch)
    if i >= num_epochs // 2:
        progress = float(i) / num_epochs
        G_lr.set_value(lasagne.utils.floatX(base_lr*2*(1 - progress)))
        D_lr.set_value(lasagne.utils.floatX(base_lr*2*(1 - progress)))
    np.random.seed(i)
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
