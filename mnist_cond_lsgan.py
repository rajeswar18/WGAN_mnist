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

import logging
import time
import os
import sys

from functools import (partial, )

import numpy as np
import numpy.random as npr
import pandas as pd

import theano
from theano import (grad, function, config, )
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import lasagne
import lasagne.layers as ll
from lasagne.init import (Normal, )
from lasagne.layers import (ReshapeLayer, DropoutLayer,
                            ConcatLayer, ElemwiseSumLayer, GaussianNoiseLayer, )
from lasagne.regularization import (regularize_layer_params, l2, )
from lasagne.nonlinearities import (tanh,sigmoid                  )
from lasagne.updates import (adam, )

from ganalysis.graphing import (get_samples_imgs, get_reconstruction_imgs, )
from ganalysis.utils import (lists2ordict,
                             lists2dict,
                             pack,
                             unpack, shared_floatx, as_array,
                             make_directory, stamp_timedate,
                             copy_script_to_folder, split_list, )
from ganalysis.symbolics import (get_norm, )
from ganalysis.streams import (create_mnist_data_streams, )

from ganalysis.lasagne.distributions import SampleGaussian, SampleDeterministic
from ganalysis.lasagne.layers import (conv_layer, ishmael_resnet,
                                      batch_norm,
                                      BilinearUpsampling,
                                      lrelu, batch_norm,
                                      batch_norm,
                                      resnet_block, )
from ganalysis.lasagne import (save_params, )
from ganalysis.lasagne.hali import HALI
from ganalysis.lasagne.parser import parse_arguments
print 'Loading data ...'
data = pickle.load(gzip.open('mnist.pkl.gz', 'r'))
train, valid, test=data
xtrain, ytrain = train
xtest, ytest=test
xtrain=xtrain.reshape(-1,1,28,28)
xtest=xtest.reshape(-1,1,28,28)
ytrain=ytrain.astype(np.int32)
ytest=ytest.astype(np.int32)
batch_size=100
num_epochs=100
#xtrain=xtrain*255
#xtrain=xtrain.astype(np.int32)
#xtest=xtest*255
#xtest=xtest.astype(np.int32)


dsize=(None,1,28,28)
out_size=10
x=T.tensor4('inputs')
Y=T.matrix('labels')
z=T.tensor4('samples')
ny=10

    # In[19]:
def discriminator(input_var,Y):
    yb = Y.dimshuffle(0, 1, 'x', 'x')

    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)

    network = conv_cond_concat(network, yb)

    network = ll.DropoutLayer(network, p=0.4)

    network = conv_layer(network, 3, 32, 1, 'same', nonlinearity=lrelu)
    network = conv_cond_concat(network, yb)

    network = conv_layer(network, 3, 64, 2, 'same', nonlinearity=lrelu)
    network = conv_cond_concat(network, yb)

    network = conv_layer(network, 3, 64, 2, 'same', nonlinearity=lrelu)
    #network = batch_norm(conv_layer(network, 3, 128, 1, 'same', nonlinearity=lrelu))
    #network = ll.DropoutLayer(network, p=0.2)

    network = conv_layer(network, 3, 128, 2, 'same', nonlinearity=lrelu)
    network = conv_cond_concat(network, yb)

    network = batch_norm(conv_layer(network, 4, 128, 1, 'valid', nonlinearity=lrelu))
    network = conv_cond_concat(network, yb)

    #network= DropoutLayer(network, p=0.5)
    network =conv_layer(network, 1, 1, 1, 'valid', nonlinearity=None)



def conv_cond_concat(x, y):
    """
    concatenate conditioning vector on feature map axis
    """
    return T.concatenate([x, y*T.ones((x.shape[0], y.shape[1], x.shape[2], x.shape[3]))], axis=1)

    return network
NLAT=128
def generator(input_var,Y):
    yb = Y.dimshuffle(0, 1, 'x', 'x')
    network = lasagne.layers.InputLayer(shape=(None, NLAT,1,1),
                                        input_var=input_var)

    network = batch_norm(conv_layer(network, 1, 4 * 4 * 128, 1, 'valid'))
    #print(input_var.shape[0])
    network = ll.ReshapeLayer(network, (-1, 128, 4, 4))
    network = conv_cond_concat(network, yb)
    network = resnet_block(network, 3,128)
    network = conv_cond_concat(network, yb)

    #network = resnet_block(network, 3, 128)
    network = BilinearUpsampling(network, ratio=2)
    network = batch_norm(conv_layer(network, 3,128, 1, 'same'))
    network = conv_cond_concat(network, yb)

    network = resnet_block(network, 3, 128)
    network = conv_cond_concat(network, yb)

    network = BilinearUpsampling(network, ratio=2)
    network = batch_norm(conv_layer(network, 3, 32, 1, 'valid'))
    network = BilinearUpsampling(network, ratio=2)
    network = batch_norm(conv_layer(network, 3, 32, 1, 'same'))
    network = conv_cond_concat(network, yb)

    #network =  resnet_block(network, 3, 32)
    network = conv_layer(network, 1, 1, 1, 'valid', nonlinearity=sigmoid)
    #network =lasagne.layers.Conv2DLayer(network, num_filters=1, filter_size=1, stride=1, nonlinearity=sigmoid)
    return network

# In[23]:
def generator_input( batch_size,nlat=NLAT):
    samples=np.zeros((batch_size,nlat,1,1))
    for i in range(batch_size):
        sample=np.float32(np.random.randn(nlat))
        #print samples[i].shape
        samples[i]=sample.reshape((nlat,1,1))
    #ymb = floatX(OneHot(np_rng.randint(0, 10,  batch_size), ny))

    return samples


print 'Compiling functions ...'
generator_network = generator(z,Y)
discriminator_network=discriminator(x,Y)
D1=lasagne.layers.get_output(discriminator_network)
D2=lasagne.layers.get_output(discriminator_network,lasagne.layers.get_output(generator_network))

#discriminator_loss=-0.5*((D1-D2).mean())
#generator_loss=-0.5*(D2.mean())
discriminator_loss=0.5*(((D1-1)**2).mean())+0.5*(((D2)**2).mean())
generator_loss=0.5*(((D2-1)**2).mean())
weight_decay = 1e-5

generator_params=lasagne.layers.get_all_params(generator_network, trainable=True)
discriminator_params=lasagne.layers.get_all_params(discriminator_network, trainable=True)
base_lr = 2e-3
G_lr = theano.shared(lasagne.utils.floatX(base_lr))
generator_updates = lasagne.updates.adam(generator_loss, generator_params, learning_rate=G_lr,  beta1=0.5)
generator_train_fun=theano.function([z,Y],generator_loss,updates=generator_updates,allow_input_downcast=True)
D_lr = theano.shared(lasagne.utils.floatX(base_lr))



clamp_discriminator=theano

discriminator_updates = lasagne.updates.adam(discriminator_loss, discriminator_params, learning_rate=D_lr,  beta1=0.5)
discriminator_train_fun=theano.function([x,z,Y],discriminator_loss,updates=discriminator_updates,allow_input_downcast=True)
gen_fn = theano.function([z,Y],lasagne.layers.get_output(generator_network,deterministic=True),allow_input_downcast=True)

lr_decay = 0.01


# In[25]:

def OneHot(X, n=None, negative_class=0.):
    X = np.asarray(X).flatten()
    if n is None:
        n = np.max(X) + 1
    Xoh = np.ones((len(X), n)) * negative_class
    Xoh[np.arange(len(X)), X] = 1.
    return Xoh

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
clamp_lower=-0.02
clamp_upper=0.02
num_batches=xtrain.shape[0]/batch_size
print 'Starting training ...'
for i in range(num_epochs):
    print "training epoch",i
    minibatch_gen_losses = []
    minibatch_disc_losses = []
    for batch in iterate_minibatches(xtrain, ytrain, batch_size, shuffle=True):
        xtrain_batch, ytrain_batch=batch
        ytrain_batch = floatX(OneHot(ytrain_batch, ny))
        z_batch=generator_input(batch_size, NLAT)
        minibatch_loss=discriminator_train_fun(xtrain_batch,z_batch,ytrain_batch)
        minibatch_disc_losses.append(minibatch_loss)

        #print "disc_params", discriminator_params

        #print "clipped_params"
        #print np.clip(discriminator_params[1], 0, 1)
        #discriminator_params_values=lasagne.layers.get_all_param_values(discriminator_network, trainable=True)
        #clamped_weights= [np.clip(w, clamp_lower, clamp_upper) for  w in discriminator_params_values]
        #lasagne.layers.set_all_param_values(discriminator_network,clamped_weights, trainable=True)
        z_batch=generator_input(batch_size,NLAT)
        minibatch_loss=generator_train_fun(z_batch,ytrain_batch)
        minibatch_gen_losses.append(minibatch_loss)
    print 'Generator Loss : %.5f' % (np.mean(minibatch_gen_losses))
    print 'Discriminator Loss : %.5f' % (np.mean(minibatch_disc_losses))
    if i >= num_epochs // 2:
        progress = float(i) / num_epochs
        G_lr.set_value(lasagne.utils.floatX(base_lr*2*(1 - progress)))
        D_lr.set_value(lasagne.utils.floatX(base_lr*2*(1 - progress)))
    np.random.seed(i)
    sampless = gen_fn(lasagne.utils.floatX(np.random.rand(batch_size, NLAT,1,1)), floatX(OneHot(np.asarray([[i for _ in range(10)] for i in range(10)]).flatten(), ny)))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pass
    else:
        plt.imsave('/u/mudumbas/mnist_samples_grey_conditional/mnist_samples'+str(i)+'.png',
                    (sampless.reshape(10, 10, 28, 28)
                               .transpose(0, 2, 1, 3)
                               .reshape(10*28, 10*28)),
                    cmap='gray')

    #recon_samples,rvon_taget = iterate_minibatches(xtrain, ytrain, batchsize=100, shuffle=True, flag=False)
