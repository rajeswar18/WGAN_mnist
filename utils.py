import numpy as np
import inspect, os, re, shutil, sys

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

def prefix_vars_list(varlist, prefix):
    prefix = prefix + '_'
    return [prefix + var.name for var in varlist]


# batch normalization wrapper
def bn(x, gammas, betas, mean, var, axis=None, prefix="", mean_only=False):
    if x.ndim == 1 or x.ndim == 3 :
        raise NotImplementedError("Punk ass dimensions in batch norm")
    elif x.ndim == 2:
        _axis = 1
        pattern = ('x',0)
    elif x.ndim == 4 :
        _axis = [0, 2, 3] # this implies spatial batch norm
        pattern = ('x',0,'x','x')
    elif x.ndim == 5 :
        _axis = [1, 3, 4] # this implies spatial batch norm
        pattern = ('x','x',0,'x','x')
    # if axis is None we assume the axises chosen above
    if axis is None :
        axis = _axis

    mean = x.mean(axis=axis, keepdims=True)
    if not mean_only :
        var = T.mean(T.sqr(x - mean), axis=axis, keepdims=True)
    else :
        var = theano.tensor.ones_like(mean)

    mean.tag.bn_statistic = True
    mean.tag.bn_label = prefix + "_mean"
    var.tag.bn_statistic = True
    var.tag.bn_label = prefix + "_var"

    if betas == 0 :
        pass
    elif betas.ndim == 1:
        betas = betas.dimshuffle(pattern)
    elif betas.ndim == 3:
        betas = betas.dimshuffle((x.ndim-3)*('x',)+(0,1,2,))
    #from theano.tests.breakpoint import PdbBreakpoint
    #bp = PdbBreakpoint('burb morty')
    #mean, var = bp(1, mean, var)

    var_corrected = var + 1e-4
    y = theano.tensor.nnet.bn.batch_normalization(
        inputs=x, gamma=gammas.dimshuffle(pattern), beta=betas,
        mean=mean,
        std=theano.tensor.sqrt(var_corrected),
        mode="low_mem")
    return y, mean, var


def infer_odim_conv(i, k, s):
    return (i-k) // s + 1


def infer_odim_convtrans(i, k, s) :
    return s*(i-1) + k


def log_sum_exp(x, axis=1):
    m = T.max(x, axis=axis)
    return m+T.log(T.sum(T.exp(x-m.dimshuffle(0,'x')), axis=axis))


# http://kbyanc.blogspot.ca/2007/07/python-aggregating-function-arguments.html
def arguments(args_to_pop=None) :
    """Returns tuple containing dictionary of calling function's
    named arguments and a list of calling function's unnamed
    positional arguments.
    """
    posname, kwname, args = inspect.getargvalues(inspect.stack()[1][0])[-3:]
    posargs = args.pop(posname, [])
    args.update(args.pop(kwname, []))
    if args_to_pop is not None :
        for arg in args_to_pop :
            args.pop(arg)
    return args, posargs


def parse_experiments(exp_root_path, save_old=False, enforce_new_name=None, prepare_files=True) :
    script_name = sys.argv[0].split('.py')[0]
    name = script_name if enforce_new_name is None else enforce_new_name

    exp_full_path = exp_root_path+name+'/'
    if prepare_files :
        print "Experiments files at " + exp_full_path

        if save_old:
            print "Save old argument passed, will move any files in this directory to a new folder"
            dirs = [x for x in os.listdir(exp_full_path) if os.path.isdir(exp_full_path + x)]
            if len(dirs) == 0:
                archive = 'run1'
            else:
                dirs = sort_by_numbers_in_file_name(dirs)
                archive = 'run' + str(int(dirs[-1].split('run')[-1]) + 1) + '/'
            cmd = 'mkdir ' + exp_full_path + archive
            print "Doing: " + cmd
            os.system(cmd)

            print "Moving all the files..."
            for f in os.listdir(exp_full_path):
                if os.path.isfile(exp_full_path + f):
                    shutil.move(exp_full_path + f,exp_full_path + archive + f)

        cmd = 'mkdir --parents ' + exp_full_path
        print "Doing: " + cmd
        os.system(cmd)
        print "Copying " + sys.argv[0] + " to " + exp_full_path+name+'.py'
        cmd = "cp " + sys.argv[0] + ' ' + exp_full_path+name+'.py'
        os.system(cmd)

    return exp_full_path, name


def sort_by_numbers_in_file_name(list_of_file_names):
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
        """
        return [ tryint(c) for c in re.split('(\-?[0-9]+)', s) ]

    def sort_nicely(l):
        """ Sort the given list in the way that humans expect.
        """
        l.sort(key=alphanum_key)
        return l

    return sort_nicely(list_of_file_names)


def monitor_in_gradient_dict(graddict, requests, request_type='all') :
    assert request_type in ['all', 'layers']

    monitored = []
    for request in requests :
        monitored_time = []
        monitored_spatial = []
        for key in graddict.keys() :
            if 'W' in graddict[key].name :
                if request in graddict[key].name :
                    var = graddict[key].mean()
                    var.name = graddict[key].name+'gradW_mean'
                    monitored_spatial += [var]
            elif 'U' in graddict[key].name :
                if request in graddict[key].name :
                    var = graddict[key].mean()
                    var.name = graddict[key].name+'gradU_mean'
                    monitored_time += [var]

        if request_type is 'all' :
            var_all = [agrad.mean() for agrad in monitored_spatial]
            val_all.name = request+'gradW_mean'
            monitored += [var_all]
            var_all = [agrad.mean() for agrad in monitored_time]
            val_all.name = request+'gradU_mean'
            monitored += [var_all]

        else :
            monitored += monitored_spatial
            monitored += monitored_time

    return monitored


def parse_tuple(tup) :
    if tup is None :
        return None
    return tup if isinstance(tup, tuple) else tuple([tup,tup])


def fill_tuple(tup):
    if not isinstance(tup, tuple):
        return (tup,), (tup, 1, 1)
    elif len(tup) == 1:
        return tup, tup + (1, 1)
    else:
        return (tup[0],), tup


def get_two_rngs(seed=None):
    if seed is None:
        seed = 4321
    else:
        seed = seed
    rng_np = np.random.RandomState(seed)
    rng_theano = MRG_RandomStreams(seed)
    return rng_np, rng_theano

rng_np, rng_theano = get_two_rngs()


def ortho_weight(ndim):
    """
    Random orthogonal weights, we take
    the right matrix in the SVD.

    Remember in SVD, u has the same # rows as W
    and v has the same # of cols as W. So we
    are ensuring that the rows are
    orthogonal.
    """
    W = rng_np.randn(ndim, ndim)
    u, _, _ = np.linalg.svd(W)
    return u.astype('float32')

def norm_weight_svd(nin, nout, scale=0.01):
    W = rng_np.randn(nin, nout)
    U, S, V = np.linalg.svd(W)
    T = np.zeros((U.shape[1], V.shape[0]), dtype='float32')
    np.fill_diagonal(T, np.ones_like(S).astype('float32'))
    W_ = np.dot(np.dot(U, T), V).astype('float32')
    return W_

def norm_weight(nin,nout=None, scale=0.01, ortho=True):
    """
    Random weights drawn from a Gaussian
    """
    if nout == None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * rng_np.randn(nin, nout)
    return W.astype('float32')


def norm_weight_tensor(shape):
    return np.random.normal(size=shape).astype('float32')


def orthogonal_weight_tensor(shape):
    """
    Random orthogonal matrix as done in blocks
    Orthogonal() for a 2D or 4D tensor.
    4D case : It will return an array of orthogonal matrices
    """
    if len(shape) == 2 :
        if shape[0] == shape[1] :
            M = rng_np.randn(*shape).astype(np.float32)
            Q, R = np.linalg.qr(M)
            Q = Q * np.sign(np.diag(R))
            return Q
        elif shape[1] % shape[0] == 0:
            print "WARNING: You asked for a orth initialization for a 2D tensor"+\
                    " that is not square, but it seems possible to make it orth by blocks"
            weight_tensor = np.empty(shape, dtype=np.float32)
            blocks_of_orth = shape[1] // shape[0]
            for i in range(blocks_of_orth):
                M = rng_np.randn(shape[0],shape[0]).astype(np.float32)
                Q, R = np.linalg.qr(M)
                Q = Q * np.sign(np.diag(R))
                weight_tensor[:,i*shape[0]:(i+1)*shape[0]] = Q
            return weight_tensor
        else :
            print "WARNING: You asked for a orth initialization for a 2D tensor"+\
                    " that is not square and not square by block. Falling back to norm"
            return norm_weight_tensor(shape)

    elif len(shape) == 3 :
        print "WARNING: You asked for a orth initialization for 3D tensor"+\
                " it is not implemented. Falling back to norm init."
        return norm_weight_tensor(shape)

    assert shape[2] == shape[3]
    if shape[2] == 1 :
        return norm_weight_tensor(shape)

    weight_tensor = np.empty(shape, dtype=np.float32)
    shape_ = shape[2:]

    for i in range(shape[0]):
        for j in range(shape[1]) :
            M = rng_np.randn(*shape_).astype(np.float32)
            Q, R = np.linalg.qr(M)
            Q = Q * np.sign(np.diag(R))
            weight_tensor[i,j,:,:] = Q

    return weight_tensor


def gaussianHe_tensor(shape, axis=0, coeff=0):
    wt = np.random.normal(size=shape).astype('float32')
    return wt * np.sqrt(2. / ((1. + coeff**2) * shape[axis]))


def gaussianHe1_tensor(shape, axis=0, coeff=0):
    return gaussianHe_tensor(shape, 1, coeff)


def ones_tensor(shape):
    return np.ones(shape).astype(np.float32)


def zeros_tensor(shape):
    return np.zeros(shape).astype(np.float32)


def identity_tensor(shape):
    assert shape[0] == shape[1]
    return np.identity(shape[0], dtype=np.float32)
