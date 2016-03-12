# Modified from Lasagne example

from __future__ import division, print_function
import lasagne
import numpy as np
import pickle
import skimage.transform
import skimage.filters
import scipy
import scipy.misc
import theano
import theano.tensor as T
from lasagne.utils import floatX

BGR = True
# VGG-19, 19-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/3785162f95cd2d5fee77
# License: non-commercial use only

def imread(fn):
    img = scipy.misc.imread(fn)
    if BGR:
        return img[:,:,::-1]
    return img


from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
try:
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
except:
    print("unable to import Conv2DDNNLayer, use Conv2DLayer instead")
    from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax

# Note: tweaked to use average pooling instead of maxpooling
def build_model():
    net = {}
    net['input'] = InputLayer((1, 3, None, None))
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
    net['pool1'] = PoolLayer(net['conv1_2'], 2, mode='average_exc_pad')
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
    net['pool2'] = PoolLayer(net['conv2_2'], 2, mode='average_exc_pad')
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
    net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1)
    net['pool3'] = PoolLayer(net['conv3_4'], 2, mode='average_exc_pad')
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
    net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1)
    net['pool4'] = PoolLayer(net['conv4_4'], 2, mode='average_exc_pad')
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
    net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1)
    net['pool5'] = PoolLayer(net['conv5_4'], 2, mode='average_exc_pad')

    return net


net = build_model()
# the modle is pickled by Python 2.7. Python 2 and Python 3 handle string and unicode differently
# so the pickled file nedds to be handled differently in different version of Python.
# I am on a train and no Internet access, forgot the right way to find the version.
if bytes == str:
    # Python 2
    values = pickle.load(open('vgg19_normalized.pkl', 'rb'))['param values']
else:
    #Python 3
    values = pickle.load(open('vgg19_normalized.pkl', 'rb'), encoding='latin1')['param values']
    basestring = str

lasagne.layers.set_all_param_values(net['pool5'], values)
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
content_layers = ['conv4_2']
layers = {k: net[k] for k in content_layers+style_layers}

MEAN_VALUES = np.array([104, 117, 123]).reshape((3,1,1))

def prep_image(im, IMAGE_W, IMAGE_H, BGR=BGR):
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.repeat(im, 3, axis=2)
    h, w, _ = im.shape
    if h*IMAGE_W < w*IMAGE_H:
        im = skimage.transform.resize(im, (IMAGE_H, w*IMAGE_H//h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*IMAGE_W//w, IMAGE_W), preserve_range=True)        

    # Central crop
    h, w, _ = im.shape
    im = im[h//2-IMAGE_H//2:h//2+IMAGE_H//2, w//2-IMAGE_W//2:w//2+IMAGE_W//2]
    
    rawim = im.astype('uint8')
    
    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    
    # Convert RGB to BGR
    if not BGR:
        im = im[::-1, :, :]

    im = im - MEAN_VALUES
    return rawim, floatX(im[np.newaxis])


def gram_matrix(x):
    x = x.flatten(ndim=3)
    g = T.tensordot(x, x, axes=([2], [2]))
    return g


def content_loss(P, X, layer):
    p = P[layer]
    x = X[layer]
    loss = 1./2 * lasagne.objectives.squared_error(x, p).sum()
    return loss


def style_loss(A, X, layer):
    a = A[layer]
    x = X[layer]
    
    A = gram_matrix(a)
    G = gram_matrix(x)
    
    N = a.shape[1]
    M = a.shape[2] * a.shape[3]
    
    loss = 1./(4 * N**2 * M**2) * ((G - A)**2).sum()
    return loss

def total_variation_loss(x):
    return (((x[:,:,:-1,:-1] - x[:,:,1:,:-1])**2 + (x[:,:,:-1,:-1] - x[:,:,:-1,1:])**2)).sum()

def deprocess(x, BGR=BGR):
    x = np.copy(x[0])
    x += MEAN_VALUES
    if not BGR:
        x = x[::-1]
    x = np.swapaxes(np.swapaxes(x, 0, 1), 1, 2)    
    x = np.clip(x, 0, 255).astype('uint8')
    return x

import time
start = time.time()
default_art_img = 'styles/starry_night.jpg' 
default_photo_img = 'styles/starry_night.jpg'
shared_mem = {}
func_mem = {}
outputs_mem = {}

def Func(i, In, Out, updates=None):
    if i not in func_mem:
        func_mem[i] = theano.function(In, Out, updates=updates)
    return func_mem[i]

def Eval(i, Out):
    return Func(i, [], Out)()

def Shared(i, v):
    if i in shared_mem:
        shared_mem[i].set_value(v)
    else:
        shared_mem[i]=theano.shared(v)
    return shared_mem[i]

def get_img(i):
    return imread(i) if isinstance(i, basestring) else i            

def transfer(IMAGE_W=512, photo_img=default_photo_img, art_img=default_art_img, 
             iters=8, maxfun=40, init_img=None, photo_weight=0.001, learning_rate=10., ADAM=False):
    IMAGE_H = IMAGE_W * 3 // 4
    print("start transfer (%d)"%IMAGE_W, time.time()-start)
    # Helper functions to interface with scipy.optimize
    def eval_loss(x0):
        x0 = floatX(x0.reshape((1, 3, IMAGE_H, IMAGE_W)))
        generated_image.set_value(x0)
        return f_loss().astype('float64')

    def eval_grad(x0):
        #x0 = floatX(x0.reshape((1, 3, IMAGE_H, IMAGE_W)))
        #generated_image.set_value(x0)
        return f_grad().flatten().astype('float64')
    
    rawim, photo = prep_image(get_img(photo_img), IMAGE_W, IMAGE_H)
    print("photo prepared", time.time()-start)

    rawim, art = prep_image(get_img(art_img), IMAGE_W, IMAGE_H)
    print("art prepared", time.time()-start)

    # Precompute layer activations for photo and artwork
    input_image = Shared((IMAGE_W, "input"), art)
    if IMAGE_W in outputs_mem:
        outputs = outputs_mem[IMAGE_W]
    else:
        outputs = outputs_mem[IMAGE_W] = lasagne.layers.get_output(layers.values(), input_image)
    art_features = {k: Shared((IMAGE_W, "art_"+k), Eval((IMAGE_W, k), output))
                    for k, output in zip(layers.keys(), outputs)}
    input_image.set_value(photo)
    photo_features = {k: Shared((IMAGE_W, "photo_"+k), Eval((IMAGE_W, k), output))
                      for k, output in zip(layers.keys(), outputs)}
    print("precomputed layers", time.time()-start)
    # Get expressions for layer activations for generated image
    generated_image = input_image
    if init_img is None:
        generated_image.set_value(floatX(np.random.uniform(-128, 128, (1, 3, IMAGE_H, IMAGE_W))))
    elif init_img is "photo":
        pass
    else:
        rawinit, init = prep_image(get_img(init_img), IMAGE_W, IMAGE_H)
        generated_image.set_value(init)        
    gen_features = {k: v for k, v in zip(layers.keys(), outputs)}
    print("gen_features", time.time()-start)
    
    # Define loss function
    # total variation penalty
    total_loss = 0.1e-7 * total_variation_loss(generated_image)
    for layer in content_layers:
        total_loss += photo_weight * content_loss(photo_features, gen_features, layer)
    for layer in style_layers:
                total_loss += 0.2e6 * style_loss(art_features, gen_features, layer)
    print("define total_loss", time.time()-start)
    if ADAM:
        # update function
        params = [generated_image]
        key = (IMAGE_W, "train")
        if key not in func_mem:
            updates = lasagne.updates.adam(total_loss, params, learning_rate=learning_rate)
            func_mem[key] = theano.function([], total_loss, updates=updates)
        train_fn = func_mem[key]    
        print("build update functions", time.time()-start)
    else:
        grad = T.grad(total_loss, generated_image)
        # Theano functions to evaluate loss and gradient
        f_loss = Func((IMAGE_W, "loss"), [], total_loss)        
        f_grad = Func((IMAGE_W, "grad"), [], grad)        
        x0 = generated_image.get_value().astype('float64')
    # Optimize, saving the result periodically
    for i in range(iters):
        if ADAM:
            for j in range(maxfun):
                loss = train_fn()
        else:
            x, loss, d = scipy.optimize.fmin_l_bfgs_b(eval_loss, x0.flatten(), fprime=eval_grad, maxfun=maxfun)
        x0 = generated_image.get_value().astype('float64')
        print("iter=%d"%i, "loss=%f"%loss, time.time()-start)
        yield  deprocess(x0)

# mixed
def p_transfer(photo_img=default_photo_img, art_img=default_art_img, precompute=False, preview=False):
    global start
    start = time.time()
    p = lambda n: 0 if precompute else n
    x0 = "photo"
    for x0 in transfer(160, photo_img, art_img, iters=p(2), maxfun=40, init_img="photo", photo_weight=0.005):
        yield x0
    for x0 in transfer(400, photo_img, art_img, iters=p(2), maxfun=20, init_img=x0, photo_weight=0.001, ADAM=False):
        yield x0
    if not preview:
        for x0 in transfer(640, photo_img, art_img, iters=p(1), maxfun=8, init_img=x0, photo_weight=0.001, ADAM=False, learning_rate=8.):
            yield x0
        
def precompute():
    for x in p_transfer(precompute=True):
        pass
    print("precompute done")


