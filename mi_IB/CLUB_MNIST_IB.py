# Sample Code of the Information Bottleneck method using CLUB on the MNIST dataset.
# Code is adapted from https://github.com/alexalemi/vib_demo

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys
import os
import argparse

from architecture import encoder, decoder, mi
from utils import vars_from_scopes, gaussian_nll

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

## Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', dest='epochs', type=int, default=200)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=200)
parser.add_argument('--seed', dest='seed', type=int, default=1234)
parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--BETA', dest='BETA', type=float, default=0.001, help='reg coefficient')
parser.add_argument('--model', type=str, default='CLUB', help='which model to use: CLUB or vCLUB')
parser.add_argument('--experiment_name', dest='experiment_name', default='exp')
args = parser.parse_args()
print(args)

tf.reset_default_graph()
os.environ['PYTHONHASHSEED']=str(args.seed)
tf.set_random_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Turn on xla optimization
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = tf.InteractiveSession(config=config)
# omit warning signs
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR) # ignore data loading warnings

mnist_data = input_data.read_data_sets('/tmp/mnistdata', validation_size=0)

images = tf.placeholder(tf.float32, [None, 784], 'images')
labels = tf.placeholder(tf.int64, [None], 'labels')
one_hot_labels = tf.one_hot(labels, 10)

ds = tf.contrib.distributions
prior = ds.Normal(0.0, 1.0)

batch_size = args.batch_size
epochs = args.epochs

##########################################################
#################### Encoding ############################
##########################################################

with tf.variable_scope('encoder'):
    encoding = encoder(images)
    z = encoding.sample(seed=1)
    if args.model == "CLUB":
        mu = encoding.mean()
        var = encoding.variance()
        
##########################################################
#################### Decoding ############################
##########################################################
with tf.variable_scope('decoder'):
    logits = decoder(z)

##########################################################
################# Mutual Information #####################
##########################################################
if args.model == "vCLUB":
    with tf.variable_scope('mi'):
        mu, logvar = mi(images)
        var = tf.exp(logvar)

        
# calculate the log likelihood of y|x positive samples
positive = -(mu - z)**2/var
positive = tf.reduce_mean(tf.reduce_sum(positive, -1))
if args.model == "vCLUB":
    lld = positive - tf.reduce_mean(tf.reduce_sum(logvar, -1))

# calculate the log likelihood of y|x negative samples
z_tile = tf.tile(tf.expand_dims(z, dim=0), tf.constant([batch_size, 1, 1], tf.int32))
mu_appr_tile = tf.tile(tf.expand_dims(mu, dim=1), tf.constant([1, batch_size, 1], tf.int32))
var_tile = tf.tile(tf.expand_dims(var, dim=1), tf.constant([1, batch_size, 1], tf.int32))
negative = -(z_tile - mu_appr_tile)**2/var_tile
negative = tf.reduce_mean(tf.reduce_sum(negative, -1))

# calculate the mutual information upper
mi_est = positive - negative

##########################################################
######################## Loss ############################
##########################################################
with tf.variable_scope('decoder', reuse=True):
    many_logits = decoder(encoding.sample(12, seed=1))
    
class_loss = tf.losses.softmax_cross_entropy(
    logits=logits, onehot_labels=one_hot_labels)

BETA = 1e-3

info_loss = tf.reduce_sum(tf.reduce_mean(
    ds.kl_divergence(encoding, prior), 0)) / math.log(2) # original regularization for VIB

# total_loss = class_loss + BETA * info_loss
total_loss = class_loss + args.BETA * mi_est

##########################################################
####################### Optimization #####################
##########################################################
total_vars = vars_from_scopes(['encoder', 'decoder'])

accuracy = tf.reduce_mean(tf.cast(tf.equal(
    tf.argmax(logits, 1), labels), tf.float32))
avg_accuracy = tf.reduce_mean(tf.cast(tf.equal(
    tf.argmax(tf.reduce_mean(tf.nn.softmax(many_logits), 0), 1), labels), tf.float32))
IZY_bound = math.log(10, 2) - class_loss
IZX_bound = info_loss

steps_per_batch = int(mnist_data.train.num_examples / batch_size)
global_step = tf.contrib.framework.get_or_create_global_step()
learning_rate = tf.train.exponential_decay(args.lr, global_step,
                                           decay_steps=2*steps_per_batch,
                                           decay_rate=0.99, staircase=True)
opt = tf.train.AdamOptimizer(learning_rate, 0.5)
if args.model == "vCLUB":
    mi_vars = vars_from_scopes(['mi'])
#     opt_mi = tf.train.AdamOptimizer(1e-4, 0.5).minimize(-lld, var_list = mi_vars)

ma = tf.train.ExponentialMovingAverage(0.999, zero_debias=True)
ma_update = ma.apply(tf.model_variables())

saver = tf.train.Saver()
saver_polyak = tf.train.Saver(ma.variables_to_restore())

train_tensor = tf.contrib.training.create_train_op(total_loss, opt,
                                                   global_step,
                                                   update_ops=[ma_update],
                                                   variables_to_train=total_vars)
if args.model == "vCLUB":
    mi_tensor = tf.contrib.training.create_train_op(-lld, opt,
                                                       global_step,
                                                       update_ops=[ma_update],
                                                       variables_to_train=mi_vars)

tf.global_variables_initializer().run()
##########################################################
####################### Evaluation #######################
##########################################################
# def evaluate():
#     IZY_t = acc_t = avg_acc_t = mi_t = 0
#     epochs = mnist_data.test.images.shape[0]//batch_size
#     for i in range(epochs):
#         IZY, acc, avg_acc, mi = \
#         sess.run([IZY_bound, accuracy, avg_accuracy, mi_est],\
#         feed_dict={images: mnist_data.test.images[batch_size*i:batch_size*(i+1)],\
#                    labels: mnist_data.test.labels[batch_size*i:batch_size*(i+1)]})
#         IZY_t, mi_t, acc_t, avg_acc_t = IZY_t+IZY, mi_t+mi, acc_t+acc, avg_acc_t+avg_acc

#     IZY_t, mi_t, acc_t, avg_acc_t = IZY_t/epochs, mi_t/epochs, acc_t/epochs, avg_acc_t/epochs
    
#     return IZY_t, mi_t, acc_t, avg_acc_t, 1-acc_t, 1-avg_acc_t

def evaluate():
    IZY_t = IZX_t = acc_t = avg_acc_t = cla_t = mi_t = 0
    epochs = mnist_data.test.images.shape[0]//batch_size
    for i in range(epochs):
        IZY, IZX, acc, avg_acc, cla, mi, mean, sigma2 = sess.run([IZY_bound, IZX_bound, accuracy, avg_accuracy, class_loss, mi_est, mu, var],
                                 feed_dict={images: mnist_data.test.images[batch_size*i:batch_size*(i+1)], labels: mnist_data.test.labels[batch_size*i:batch_size*(i+1)]})
        IZY_t += IZY
        IZX_t += IZX
        acc_t += acc
        avg_acc_t += avg_acc
        
        cla_t += cla
        mi_t += mi
    IZY_t /= epochs
    IZX_t /= epochs
    acc_t /= epochs
    avg_acc_t /= epochs
    
    cla_t /= epochs
    mi_t /= epochs

    return IZY_t, mi_t, acc_t, avg_acc_t, 1-acc_t, 1-avg_acc_t
##########################################################
####################### Training #########################
##########################################################
smallest_error = 1.

for epoch in range(epochs):
    for step in range(steps_per_batch):
        im, ls = mnist_data.train.next_batch(batch_size)
        
        if args.model == "vCLUB":
            for i in range(1):
#                 _, lld_loss = sess.run([opt_mi, lld], feed_dict={images: im, labels: ls})
                sess.run(mi_tensor, feed_dict={images: im, labels: ls})
                
        sess.run(train_tensor, feed_dict={images: im, labels: ls})
          
    evaluation = evaluate()
    print("{}: IZY={:.2f}\tIZX={:.2f}\tacc={:.4f}\tavg_acc={:.4f}\terr={:.4f}\tavg_err={:.4f}".format(epoch, *evaluation))
    smallest_error = min(evaluation[5], smallest_error)
    sys.stdout.flush()
    
print("The smallest error is", smallest_error)
savepth = saver.save(sess, '/tmp/mnistvib', global_step)

saver_polyak.restore(sess, savepth)
print(evaluate())

saver.restore(sess, savepth)
print(evaluate())