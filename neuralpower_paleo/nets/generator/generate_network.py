import numpy as np
import h5py

import os
import time, threading, shutil, fnmatch, subprocess
from random import randint
import matplotlib.pyplot as plt
import generate_module, generate_parameters
import json, argparse
import sys

# this script samples the net configuration and creates the prototxt files for caffe
#param_values_file = './cnn_network_param.json'
#dataset = 'cifar10' # this is either 'mnist' or 'cifar10'
dataset = sys.argv[1]
prototxt_file = sys.argv[2]
sample_ranges ={
    'train_source_mnist': "./mnist_train_lmdb",
    'train_batch_size_mnist': [10 200],
    'test_source_mnist': "./mnist_test_lmdb",
    'test_batch_size_mnist': 100,
    'train_source_cifar10': "./cifar10_train_lmdb",
    'train_batch_size_cifar10': 200,
    'test_source_cifar10': "./cifar10_test_lmdb",
    'test_batch_size_cifar10': 200,
    'num_conv_layers': [2,6],
    'num_fc_layers': [1,4], # this does not include the last one of softmax
    'conv_num_output': [30, 60], # the two entries are the min max of the range
    'conv_kernel_size': [2, 5],
    'conv_stride': [1, 2],
    'pool_kernel_size': [1, 2],
    'pool_stride': [1, 2],
    'fc_num_output': [50, 1000]
}


# step 1 -- sample a new random configuration
sample_config = generate_parameters.sample_cnn_config(sample_ranges)
#with open(param_values_file,'w') as f:
#    json.dump(sample_config, f, indent=4, sort_keys=True)
# step 2 -- generate module
generate_module.cnn_to_proto_loop(sample_config, sample_ranges, dataset, prototxt_file)

