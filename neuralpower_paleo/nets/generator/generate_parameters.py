import numpy as np
import caffe
from caffe import layers as L
from caffe import params as P
import shutil, os, fnmatch, json
import os.path
from random import randint

def sample_conv_layer(layer_cnt, num_layers, sample_ranges):
    if layer_cnt < num_layers:
        return {
            'num_output': randint(sample_ranges['conv_num_output'][0], sample_ranges['conv_num_output'][1]),
            'kernel_size': randint(sample_ranges['conv_kernel_size'][0],sample_ranges['conv_kernel_size'][1]),
            'stride': randint(sample_ranges['conv_stride'][0],sample_ranges['conv_stride'][1])
        }
    else:
        return {'num_output': 0, 'kernel_size': 0, 'stride': 0}

def sample_pool_layer(layer_cnt, num_layers, sample_ranges):
    if layer_cnt < num_layers:
        return {
                    'pool': 'MAX',
                    'kernel_size': randint(sample_ranges['pool_kernel_size'][0], sample_ranges['pool_kernel_size'][1]),
                    'stride': randint(sample_ranges['pool_stride'][0], sample_ranges['pool_stride'][1])
        }
    else:
        return {'pool': 'MAX', 'kernel_size': 0, 'stride': 0}


def sample_fc_layer(layer_cnt, num_layers, sample_ranges):
    if layer_cnt < num_layers:
        return {'num_output': randint(sample_ranges['fc_num_output'][0], sample_ranges['fc_num_output'][1])}
    else:
        return {'num_output': 0}

def sample_cnn_config(sample_ranges):

    new_data_point_config = {}

    # randomly select num of convolution layers = 1..5
    num_conv_layers_range = sample_ranges['num_conv_layers']
    new_data_point_config['max_num_conv_layers'] = num_conv_layers_range[1]

    if num_conv_layers_range[0] == num_conv_layers_range[1]: # this includes the case that are 0
        num_conv_layers = num_conv_layers_range[1]
    else:
        num_conv_layers = randint(num_conv_layers_range[0], num_conv_layers_range[1])
    new_data_point_config['num_conv_layers'] = num_conv_layers


    # ... and for each sample their parameter values
    #for conv_layer_cnt in range(new_data_point_config['max_num_conv_layers']):

    for conv_layer_cnt in range(new_data_point_config['num_conv_layers']):
        conv_layer_name = 'conv' + str(conv_layer_cnt+1)
        pool_layer_name = 'pool' + str(conv_layer_cnt + 1)
        new_data_point_config[conv_layer_name] = sample_conv_layer(conv_layer_cnt, num_conv_layers, sample_ranges)
        new_data_point_config[pool_layer_name] = sample_pool_layer(conv_layer_cnt, num_conv_layers, sample_ranges)

    # randomly select number of fully connected layers = 1..5
    num_fc_layers_range = sample_ranges['num_fc_layers']
    if num_fc_layers_range[0] == num_fc_layers_range[1]:
        num_fc_layers = num_fc_layers_range[1]
    else:
        num_fc_layers = randint(num_fc_layers_range[0], num_fc_layers_range[1])
    new_data_point_config['num_fc_layers'] = num_fc_layers
    new_data_point_config['max_num_fc_layers'] = num_fc_layers_range[1]

    # ... and for each sample their parameter values
    #for fc_layer_cnt in range(new_data_point_config['max_num_fc_layers']):

    for fc_layer_cnt in range(new_data_point_config['num_fc_layers']):
        fc_layer_name = 'ip' + str(fc_layer_cnt + 1)
        relu_layer_name = 'relu' + str(fc_layer_cnt + 1)
        new_data_point_config[fc_layer_name] = sample_fc_layer(fc_layer_cnt, num_fc_layers, sample_ranges)
        new_data_point_config[relu_layer_name] = {'type': "ReLU"}

    return new_data_point_config

