import numpy as np
import caffe
from caffe import layers as L
from caffe import params as P
import shutil, os, fnmatch, json


def cnn_to_proto_loop(module_options, sample_ranges, dataset):

    prototxt_buffer = 'name: "Net"\n'

    n = caffe.NetSpec()
    if dataset == 'mnist':

        n.data, n.label = L.Data(name="mnist",
                                 batch_size=sample_ranges['train_batch_size_mnist'],
                                 backend=1,
                                 source=module_options['train_source_mnist'], ntop=2,
                                 transform_param=dict(scale=0.00390625),
                                 include=dict(phase = 0))

        prototxt_buffer += str(n.to_proto())
        n = caffe.NetSpec()
        n.data, n.label = L.Data(name="mnist",
                                 batch_size=sample_ranges['test_batch_size_mnist'],
                                 backend=1,
                                 source=module_options['test_source_mnist'], ntop=2,
                                 transform_param=dict(scale=0.00390625),
                                 include=dict(phase = 1))
        previous_layer = n.data

    elif dataset == 'cifar10':

        n.data, n.label = L.Data(name="cifar",
                                 batch_size=sample_ranges['train_batch_size_cifar10'],
                                 backend=1,
                                 source=sample_ranges['train_source_cifar10'], ntop=2,
                                 transform_param=dict(mean_file="./mean.binaryproto"),
                                 include=dict(phase = 0))

        prototxt_buffer += str(n.to_proto())
        n = caffe.NetSpec()
        n.data, n.label = L.Data(name="cifar",
                                 batch_size=sample_ranges['test_batch_size_cifar10'],
                                 backend=1,
                                 source=sample_ranges['test_source_cifar10'], ntop=2,
                                 transform_param=dict(mean_file="./mean.binaryproto"),
                                 include=dict(phase = 1))
        previous_layer = n.data


    for num_layer in range(module_options['num_conv_layers']):

        layer_name_conv = 'conv' + str(num_layer+1)
        n.conv_ = L.Convolution(previous_layer, name=layer_name_conv,
                                kernel_size=module_options[layer_name_conv]['kernel_size'],
                                stride=module_options[layer_name_conv]['stride'],
                                num_output=module_options[layer_name_conv]['num_output'],
                                weight_filler=dict(type='xavier'),
                                bias_filler = dict(type='constant'),
                                param = dict(lr_mult=2))

        layer_name_pool = 'pool' + str(num_layer+1)
        n.pool_ = L.Pooling(n.conv_, name=layer_name_pool,
                            kernel_size=module_options[layer_name_pool]['kernel_size'],
                            stride=module_options[layer_name_pool]['stride'],
                            pool=P.Pooling.MAX)

        previous_layer = n.pool_

    for num_layer in range(module_options['num_fc_layers']):

        layer_name_fc = 'ip' + str(num_layer+1)
        n.ip_ = L.InnerProduct(previous_layer, name=layer_name_fc,
                               num_output=module_options[layer_name_fc]['num_output'],
                               weight_filler=dict(type='xavier'),
                               bias_filler = dict(type='constant'),
                               param = dict(lr_mult=2))
        layer_name_relu = 'relu' + str(num_layer + 1)
        n.relu_ = L.ReLU(n.ip_, name=layer_name_relu, in_place=True)

        previous_layer = n.relu_

    n.ip_last = L.InnerProduct(previous_layer, num_output=10,
                           weight_filler=dict(type='xavier'),
                           bias_filler = dict(type='constant'),
                           param = dict(lr_mult=2))

    n.accuracy = L.Accuracy(n.ip_last, n.label, include=dict(phase = 1))
    n.loss = L.SoftmaxWithLoss(n.ip_last, n.label)

    prototxt_buffer += str(n.to_proto())
    # generate model prototxt file
    with open("./net/net.prototxt", 'w') as f:
        f.write(prototxt_buffer)
