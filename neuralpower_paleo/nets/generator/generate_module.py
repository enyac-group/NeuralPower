import numpy as np
import caffe
from caffe import layers as L
from caffe import params as P
import shutil, os, fnmatch, json


def cnn_to_proto_loop(module_options, sample_ranges, dataset, prototxt_file):

    if dataset == 'mnist':
        prototxt_buffer = 'name: "LeNet"\n'
        prototxt_buffer += '"layer {"\n'
        prototxt_buffer += '  name: "data"\n'
        prototxt_buffer += '  type: "Input"\n'
        prototxt_buffer += '  top: "data"\n'
        prototxt_buffer += '  input_param { shape: { dim: ' + str(sample_ranges['train_batch_size_mnist'])  + ' dim: 1 dim: 28 dim: 28 } }\n}\n'
    elif dataset == 'cifar10':
        prototxt_buffer = 'name: "CIFAR10_full"\n'
        prototxt_buffer += 'input: "data"\n'
        prototxt_buffer += 'input_dim: ' + str(int(sample_ranges['train_batch_size_cifar10'])) + '\n'
        prototxt_buffer += 'input_dim: 3\n'
        prototxt_buffer += 'input_dim: 32\n'
        prototxt_buffer += 'input_dim: 32\n'
    else:
        print "Unknown dataset specified -- Exiting!!"
        exit()

    prototxt_buffer += '#pragma-off\n'

    n = caffe.NetSpec()
    if dataset == 'mnist':

        n.data, n.label = L.Data(name="mnist",
                                 batch_size=sample_ranges['train_batch_size_mnist'],
                                 backend=1,
                                 source=sample_ranges['train_source_mnist'], ntop=2,
                                 transform_param=dict(scale=0.00390625),
                                 include=dict(phase = 0))

        prototxt_buffer += str(n.to_proto())
        n = caffe.NetSpec()
        n.data, n.label = L.Data(name="mnist",
                                 batch_size=sample_ranges['test_batch_size_mnist'],
                                 backend=1,
                                 source=sample_ranges['test_source_mnist'], ntop=2,
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

    prototxt_buffer += '\n#pragma-on\n'

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

    write_on_1 = True
    past_first_layer = False
    write_on_2 = True
    lines = prototxt_buffer.split('\n')


    # generate model prototxt file
    with open(prototxt_file, 'w') as f:
        for line in lines:

            if line.strip() == '#pragma-off':
                write_on_1 = False
                continue
            if line.strip() == '#pragma-on':
                write_on_1 = True
                continue

            if line.strip() == '"layer {"':
                f.write("layer {\n")
                continue

            if line.strip() == 'layer {' and write_on_1 and not past_first_layer:
                if dataset == 'cifar10':
                    f.write("\n")
                past_first_layer = True
                write_on_2 = False
                continue
            if line.strip() == 'layer {' and past_first_layer:
                write_on_2 = True

            if write_on_1 and write_on_2:
                #print line

                # dstam -- find and replace pycaffe default naming to match paleo syntax
                line_ = line
                for next_conv in range(module_options['num_conv_layers']-1):

                    if line == '  top: "Convolution' + str(next_conv+1) + '"':
                        line_ = '  top: "conv' + str(next_conv+1) + '"'
                    if line == '  top: "Pooling' + str(next_conv+1) + '"':
                        line_ = '  top: "pool' + str(next_conv+1) + '"'

                    if line == '  bottom: "Convolution' + str(next_conv+1) + '"':
                        line_ = '  bottom: "conv' + str(next_conv+1) + '"'
                    if line == '  bottom: "Pooling' + str(next_conv+1) + '"':
                        line_ = '  bottom: "pool' + str(next_conv+1) + '"'

                if line == '  top: "conv_"':
                    line_ = '  top: "conv' + str(module_options['num_conv_layers']) + '"'
                if line == '  top: "pool_"':
                    line_ = '  top: "pool' + str(module_options['num_conv_layers']) + '"'
                if line == '  bottom: "conv_"':
                    line_ = '  bottom: "conv' + str(module_options['num_conv_layers']) + '"'
                if line == '  bottom: "pool_"':
                    line_ = '  bottom: "pool' + str(module_options['num_conv_layers']) + '"'


                for next_ip in range(module_options['num_fc_layers'] - 1):

                    if line == '  top: "InnerProduct' + str(next_ip + 1) + '"':
                        line_ = '  top: "ip' + str(next_ip + 1) + '"'
                    if line == '  bottom: "InnerProduct' + str(next_ip + 1) + '"':
                        line_ = '  bottom: "ip' + str(next_ip + 1) + '"'

                if line == '  top: "ip_"':
                    line_ = '  top: "ip' + str(module_options['num_fc_layers']) + '"'
                if line == '  top: "ip_last"':
                    line_ = '  top: "ip' + str(module_options['num_fc_layers']+1) + '"'

                if line == '  bottom: "ip_"':
                    line_ = '  bottom: "ip' + str(module_options['num_fc_layers']) + '"'
                if line == '  bottom: "ip_last"':
                    line_ = '  bottom: "ip' + str(module_options['num_fc_layers']+1) + '"'
                if line == '  name: "ip_last"':
                    line_ = '  name: "ip' + str(module_options['num_fc_layers']+1) + '"'

                f.write(line_ + "\n")