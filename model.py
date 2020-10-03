import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from util import * 

class DarknetModel(nn.Module):
    def __init__(self, blocks, network_info, module_list, use_cuda):
        super(DarknetModel, self).__init__()
        self.blocks = blocks
        self.network_info = network_info
        self.module_list = module_list
        self.use_cuda = use_cuda

        if use_cuda:
            self.cuda()   

    def forward(self, x):
        if self.use_cuda:
            x = x.cuda()
            
        # stores outputs of every layer
        outputs = [0]  # pre-initialise with 0 for padding as we skip first block
        
        stored_predictions = None  # stores our predictions thus far (at each yolo layer, so 3 different sets of predictions for different grid sizes)
        for i, block in enumerate(self.blocks[1:]):

            if block['type'] in ('convolutional', 'upsample'):
                # simply pass through layer normally
                x = self.module_list[i](x)

            elif block['type'] == 'route':
                # get indices of all layers we are routing from
                layers = [int(i) for i in block['layers']]

                # concat all these layers' outputs together
                x = torch.cat([outputs[i] for i in layers], dim=1)

            elif block['type'] == 'shortcut':
                # simply add outputs of layer we are shortcutting from to outputs of prev layer
                x = x + outputs[int(block['from'])]

            elif block['type'] == 'yolo':
                # detection layer -> get and transform our predictions 

                anchors = self.module_list[i][0].anchors

                # get dimensions of input image (assumes square input)
                input_size = int(self.network_info['height'])
                num_classes = int(block['classes'])

                # transform predictons using our function in util.py
                x = transform_prediction(x.data, input_size, anchors, num_classes, self.use_cuda)

                if stored_predictions is None:
                    stored_predictions = x
                else:
                    stored_predictions = torch.cat((stored_predictions, x), 1)
                
            outputs.append(x)
        
        return stored_predictions


    def load_weights(self, file_name):
        with open(file_name, 'rb') as weights_file:

            # First 5 values are header information, i.e. version numbers and num images seen during training
            header = np.fromfile(weights_file, dtype = np.int32, count = 5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]  # store # images seen during training

            weights = np.fromfile(weights_file, dtype = np.float32)

            weights_i = 0
            for i in range(len(self.module_list)):

                # only load weights if this module is a convolutional module
                if self.blocks[i + 1]['type'] == 'convolutional':  # +1 as blocks array includes network information as 1st block
                    module = self.module_list[i]

                    # check if batch normalization was specified
                    if 'batch_normalize' in self.blocks[i+1]:
                        batch_normalize = self.blocks[i+1]['batch_normalize']
                    else:
                        batch_normalize = 0

                    conv = module[0]

                    if batch_normalize:

                        ## load weights for batch normalize layer

                        batch_norm = module[1]

                        # number of weights in batch norm layer
                        num_weights = batch_norm.bias.numel()

                        # load weights for biases and regular weights
                        bn_biases = torch.from_numpy(weights[weights_i:weights_i + num_weights])
                        weights_i += num_weights

                        bn_weights = torch.from_numpy(weights[weights_i: weights_i + num_weights])
                        weights_i  += num_weights

                        bn_running_mean = torch.from_numpy(weights[weights_i: weights_i + num_weights])
                        weights_i  += num_weights

                        bn_running_var = torch.from_numpy(weights[weights_i: weights_i + num_weights])
                        weights_i  += num_weights

                        # cast these loaded weights to the dimensions of the models weights
                        bn_biases = bn_biases.view_as(batch_norm.bias.data)
                        bn_weights = bn_weights.view_as(batch_norm.weight.data)
                        bn_running_mean = bn_running_mean.view_as(batch_norm.running_mean)
                        bn_running_var = bn_running_var.view_as(batch_norm.running_var)

                        # copy weights to the model
                        batch_norm.bias.data.copy_(bn_biases)
                        batch_norm.weight.data.copy_(bn_weights)
                        batch_norm.running_mean.copy_(bn_running_mean)
                        batch_norm.running_var.copy_(bn_running_var)

                    else:

                        ## loads biases of the conv layer (don't need to do this if there is batch norm layer
                        ## as the conv layer does not have biases in this case)

                        # number of biases
                        num_biases = conv.bias.numel()

                        # load the biases
                        conv_biases = torch.from_numpy(weights[weights_i: weights_i + num_biases])
                        weights_i = weights_i + num_biases

                        # reshape loaded biases to shape of model's biases
                        conv_biases = conv_biases.view_as(conv.bias.data)

                        # copy the biases to the model
                        conv.bias.data.copy_(conv_biases)

                    ## load weights of conv layer itself

                    num_weights = conv.weight.numel()

                    conv_weights = torch.from_numpy(weights[weights_i:weights_i+num_weights])
                    weights_i = weights_i + num_weights

                    conv_weights = conv_weights.view_as(conv.weight.data)
                    conv.weight.data.copy_(conv_weights)