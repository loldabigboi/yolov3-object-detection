import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

# Layer that simply acts as a place-holder
# Used for route and shortcut modules as their functionalities can simply be handled
# in the overarching module implementation's forward() method.
class PlaceholderLayer(nn.Module):
    def __init__(self):
        super(PlaceholderLayer, self).__init__()

# Layer that acts as the output detection layer
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def create_modules(blocks):
    """
        Creates and returns a ModuleList instance, containing all the modules of the model.
        Takes as input the list of all blocks as outputted by parse_cfg.
    """

    network_info = blocks[0]  # information relating to network inputs and input pre-processing
    module_list = nn.ModuleList()

    prev_filters = 3  # number of filters for prev layer - input layer has 3 filters (R G and B)
    output_filters = []  #  need to store all prev filter output dimensions in case of route / shortcut layers

    for i, block in enumerate(blocks[1:]):
        module = nn.Sequential()

        # create 2d conv module
        if block['type']  == 'convolutional':
            
            activation_type = block['activation']

            try:
                batch_normalize = int(block['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters= int(block["filters"])
            padding = int(block["pad"])
            kernel_size = int(block["size"])
            stride = int(block["stride"])

            padding = (kernel_size - 1) // 2 if padding else 0

            # add conv layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, padding, bias=bias)
            module.add_module(f'convolutional_{i}', conv)

            # add batch norm if specified
            if batch_normalize:
                module.add_module(f'batch_norm_{i}', nn.BatchNorm2d(filters))

            # check activation type
            if activation_type == 'linear':
                # do nothing
                pass
            elif activation_type == 'relu':
                # add relu layer
                module.add_module(f'relu_{i}', nn.ReLU(inplace = True))
            elif activation_type == 'leaky':
                # add leaky relu layer
                module.add_module(f'leaky_relu_{i}', nn.LeakyReLU(0.1, inplace = True))
            elif activation_type == 'sigmoid':
                # add sigmoid layer
                module.add_module(f'sigmoid_{i}', nn.Sigmoid())
            else:
                # should not get here
                raise Exception(f'invalid activation type specified: {activation_type}')
        
        # create upsample module
        elif block['type'] == 'upsample':
            # not sure if align_corners should be True or False
            upsample_layer = nn.Upsample(scale_factor = block['stride'], mode = 'bilinear', align_corners=True)
            module.add_module(f'upsample_{i}', upsample_layer)

        # create routing module
        elif block['type'] == 'route':

            block['layers'] = [int(i) for i in block['layers'].split(',')]

            # calculate number of output filters according to layers we are routing from
            filters = 0
            for output_filter_i in block['layers']:
                filters += output_filters[output_filter_i]

            if filters == 0:
                raise Exception('Number of filters should not be 0.')

            module.add_module(f'route_{i}', PlaceholderLayer())
        
        # create shortcut module (skip connection)
        elif block['type'] == 'shortcut':
            module.add_module(f'shortcut_{i}', PlaceholderLayer())

        # create the yolo layer (detection layer)
        elif block['type'] == 'yolo':

            mask = [int(x) for x in block['mask'].split(',')]
            anchors = block['anchors'].split(',')
            anchors = [(int(anchors[i*2]), int(anchors[i*2+1])) for i in mask]

            module.add_module(f'yolo_{i}', DetectionLayer(anchors))

        # should not get to this point unless invalid module type (likely due to parsing / logic error)
        else:
            raise Exception(f'Invalid module type: {block["type"]}')

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (network_info, module_list)