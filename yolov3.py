from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from modules import create_modules
import argparse
import os 
import os.path as osp
from model import DarknetModel
import pickle as pkl

def setup_args():
    """
        Set up the arguments to be parsed at command line.
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Object Detection algorithm')
   
    parser.add_argument('--image', dest = 'img_path', help = 
                        'Path to the image that the algorithm should be run on',
                        default = 'imgs', type = str)
    parser.add_argument('--output', dest = 'output_path', help = 
                        'Directory to store output image in (with BBs and object labels drawn onto it)',
                        default = '', type = str)
    parser.add_argument('--conf_thresh', dest = 'conf_thresh', 
                        help = 'Object Confidence threshold for filtering predictions', default = 0.4)
    parser.add_argument('--nms_thresh', dest = 'nms_thresh', 
                        help = 'NMS threshhold for filtering predictions', default = 0.4)
    parser.add_argument('--cuda', dest = 'use_cuda', 
                        help = 'Flag specifying whether or not CUDA should be used (requires CUDA-enabled GPU).', default = 1)
    
    return parser.parse_args()

def write_pred_to_img(x):

    c1 = tuple(x[0:2].int())
    c2 = tuple(x[2:4].int())

    cls = int(x[-1])

    np.random.seed(cls)  # seed by class index so each class gets same colour
    color = colors[np.random.randint(0,len(colors))]
    label = '{0}'.format(classes[cls])

    cv2.rectangle(loaded_img, c1, c2,color, 3)

    fontScale = 1.2
    fontThickness = 1

    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, fontScale, fontThickness)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4

    cv2.rectangle(loaded_img, c1, c2,color, -1)
    cv2.putText(loaded_img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, fontScale, [225,255,255], fontThickness)
    return loaded_img
    
args = setup_args()
if args.use_cuda and not torch.cuda.is_available():  # CUDA requested without CUDA-enabled GPU
    raise Exception('Input Error: cannot use CUDA without a CUDA-enabled GPU (call with flag --cuda 0).')

# Initialise model
print('Initialising model.....')
blocks = parse_cfg('data/yolov3.cfg')
network_info, module_list = create_modules(blocks)
model = DarknetModel(blocks, network_info, module_list, args.use_cuda)
model.load_weights('data/yolov3.weights')
print('Model successfully initialised.\n')

# Put model on CUDA-enabled GPU if one is available 
if args.use_cuda:
    model.cuda()

# Disable batch-norm statistics in model
model.eval()

input_size = 416

read_dir = time.time()

# If output directory does not exist, we create it
if args.output_path != '' and not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

loaded_img = cv2.imread(args.img_path)
img_input = prep_image(loaded_img, input_size)

if args.use_cuda:
    img_input.cuda()

classes = load_classes('data/coco.names')
num_classes = 80
    
pred_start = time.time()

# get predictions
with torch.no_grad():
    prediction = model(img_input)

output = filter_predictions(prediction, args.conf_thresh, num_classes, nms_thresh = args.nms_thresh)

pred_end = time.time()

if output is None:
    print('No objects were detected in the image :(')
    exit()

objs = [ classes[int(x[-1])] for x in output ]
print('{0:20s} predicted in {1:6.3f} seconds'.format(args.img_path.split('/')[-1], pred_end - pred_start))
print('{0:20s} {1:s}'.format('Objects Detected:', ' '.join(objs)))
print('----------------------------------------------------------')

if args.use_cuda:
    torch.cuda.synchronize()

scaling_factor = torch.min(input_size / torch.Tensor(loaded_img.shape)).item()

# convert coordinates and dimensions of the bounding boxes to the scale of the original image
output[:,[0,2]] -= (input_size - scaling_factor * loaded_img.shape[1])/2
output[:,[1,3]] -= (input_size - scaling_factor * loaded_img.shape[0])/2
output[:,0:4] /= scaling_factor  

# clamp the boundaries of the bounding boxes to the region of the image
for i in range(output.shape[0]):
    output[i, [0,2]] = torch.clamp(output[i, [0,2]], 0.0, loaded_img.shape[1])
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, loaded_img.shape[0])
    
output_recast = time.time()
class_load = time.time()
colors = pkl.load(open('data/pallete', 'rb'))

draw = time.time()

list(map(lambda x: write_pred_to_img(x), output))

# write altered image to output path
cv2.imwrite(args.output_path + 'output-' + args.img_path.split('/')[-1], loaded_img)

torch.cuda.empty_cache()