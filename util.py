from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

def parse_cfg(config_file):
    """
        Parses the input configuration file, outputting a list of blocks.
        Each block is a dictionary, with each key-value pair corresponding to
        a line in the input configuration file.
    """
    with open(config_file, 'r') as file:

        # get all lines from config file, filtering out all comments and empty lines
        lines = [x.strip() for x in file.read().split('\n') if len(x) > 0 and x[0] != '#']

        curr_block = {}
        blocks = []
        for line in lines:
            if line[0] == '[':  # start of a new block
                if len(curr_block) != 0:  # if block is not empty (i.e. not at first line of file)
                    blocks.append(curr_block)
                    curr_block = {}
                curr_block['type'] = line[1:-1].strip()
            else:
                key, value = line.split('=')
                curr_block[key.strip()] = value.strip()
        blocks.append(curr_block)  # append final block

    return blocks

def bbox_iou(box1, box2):
    """
        Returns the IoU (Intersection over Union) of two sets of bounding boxes    
    """
    # get bounding box coords
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    # calculate area of intersection
    inter_area = torch.clamp(torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1) + 1, min=0) * \
                 torch.clamp(torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1) + 1, min=0) # could be < 0 if no intersection, hence clamp()

    # area of union
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

def transform_prediction(prediction, input_dim, anchors, n_classes, use_cuda):
    """
        Converts the input prediction into a 2D tensor representation.
        The nth row of the tensor stores the objectness score, class probablities, and dimensions
        of the n%3 bounding box at position ((n%W), H*(n//W)) (W, H = grid dimensions) in the feature map.
        Basicallly, it just converts the prediction feature map into an easier to process format.
    """

    batch_size = prediction.size(0)
    stride = input_dim // prediction.size(2)
    grid_size = input_dim // stride
    n_bbox_attrs = 5 + n_classes

    # convert (W,H,len(anchors)*(5 + C)) tensor into a (W*H, len(anchors)*(5+C)) tensor (i.e. 2D)
    prediction = prediction.view(batch_size, n_bbox_attrs * len(anchors), pow(grid_size, 2))
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, pow(grid_size, 2) * len(anchors), n_bbox_attrs)

    # anchors' dimensions are specified in terms of the input image dimensions
    # so we have to reduce them in size by stride so that their dimensions
    # are described in terms of the feature map dimensions.
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # apply sigmoid activation function to the bounding box centre coords and object confidence
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    ## add grid cell offsets to the bounding box center coordinates predictions

    # create mesh grids
    grid_i_arr = np.arange(grid_size)
    xgrid, ygrid = np.meshgrid(grid_i_arr, grid_i_arr)

    # flatten these mesh grids
    x_offset = torch.FloatTensor(xgrid).view(-1,1)
    y_offset = torch.FloatTensor(ygrid).view(-1,1)

    if use_cuda:  # put tensors on cuda device if specified
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    # concatenate the offset meshes, and repeat for each anchor
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,len(anchors)).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    ## apply the log space transforms to the height and width of the anchors

    anchors = torch.FloatTensor(anchors)

    if use_cuda:
        anchors = anchors.cuda()

    anchors = anchors.repeat(pow(grid_size, 2), 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    # apply sigmoid activation to the class confidence scores
    prediction[:,:,5:] = torch.sigmoid(prediction[:,:,5:])

    # resize bounding box coords and dimensions to scale of input image
    prediction[:,:,:4] *= stride

    return prediction

def filter_predictions(prediction, confidence, num_classes, nms_thresh = 0.4):
    
    # convert center-based coordinates to corner coordinates
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]    

    # filter out predictions with insufficient confidence
    image_pred = torch.stack([pred for pred in prediction[0] if pred[4] > confidence])
    if len(image_pred) == 0:  # no predictions with sufficient confidence
        return None

    # transform predictions to only specify class with maximal confidence (and said confidence)
    class_conf, class_i = (tensor.unsqueeze(1) for tensor in torch.max(image_pred[:,5:5 + num_classes], 1))
    image_pred = torch.cat((image_pred[:,:5], class_conf, class_i), 1)

    # get the object prediction classes
    obj_classes = {clss.item() for clss in image_pred[:,-1]}  # -1 index holds the class index
    
    # perform NMS on a class-by-class basis
    output = []
    for clss in obj_classes:

        # get the detections with this class
        image_pred_class = torch.stack([pred for pred in image_pred if pred[-1] == clss])
        
        # sort the detections in reverse conf order (higher conf take priority)
        sort_index = torch.sort(image_pred_class[:,4], descending=True)[1]
        image_pred_class = image_pred_class[sort_index].view(-1, 7)
        
        i = 0
        while i < image_pred_class.size(0)-1: # loop over predictions
            # get the IOUs of all boxes that come after current one
            ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
        
            # remove the predictions with IoU > threshold
            filtered_preds = [image_pred_class[i]] + [image_pred_class[i+1] for i, iou in enumerate(ious) if iou.item() < nms_thresh]

            if len(filtered_preds) == 0:  # no predictions for this class
                break

            image_pred_class = torch.stack(filtered_preds)
            i += 1

        output.extend([pred for pred in image_pred_class])
    
    return torch.stack(output) if len(output) > 0 else None

def to_letterbox_image(img, input_size):
    """
        Resizes img to have both its width and height equal to input_size, while maintaining
        its aspect ratio (empty space is filled with grey). 
    """

    img_w, img_h = img.shape[1], img.shape[0]
    scale_factor = min(input_size/img_w, input_size/img_h)

    new_w = int(img_w * scale_factor)
    new_h = int(img_h * scale_factor)

    resized_image = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
    
    # create 'canvas' with size (input_size, input_size)
    canvas = np.full((input_size, input_size, 3), 128)

    # draw image in center of canvas (leaving the grey padding)
    w, h = input_size, input_size  # simply renaming to make below expression cleaner
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w, :] = resized_image
    
    return canvas

def prep_image(img, input_size):
    """
        Prepares the input img for inputting to the neural network
        (i.e. sets its dimensions to be (input_size, input_size)) using letterbox
        padding, and converts it to a tensor. 
    """
    img = to_letterbox_image(img, input_size)
    img = img[:,:,::-1].transpose((2,0,1)).copy()  # convert BGR to RGB

    return torch.from_numpy(img).float().div(255.0).unsqueeze(0)  # divide by 255.0 to normalize rgb values (from 0->1)

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names