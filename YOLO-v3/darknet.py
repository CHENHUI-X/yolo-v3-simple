# from __future__ import division
# https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from util import *

def parse_cfg(cfgfile):
    """
    Takes a configuration file : e.g yolo-v3.cfg

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    store every block as a dict

    """
    # 获取网络架构安排
    with open(cfgfile, 'r',encoding = "utf-8") as f:
        lines = f.read().split('\n') # store the lines in a list

    lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']  # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    block = {} # parameter  of  each  layer
    blocks = [] # store the layers
    for line in lines:
        if line[0] == "[":  # This marks the start of a new block，like [convolutional]
            if len(block) != 0:
                # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)  # add it( last one  block ) the blocks list
                block = {}  # re-init the block ,for  update  next  time layer

            block["type"] = line[1:-1].rstrip() # net or rote or convolutional or upsample
        else:
            key, value = line.split("=") # add it in the above block
            block[key.rstrip()] = value.lstrip()
    blocks.append(block) # add the  last  layer  in  layers list
    """
    return : 
    [{'type': 'net',
      'batch': '64',
      'subdivisions': '16',
      'width': '608',
      'height': '608',
      'channels': '3',
      ...
      'max_batches': '500200',
      'policy': 'steps',
      'steps': '400000,450000',
      'scales': '.1,.1'},..{}...]
    """
    return blocks

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
        # do nothing
        # just a layer for get a index or a position

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def create_modules(blocks):
    '''

    :param blocks: list  of  layers

    :return:

    The create_modules function takes a list blocks returned by the parse_cfg function.
    '''
    # Captures the information about the input and pre-processing : some  parameters
    net_info = blocks[0] # a dict

    module_list = nn.ModuleList() # element is a Sequential one by one

    # we need  to  know  how many  channel(deepth) out from previous layer
    #  here previous channel  is the input image , which is 3
    prev_filters = 3
    # store   filter num  of  all  the  kernel for  route layer
    output_filters = []

    #  iterate over the list of blocks, and create a PyTorch module for each block
    for index, x in enumerate(blocks[1:]):
        # a block(dic) maybe have  some layers,e.g  normal + conv + relu
        module = nn.Sequential()

        # check the type of block
        # create a new module for the block
        # append to module_list
        # a block of type convolutional has a batch norm layer as well as leaky ReLU
        # We string together these layers using the nn.Sequential

        if (x["type"] == "convolutional"):
            # Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                # if have a normal ,then the conv layer not need bias
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2  # make output size same with  input
            else:
                pad = 0

            # Add the convolutional layer
            # the input shape should be (N,C,H,W)
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            # Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # Check the activation.
            # It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

            # If it's an upsampling layer
            # We use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="nearest") # or bilinear
            module.add_module("upsample_{}".format(index), upsample)

            # If it is a route layer
        elif (x["type"] == "route"):
            # layer  = -4，means that getoutput from  previous 4 layer
            x["layers"] = x["layers"].split(',')
            # Start  of a route
            start = int(x["layers"][0])
            # like -1, 61 means concatenation them with  channel dimension
            # end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            # Positive anotation
            if start > 0:
                start = start - index # index : current layer
            if end > 0:
                end = end - index
            route = EmptyLayer() # just a virtual layer ,do not anything but need a position
            module.add_module("route_{0}".format(index), route)
            # here just  updated the  channel num , the real operation write in the main forward
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

            # shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            # here  do nothing  , all operation in the  main  forward
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",") # get which anchor size used
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",") # w,h of anchor
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask] # get which anchor size used

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        # current num  of output channel is next input num of  channel
        # (for conv layer ) : so we need update the parameter
        prev_filters = filters
        output_filters.append(filters) # store the num of filter of  every time
    return  (net_info, module_list) # info of  net (a dictionary ) and layer of model

# blocks = parse_cfg("yolov3.cfg")
# print(create_modules(blocks))

# define  the  main  net work structure
class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile) # get net info : [{info of a block },...]
        self.net_info, self.module_list =   \
            create_modules(self.blocks) # get layer of model

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        # [0] is the net info (some parameter )
        # [{info of a block},{'type': 'convolutional','filter' : '3' ...} ...]

        outputs = {}   #We cache the outputs for the route layer
        write = 0  # indicator that

        for i, module in enumerate(modules):
            module_type = (module["type"])
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)

            elif module_type == "route":
                # route layer have a parameter like: layers = -4 or -1,61
                layers = module["layers"]
                layers = [int(a) for a in layers] # get all index need Concatenates
 
                if (layers[0]) > 0:
                    # here repeat the judge operation  is
                    #  because of  the layer info is new getted from cfg
                    layers[0] = layers[0] - i # I is current layer index

                if len(layers) == 1:
                    x = outputs[i + (layers[0])] # just output from that layer

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1) # merge it in channel dimension
                    # dim = 1 because of our input is (N,C,W,H)
                    # where C is  num  of channel dim

            elif module_type == "shortcut":
                from_ = int(module["from"]) #from = -3 means add the output from previous 3 layer
                # output_i = input_i(that's output_i-1) + output_i+from_
                x = outputs[i - 1] + outputs[i + from_]

            elif module_type == 'yolo':

                anchors = self.module_list[i][0].anchors
                # module_list[i] : instead it's  a sequential,
                # while only have a layer : detection layer
                # so module_list[i][0] just call the detection layer
                # and the layer have a Attributes : anchor

                # Get the input size (original image )
                inp_dim = int(self.net_info["height"])

                # Get the number of classes
                num_classes = int(module["classes"])

                # Transform
                x = x.data # calculate from conv_output_layer
                """
                    well,how to process the output ?
                    the data is a convolutional output (e.g 13*13*[5 + 80] )that contains the bounding box 
                    attributes along the depth of the feature map. The attributes bounding boxes 
                    predicted by a cell are stacked one by one along each other. 
                    So, if you have to access the second bounding of cell at (5,6),
                    then you will have to index it by map[5,6, (5+C): 2*(5+C)]. 
                    This form is very inconvenient for output processing 
                    such as thresholding by a object confidence, 
                    adding grid offsets to centers, applying anchors etc.
                    
                    Another problem is that since detections happen at three scales,
                    the dimensions of the prediction maps will be different. 
                    Although the dimensions of the three feature maps are different,
                    the output processing operations to be done on them are similar.
                    It would be nice to have to do these operations on a single tensor,
                    rather than three separate tensors.
                    
                    To remedy these problems, we introduce the function predict_transform
                    which defined in until.py
                
                """
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                # now the x have a same size although it come from different future map
                if not write:  # if no collector has been intialised.
                    detections = x
                    write = 1

                else: # concatenate the detection output from different level FM
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x
            # Record the output of each layer
        print('detection shape: {} '.format(detections.shape)) # (batchsize,
        return detections

    # now you can load the pretrain weight from author
    # The official weights file is binary file that contains weights stored in a serial fashion.
    # First, the weights belong to only two types of layers,
    # either a batch norm layer or a convolutional layer.
    # because the shortcut and route do not need any parameter
    # the structure is following here :
    # if the conv layer(block or  sequential ) has a batch normal layer
    #   then  we do not add the bias to conv
    #   so the weight stored as : bn_bias bn_weight bn_mean bn_var conv_weight
    # if  the conv do not  followed  by the normal layer
    #   then wo add the bias for output of conv
    #   so the weight stored as : conv_bias , conv_weights
    def load_weights(self, weightfile):
        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        #The weights are stored as float32 or 32-bit floats.
        # Let's load rest of the weights in a np.ndarray.
        # current pointed have position at 5
        weights = np.fromfile(fp, dtype=np.float32)
        ptr = 0 # pointer for read weight

        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"] # i = 0 is the info for net

            # If module_type is convolutional load weights
            # Otherwise ignore.that is other layer do not need weight
            # we first check whether the convolutional block has
            # batch_normalise True or not. Based on that, we load the weights.
            if module_type == "convolutional":
                # the model sequential  is : conv -> normal -> active function
                model = self.module_list[i] # the current layer ( or a sequential )
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0
                # the model sequential  is : conv -> normal -> active function
                conv = model[0]


                if (batch_normalize):
                    # the model sequential  is : conv -> normal -> active function
                    bn = model[1] # batch normal layer

                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data) # resize
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else:
                    # Number of biases
                    # if have no normal layer,then conv must to have  the bias
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # last , Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


# now test your net
def get_test_input():
    img = cv2.imread("./test/dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W X C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

def random_test():
    model = Darknet("yolov3.cfg")
    inp = get_test_input()
    pred = model(inp, torch.cuda.is_available())
    print(pred)
    print(pred.shape)
    # ([1, 10647, 85]):where 85 = ( 5 + 80)
    # 10647 = 3 * ( 13*13 + 26*26 + 52*52) 3 different level FM

def pretrain_test():

    model = Darknet("yolov3.cfg")
    model.load_weights("yolov3.weights")
    inp = get_test_input()
    pred = model(inp, torch.cuda.is_available())
    print(pred)
    print(pred.shape)

if __name__ == '__main__':
    # random_test() # the weight are  random
    pretrain_test() # the weight are from pretrain


















