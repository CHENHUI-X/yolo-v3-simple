from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random
from matplotlib import pyplot as plt

def arg_parse():
  """
  Parse arguements to the detect module

  """

  parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

  parser.add_argument("--images_path", dest='images_path', help=
  "Image / Directory containing images to perform detection upon",
                      default="imgs", type=str)
  parser.add_argument("--det", dest='det', help=
  "Image / Directory to store detections to",
                      default="det", type=str)
  parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
  parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
  parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
  parser.add_argument("--cfg", dest='cfgfile', help=
  "Config file",
                      default="cfg/yolov3.cfg", type=str)
  parser.add_argument("--weights", dest='weightsfile', help=
  "weightsfile",
                      default="yolov3.weights", type=str)
  parser.add_argument("--reso", dest='reso', help=
  "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                      default="416", type=str)

  return parser.parse_args()


def load_classes(namesfile):
  fp = open(namesfile, "r")
  names = fp.read().split("\n")[:-1] # 最后多了一行空白行
  return names

args = arg_parse()
images = args.images_path # input
batch_size = int(args.bs) # default = 1
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()
num_classes = 80    #For COCO
classes = load_classes("data/coco.names")

#Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"]) # weight  == height

assert inp_dim % 32 == 0
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()

#Set the model in evaluation mode
model.eval()
read_dir = time.time()
#Detection phase
try:
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print ("No file or directory with the name {}".format(images))
    exit()

if not os.path.exists(args.det): # store the detection result
    os.makedirs(args.det)

# print(imlist)
load_batch = time.time()
loaded_ims = [cv2.imread(x) for x in imlist]  # shape :(H,W,C)
# print(loaded_ims[0].shape)# (452, 602, 3)

'''
  
OpenCV loads an image as an numpy array, 
with BGR as the order of the color channels. 
PyTorch's image input format is (Batches x Channels x Height x Width),
with the channel order being RGB.
Therefore, we write the function prep_image in util.py to 
transform the numpy array into PyTorch's input format.

meanwhile , we must process the image to fix shape,so
we must write a function letterbox_image(also in util.py)
that resizes our image, keeping the aspect ratio consistent, 
and padding the left out areas with the color (128,128,128)

'''
# next we need the original image to show the final result (like box position )
#PyTorch Variables for images

im_batches = list(map(letterbox_image, loaded_ims, [inp_dim for x in range(len(imlist))]))
# im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))
# output is (C,H,W) with inputdim and padding

# get   dimensions of  all original images, transform (W,H)
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
# shape[0]:H,  shape[1] :W -> (W,H)

# im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
im_dim_list = torch.FloatTensor(im_dim_list)
# (W,H,W,H) , but why repeat(1，2)

if CUDA:
    im_dim_list = im_dim_list.cuda()


leftover = 0
if (len(im_dim_list) % batch_size):
  # means we need a addition batch for rest images
   leftover = 1

if batch_size != 1:
   num_batches = len(imlist) // batch_size + leftover
   im_batches = [
     torch.cat(
       (im_batches[i*batch_size : min((i +  1)*batch_size, len(im_batches))])
     )
     for i in range(num_batches)
   ]

# The Detection Loop
write = 0
start_det_loop = time.time()
for i, batch in enumerate(im_batches):
    #load the image
    start = time.time()
    if CUDA:
        batch = batch.cuda()

    # prediction = model(Variable(batch, volatile = True), CUDA)
    # volatile = True means that do not calculate the gradient
    prediction = model(batch.detach(), CUDA)
    # print(prediction.shape)  # torch.Size([1, 10647, 85])
    prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thesh)
    # it defined in util.py, the function output is like follow :
    # [ID,box_left, box_top, box_right, box_down, box_conf, max_class_score, max_class_score_index]
    # for every row

    '''
        the output is like follow (assume batch size = 64) :
        (1, box_left, box_top, box_right, box_down, box_conf, max_class_score, max_class_score_index) 
        # 1 indicator the image index in current batch 
        (1, box_left, box_top, box_right, box_down, box_conf, max_class_score, max_class_score_index)
        
          ...... some rows ,notice rows not the 3*(13*13+26*26+52*52),
          because we has filter some box 
                          
        (2, box_left, box_top, box_right, box_down, box_conf, max_class_score, max_class_score_index)
        (2, box_left, box_top, box_right, box_down, box_conf, max_class_score, max_class_score_index)  
        
          ...... some rows ,notice rows not the 3*(13*13+26*26+52*52),
          because we has filter some box 
                          
        (64,box_left, box_top, box_right, box_down, box_conf, max_class_score, max_class_score_index)
        
          ...... some rows ,notice rows not the 3*(13*13+26*26+52*52),
          because we has filter some box 
        
    '''
    end = time.time()

    if type(prediction) == int:
        # If the output of the write_results function for batch is an int(0),
        # meaning there is no detection result ,
        # but it still has to spend time calculating and going to next batch
        # we just use continue to skip the rest loop.

        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("\\")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue
    # transform the atribute 0(index in that batch )
    # from index in batch to index in all image(im_list)
    prediction[:,0] += i*batch_size
    '''
    *************(assume batch size = 64 and current  batch  is the 2th batch )*********
    now the prediction is like this :
            (65, box_left, box_top, box_right, box_down, box_conf, max_class_score, max_class_score_index)
            (65, box_left, box_top, box_right, box_down, box_conf, max_class_score, max_class_score_index)
                ...... some  rows (notice rows not the 3*(13*13+26*26+52*52),
                because we has filter some box  
                
            (67, box_left, box_top, box_right, box_down, box_conf, max_class_score, max_class_score_index)
            (67, box_left, box_top, box_right, box_down, box_conf, max_class_score, max_class_score_index)  
                ...... some  rows (notice rows not the 3*(13*13+26*26+52*52),
                because we has filter some box  
                
            (128,box_left, box_top, box_right, box_down, box_conf, max_class_score, max_class_score_index)
                                ....
            (box_left, box_top, box_right, box_down, box_conf, max_class_score, max_class_score_index)
            ...... some  rows (notice rows not the 3*(13*13+26*26+52*52),
                because we has filter some box  
                             
        ****  where 67 actually indicator that predict is belong 2th image in current  batch , when batch is 2  ****
    '''

    if not write:                      #If we have't initialised output
        output = prediction
        write = 1
    else:
        output = torch.cat((output,prediction))
        # stack the prediction  batch by batch

    # we only concern these  images of current batch size
    for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
        im_id = i*batch_size + im_num # im_num indicator the  image  index in current batch
        # then Get the detection result belonging to the current picture
        # from all detection results
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("\\")[-1], (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()

# Drawing bounding boxes on images
try:
    output
    '''
        (assume batch size = 64 )
        now the output is like this
         :
                (1, box_left, box_top, box_right, box_down, box_conf, max_class_score, max_class_score_index)
                (1, box_left, box_top, box_right, box_down, box_conf, max_class_score, max_class_score_index)
                    ...... some  rows (notice rows not the 3*(13*13+26*26+52*52),
                    because we has filter some box  
                (2, box_left, box_top, box_right, box_down, box_conf, max_class_score, max_class_score_index)
                (2, box_left, box_top, box_right, box_down, box_conf, max_class_score, max_class_score_index)   
                    ...... some  rows (notice rows not the 3*(13*13+26*26+52*52),
                    because we has filter some box 

                (128,box_left, box_top, box_right, box_down, box_conf, max_class_score, max_class_score_index)
                                    ....
                (128,box_left, box_top, box_right, box_down, box_conf, max_class_score, max_class_score_index)
                     ...... some  rows (notice rows not the 3*(13*13+26*26+52*52),
                    because we has filter some box 
                    .......
            where 128 indicator that predict is belong 128th image in all images
        '''

except NameError:
    print ("No detections were made")
    exit()

output_recast = time.time()
'''
1.
Before we draw the bounding boxes, 
the predictions contained in our output tensor 
conform to the input size of the network, 
and not the original sizes of the images. 
thus ,the predictions contained in our output tensor 
are predictions on the "padded" image, and not the original image.
2.
Merely, re-scaling them to the dimensions of the input image 
won't work here. We first need to transform the 
coordinates of the boxes to be measured with respect to 
boundaries of the area on the padded image that contains the original image.

'''

# get of image original shape
im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
# calculate scaling factor for all inmage
scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)

output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
'''
这里解释一下: 我们现在输出的结果box坐标是在有padding的图像上的坐标,举个例子
                *  *  *  *  *  *  
                *  -  -  -  -  * 
                *  |  0  0  |  * 
                *  |  0  0  |  * 
                *  -  -  -  -  * 
                *  *  *  *  *  *
这里中间0是原图像经过乘一个scaling factor（比如1/4），其余*号为padding，
假设这里预测有一个框，由于传入CNN的是经过 padding 的image，那么
prediction 的框的对角坐标 ，是对应到padding image 的，因此真实的
框对角坐标，需要将现有坐标减去左边界和上边界即可
(因为之前保证了图片在convas的正中心)

'''
output[:,1:5] /= scaling_factor
#再把结果resize回去 ,之前预测用图片为original image * scaling_factor

# clip the box cross image boundary
for i in range(output.shape[0]):
    # im_dim_list[i,0] : weigh im_dim_list[i,1] : height
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
    # 裁剪框的坐标，不超过image边界

'''
NOW WE CAN DRAW THE BOX
If there are too many bounding boxes in the image,
 drawing them all in one color may not be such a nice idea.
'''
class_load = time.time()
colors = pkl.load(open("./pallete", "rb"))
draw = time.time()
# print(colors,len(colors))

def drawbox(x, results, color):
    # x : (ID,x,y,w,h,max_conf,max_conf_index)
    c1 = tuple([int(i.item()) for  i in x[1:3]]) # object rectangle : ( left ,top )
    c2 = tuple([int(i.item()) for  i in x[3:5]]) # object rectangle : ( right ,down  )
    img = results[int(x[0])] # image  ID
    cls = int(x[-1]) # clas
    label = "{0}".format(classes[cls])
    cv2.rectangle(img,c1,c2, color, 1)

    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)

    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);

    return img

list(map(lambda x: drawbox(x, loaded_ims,random.choice(colors)), output))
# parameter is the list, so original image is also changed
# drawbox is defined in util.py

det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det,x.split("\\")[-1]))
list(map(cv2.imwrite, det_names, loaded_ims))

end = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
print("----------------------------------------------------------")
torch.cuda.empty_cache()





