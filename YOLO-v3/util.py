from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
        '''

        :param prediction:
                prediction is a tensor feature(out) map with shape (N , C , W , H)
                where C = 255 for class num = 80 ,means that 3 * ( x,y,w,h,conf, 80 class)

        :param inp_dim:
        :param anchors: width , height - 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326

        :param num_classes: 80
        :param CUDA:
        :return:a 2-D tensor, where each row of the tensor corresponds to
                attributes of a bounding box(same level feature map),
                like following  :
                        1st box at(0,0)
                        2nd box at(0,0)
                        3rd box at(0,0)
                        1st box at(0,1)
                        2nd box at(0,1)
                        3rd box at(0,1)...
        '''
        # prediction is a tensor feature map : ( N , (5 + num of class as same as channel ) , W , H) for predict :
        batch_size = prediction.size(0) # N (batch size)
        # because the anchor size is according image size ,
        #  for feature map it's to large
        # Therefore, we must divide the anchors by the
        # stride of the detection feature map.
        # stride = input_dim(imgae size) // W e.g  = 416 / 13 = 32
        #  means move a pixel on  feature map equals move 32 pixel on original image
        stride = inp_dim // prediction.size(2)  # like a scaling factor
        anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
        grid_size = inp_dim // stride
        # get the feature map size , which is prediction.size(3)

        bbox_attrs = 5 + num_classes  # (x,y,w,h,conf) + num of class
        num_anchors = len(anchors)  # anchors num , 3 for a level feature map
        '''
        print('the model output shape is {}'.format(prediction.shape))
                the model output shape is torch.Size([1, 255, 13, 13])
                the model output shape is torch.Size([1, 255, 26, 26])
                the model output shape is torch.Size([1, 255, 52, 52]) 
        '''

        prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
        # here bbox_attrs * num_anchors = ( 5 + e.g. 80) * 3 = 255, just is channel
        # grid_size * grid_size  = num  of  cell on the one feature map
        # now the prediction is ( batchsize , ( 5 + e.g. 80) * 3 = 255, num of tensor（ num of cell)  )

        '''
        print('the view output shape is {}'.format(prediction.shape))
                the view output shape is torch.Size([1, 255, 169])
                the view output shape is torch.Size([1, 255, 676])
                the view output shape is torch.Size([1, 255, 2704])  
        '''

        # then transform it to a column
        prediction = prediction.transpose(1, 2).contiguous()
        # https://stackoverflow.com/questions/48915810/pytorch-what-does-contiguous-do
        # now  the prediction is ( batchsize , num of tensor, dim of tensor which is 3 * 85  )

        # then extract  the box num

        prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
        # now the output is have a tensor , shape:
        # ( batchsize , num of tensor * 3,  85)
        '''
        but how it make the prediction compatible the original data , 
        how to ensure that when the original 100*9 becomes 300*3,
        the new first three lines are composed of the original 
        first line divided into 3 evenly? This is the benefit of "continuity", 
        because the existence of "continuity" enables new results to 
        be "connected" in storage, so that the current 3 consecutive rows 
        are the previous 1 row
        '''

        # Sigmoid the x,y coordinates and the objectness score.
        # Sigmoid the  centre_X, centre_Y. and object confidencce
        prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0]) # offset x related the current cell
        prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1]) # offset y related the current cell
        prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4]) # conf

        # Add the center offsets
        grid = np.arange(grid_size)

        a, b = np.meshgrid(grid, grid)

        x_offset = torch.FloatTensor(a).view(-1, 1)
        y_offset = torch.FloatTensor(b).view(-1, 1)

        if CUDA:
            x_offset = x_offset.cuda()
            y_offset = y_offset.cuda()

        x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
        # cat : connected by column -> (M,2) , repeat : -> ( M , 6)
        # view(-1,2) -> ( 3*M,2) ,
        # unsqueeze(0) : add a dimension for index = 0 to adapt dimension  of prediction
        '''
            now x_y_offset with shape ( num of anchors  * 3 ,  2 )like :
               [[[0., 0.],# 1st box at(0,0)
                 [0., 0.],# 2nd box at(0,0)
                 [0., 0.],# 3rd box at(0,0)
                 [1., 0.],#  ...
                 [1., 0.],#  ...
                 [1., 0.],#  ...
                 [2., 0.],#  ...
                 [2., 0.],#  ...
                 [2., 0.],#  ...
                    . ... ]]
            写代码时注意 0,0 0,0 0,0 1,0 1,0 1,0 2,0 .... 的顺序要和之前output map 拉直以后的索引对应上
            即之前的output map 拉直后的每一行依次为 00位置的第1个框 00位置的框2 00位置的框3 1,0位置的框1 ...
        '''

        prediction[:, :, :2] += x_y_offset # updated offset for center


        # log space transform height and the width
        anchors = torch.FloatTensor(anchors)

        if CUDA:
            anchors = anchors.cuda()

        anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
        # Pw*exp(pred) ，where the Pw is the bounding box prior get by hand or kmeans
        # exp(pred) like a zoom factor , index [2，3] is that pre_W and pre_H
        # so the anchor is  corresponding the w , h , which have  1:2 means w:h = 1 : 2
        prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors
        # jump over prediction[:, :, 4] : the conference

        # Apply sigmoid activation to the the class scores
        prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))
        '''
        The last thing we want to do here, is to 
        resize the detections map to the size of the input image. 
        The bounding box attributes here are sized according to 
        the feature map (say, 13 x 13). If the input image was 416 x 416, 
        we multiply the attributes by 32, or the stride variable.
        '''
        prediction[:, :, :4] *= stride # offset x ,y and the w , h
        # assume stride is 32 ,which means (2,1) on FM equals (64，32) on original image
        # The width and height on the feature map are the same
        return prediction

'''
        Detection Layer Revisited
        Now that we have transformed our output tensors, 
        we can now concatenate the detection maps at three different scales into one big tensor.
        Notice this was not possible prior to our transformation, 
        as one cannot concatenate feature maps having different spatial dimensions. 
        But since now, our output tensor acts merely as a table with bounding boxes as it's rows,
        concatenation is very much possible.
        
        An obstacle in our way is that we cannot initialize an empty tensor, 
        and then concatenate a non-empty (of different shape) tensor to it. 
        So, we delay the initialization of the collector (tensor that holds the detections)
        until we get our first detection map, 
        and then concatenate to maps to it when we get subsequent detections.
        
        Notice the write = 0 line just before the loop in the function forward. 
        The write flag is used to indicate whether we have encountered the first detection 
        or not. If write is 0, it means the collector hasn't been initialized. 
        If it is 1, it means that the collector has been initialized 
        and we can just concatenate our detection maps to it.
        
        Now, that we have armed ourselves with the predict_transform function, 
        we write the code for handling detection feature maps in the forward function.
        
        At the top of your darknet.py file, add the following import.

'''
'''
        When we convert all the feature map results of different levels to the 
        same size and connect, we also need to select the results, 
        that is, combine the confidence threshold and non-maximum suppression 
        to get the final result

'''


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
        '''
        The functions take the prediction as  input
        confidence (objectness score threshold),
        num_classes (80, in our case) and nms_conf (the NMS IoU threshold).
        '''
        conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
        # first index :  batchsize, second index  : is 3*(13*13+26*26+52*52) ,third index: 85
        # unsqueeze(2) is used for make the conf_mask have same dimension
        # then they can mutual on the class conf dimension to make some conf -> 0
        prediction = prediction * conf_mask
        # get conf > conf_threshold box,rest of box is set to be 0

        # now we need calculate the IOU ,it's easier to calculate IoU of two boxes,
        # using coordinates of a pair of diagnal corners of each box.
        #  we transform the (center x, center y, width,height  ) attributes of our boxes,
        #  to (top-left corner x, top-left corner y, right-bottom corner x, right-bottom corner y).
        box_corner = torch.zeros_like(prediction) # crate a tensor like prediction with shape and derive
        box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2) # x - w/2
        box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2) # y - h/2
        box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2) # x + w/2
        box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2) # y - h/2

        prediction[:, :, :4] = box_corner[:, :, :4]
        # now the prediction is
        # batch * 10647 * (box_left,box_top,box_right,box_down，box_conf,80 categories)
        '''
                The number of true detections in every image may be different.
                 For example, a batch of size 3 where images 1, 2 and 3 have 
                 5, 2, 4 true detections respectively. Therefore, confidence thresholding 
                 and NMS has to be done for one image at once. 
                 This means, we cannot vectorise the operations involved, 
                 and must loop over the first dimension of prediction
                (containing indexes of images in a batch)
                    
        '''
        batch_size = prediction.size(0)

        write = False
        # write flag is used to indicate that we haven't initialized

        for ind in range(batch_size):
                image_pred = prediction[ind]
                # all predict for one image of every batch size
                # so the  image_pred  should be a (10647,85) tensor,one row is a
                # tensor([box_left ,box_top,box_right,box_down,box_conf,80 categories])
                # confidence threshholding
                # NMS
                max_class_score , max_class_score_index = torch.max(image_pred[:, 5:5 + num_classes], 1)

                # chose max class score for all box on one  image
                max_class_score = max_class_score.float().unsqueeze(1)
                # print(max_class_score.shape) # (10647，1)

                # a box have a max score . so transformer the tens to a  column :
                #       [[0.5], // box1 on (0,0)
                #        [0.65],// box2 on (0,0)
                #        [0.95], // box3 on (0,0)
                #        [0.63], // box1 on (1,0)
                #          ....
                #        [0] // which is that box output set to be zero because of conf < threshold
                #         ....
                #        [0.61]]
                max_class_score_index = max_class_score_index.float().unsqueeze(1)
                seq = (image_pred[:, :5], max_class_score, max_class_score_index)
                image_pred = torch.cat(seq, 1)
                # print('image_pred'+str(image_pred.shape)) # Size([10647, 7])

                # now the image_pred =
                # [
                # tensor([box_left ,box_top,box_right,box_down,box_conf,max_class_score,max_class_score_index]),
                #                                 .....
                # tensor([box_left ,box_top,box_right,box_down,box_conf,max_class_score,max_class_score_index])
                # ]
                # have 10647 rows ,that is the num of box on one image

                # Remember we had set the bounding box rows having a object
                # confidence less than the threshold to zero? Let's get rid of them

                non_zero_ind = (torch.nonzero(image_pred[:, 4]))
                # get box_conf that not equals zero  for all box on Same image

                try:
                        image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
                        # 7 is dim with
                        # tensor([box_left ,box_top,box_right,box_down,box_conf,max_class_score,max_class_score_index])
                        # the num of rows is that number of  no zero box conf
                except:
                        continue

                # For PyTorch 0.4 compatibility
                # Since the above code with not raise exception for no detection
                # as scalars are supported in PyTorch 0.4
                if image_pred_.shape[0] == 0:
                        continue

                # Get the various classes detected in the image
                img_classes = unique(image_pred_[:, -1])  # -1 index holds the class index
                # first : there are many object on one image
                # second : for one object ,the box has different conf and
                # different predict class about the object
                # so for non-maximal suppress : we need to  execute  NMS
                # for all prediction class one by one ,until get one object for one box
                # Then, we perform NMS classwise.
                for cls in img_classes:
                        # perform NMS
                        # get the detections with one particular class
                        cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
                        # get current  class box that have no zero conf
                        class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
                        # get these conf
                        image_pred_class = image_pred_[class_mask_ind].view(-1, 7) # transformer shape

                        # sort the detections such that the entry with the maximum objectness
                        # confidence is at the top
                        conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1] # get index
                        image_pred_class = image_pred_class[conf_sort_index] # get these box sorted
                        idx = image_pred_class.size(0)  # Number of detections that useful

                        for i in range(idx):
                                # Get the IOUs of all boxes that come after the one we are looking at
                                # in the loop
                                try:
                                        # calculate current box(have a no-zero conf and belong current class : cls)
                                        # about reset box
                                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                                        # return the ious that
                                except ValueError:
                                        break

                                except IndexError:
                                        break

                                # Zero  out all the detections that have IoU > treshhold,which means
                                # someone box IOU with  current box is too large - do not need this  box
                                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                                image_pred_class[i + 1:] *= iou_mask
                                # remain the have little IOU about current box, others are
                                # remove because of they overlap too much with the current box
                                #     最后只有互相IOU都小于阈值的框才能保留下来

                                # get the non-zero entries
                                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                                # index  4 means conf
                                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

                                batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
                                # Repeat the batch_id for as many detections of the class cls in the image
                                # 这个batch的第ind张图片，第cls个类的那些框
                                #                          *****NOTICES******
                                # The batch_ind indicator the predict is belong which image in current batch
                                #  --------------------JUST  IN CURRENT BATCH---------------------

                                seq = batch_ind, image_pred_class

                                if not write:
                                        output = torch.cat(seq, 1)
                                        write = True
                                else:
                                        out = torch.cat(seq, 1) # (index_in_current_batch,image_pred_calss)
                                        output = torch.cat((output, out)) # stack the output row by row
        try:
                return output
        except:
                return 0




def unique(tensor):
        tensor_np = tensor.cpu().numpy()
        unique_np = np.unique(tensor_np)
        unique_tensor = torch.from_numpy(unique_np)

        tensor_res = tensor.new(unique_tensor.shape)
        tensor_res.copy_(unique_tensor)
        return tensor_res


def bbox_iou(box1, box2):
        """
        Returns the IoU of two bounding boxes

        """
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # get the corrdinates of the intersection rectangle
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)

        # Intersection area
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
                inter_rect_y2 - inter_rect_y1 + 1, min=0)

        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area)

        return iou


def  letterbox_image(img, inp_dim):
        '''resize image with unchanged aspect ratio using padding
        img = cv2.imread(r".\test\dog-cycle-car.png")
        img.shape
        Out[58]: (452, 602, 3)  # H,W,C(BGR)
        '''

        img_w, img_h = img.shape[1], img.shape[0] # 0 is h,1 is w
        w, h = (inp_dim,inp_dim) # maybe 416*416 or something else
        new_w = int(img_w * min(w / img_w, h / img_h)) # Take the minimum zoom ratio
        new_h = int(img_h * min(w / img_w, h / img_h)) # then get minimum new image
        resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        # just resize image ,do not change the order of height and width
        # here parameter is needed ( new_W,new_h)

        '''
                这里并不是直接把图片resize到inputdim，而是先计算当前image
                和inputdim之间的比例，选择最小的比例
                比如 inputdim = 128*128，而当前image是 512*256 ： H*W
                那么 比例为 1/4 和 1/2 ，则将当前image resize为 128*64
                then do  follow by these : 
        '''
        # canvas = np.full((inp_dim[1], inp_dim[0], 3), 128) # 创建一个 inputdim*inputdim 全128的image

        canvas = np.full((inp_dim, inp_dim, 3), 128)  # 创建一个 inputdim*inputdim 全128的image
        # 然后把reized_image 填充到这个inputdim*inputdim图片中
        canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

        # set the resized image on center of convas , others position are 128
        canvas = canvas[:, :, ::-1].transpose((2, 0, 1)).copy()
        # then result the image : (C:RGB,H,W) ,because the pytorch conv2d input is (N,C,H,W)
        canvas = torch.from_numpy(canvas).float().div(255.0).unsqueeze(0)

        return canvas


def prep_image(img, inp_dim):
        """
        Prepare image for inputting to the neural network.

        Returns a Variable

        process the image to # BGR->RGB && (H,W,C)->(C,H,W)

        **********************************************************************
         2022-06-17
         this method is instead by above method called letterbox_image
         because that method can be resize image with unchanged aspect ratio
         using padding , so it has better performance

        *********************************************************************

        """

        img = cv2.resize(img, (inp_dim, inp_dim))
        img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
        img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
        return img





