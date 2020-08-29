import tensorflow as tf
import time
import os
import json
import sys
import itertools
import random
import numpy as np
import math
import tensorflow as tf
from PIL import Image
from cv2 import cv2
from scipy.interpolate import interp1d

class TusimpleLane(tf.data.Dataset):

    # groundSize = (256, 256)
    # map_x = None
    # map_y = None
    # H = None

    def create_map(src_img_size, ground_size, augmentation_deg=None):
        h, w = src_img_size
        gh, gw = ground_size

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
        # calc homography (TuSimple fake)
        imgP = [[501, 362], [781, 357],  [924, 499], [337, 510]]
        groundP = [[-180, 1300], [180, 1300],  [180, 300], [-180, 300]]
        for i in range(4):
            imgP[i][0] *= w / 1280.0
            imgP[i][1] *= h / 720.0
            groundP[i][0] = groundP[i][0] * 0.16 + gw / 2.0
            groundP[i][1] = gh - groundP[i][1] * 0.1

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
        # # calc homography (set_A)
        # imgP = [[361, 298], [600, 295],  [773, 411], [247, 421]]
        # groundP = [[-180, 1300], [180, 1300],  [180, 300], [-180, 300]]
        # for i in range(4):
        #     imgP[i][0] *= 1280.0 / 1024.0
        #     imgP[i][1] *= 720.0 / 576.0
        #     groundP[i][0] = groundP[i][0] * 0.16 + gw / 2.0
        #     groundP[i][1] = gh - groundP[i][1] * 0.1
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
        # # calc homography (set_B)
        # imgP = [[376, 365], [706, 359],  [731, 407], [349, 413]]
        # groundP = [[-90, 190], [90, 190],  [90, 100], [-90, 100]]
        # for i in range(4):
        #     imgP[i][0] *= 1280.0 / 1024.0
        #     imgP[i][1] *= 720.0 / 576.0
        #     groundP[i][0] = groundP[i][0] * 0.16 + gw / 2.0
        #     groundP[i][1] = gh - groundP[i][1] * 0.1
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

        list_H = []
        list_map_x = []
        list_map_y = []

        groud_center = tuple(np.average(groundP, axis=0))
        if augmentation_deg is None:
            augmentation_deg = [0.0]

        for deg in augmentation_deg:
            R = cv2.getRotationMatrix2D(groud_center, deg, 1.0)
            rotate_groupP = []
            for gp in groundP:
                pp = np.matmul(R, [[gp[0]], [gp[1]], [1.0]])
                rotate_groupP.append([pp[0], pp[1]])
              
            H, _ = cv2.findHomography(np.float32(imgP), np.float32(rotate_groupP))
            _, invH = cv2.invert(H)
                
            map_x = np.zeros((gh, gw), dtype=np.float32)
            map_y = np.zeros((gh, gw), dtype=np.float32)
                
            for gy in range(gh):
                for gx in range(gw):
                    nx, ny, nz = np.matmul(invH, [[gx], [gy], [1.0]])
                    nx /= nz
                    ny /= nz
                    if (nx >= 0 and nx < w and ny >= 0 and ny < h):
                        map_x[gy][gx] = nx
                        map_y[gy][gx] = ny
                    else:
                        map_x[gy][gx] = -1
                        map_y[gy][gx] = -1
            
            list_H.append(H)
            list_map_x.append(map_x)
            list_map_y.append(map_y)


        return list_H, list_map_x, list_map_y

    # ----------------------------------------------------------------------------------------
    def __new__(self,
                dataset_path,
                label_set,
                net_input_img_size,
                x_anchors,
                y_anchors,
                max_lane_count,
                ground_img_size = (256, 256),
                augmentation=False):
        if (os.path.exists(dataset_path) == False):
            print("File doesn't exist, path : ", dataset_path)
            exit()

        # build map
        augmentation_deg = None
        if augmentation:
            augmentation_deg=[0.0, -15.0, 15.0, -30.0, 30.0]
        else:
            augmentation_deg=[0.0]
        
        H_list, map_x_list, map_y_list = self.create_map(src_img_size=(720, 1280),
                                                         ground_size=ground_img_size,
                                                         augmentation_deg=augmentation_deg)
        H_list = tf.constant(H_list)
        map_x_list = tf.constant(map_x_list)
        map_y_list = tf.constant(map_y_list)

        # build dataset
        label_set = tf.data.Dataset.from_tensor_slices(label_set)
        pipe = label_set.interleave(
            lambda label_file_name: tf.data.Dataset.from_generator(self._data_reader,
                                               output_types=(tf.dtypes.uint8,
                                                             tf.dtypes.int32,
                                                             tf.dtypes.int32,
                                                             tf.dtypes.int32),
                                               output_shapes=((None),
                                                              (None),
                                                              (None),
                                                              (None)),
                                               args=(dataset_path,
                                                     label_file_name,
                                                     augmentation_deg)
                                               )
            )

        pipe = pipe.map (
                # convert data to training label and norimalization
                lambda image, label_lanes, label_h_samples, refIdx : 
                    tf.numpy_function(func=self.tttmap_projection_data_generator,
                                   inp=[image,
                                        label_lanes,
                                        label_h_samples,
                                        net_input_img_size,
                                        x_anchors,
                                        y_anchors,
                                        max_lane_count,
                                        H_list[refIdx],
                                        map_x_list[refIdx],
                                        map_y_list[refIdx],
                                        ground_img_size],
                                   Tout=[tf.dtypes.float32,
                                         tf.dtypes.float32])
                ,
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )

        # pipe = pipe.map (
        #         # convert data to training label and norimalization
        #         lambda image, label_lanes, label_h_samples, refIdx : 
        #             tf.numpy_function(func=self.map_instance_segmentation_data_generator,
        #                            inp=[image,
        #                                 label_lanes,
        #                                 label_h_samples,
        #                                 net_input_img_size,
        #                                 x_anchors,
        #                                 y_anchors,
        #                                 max_lane_count,
        #                                 H_list[refIdx],
        #                                 map_x_list[refIdx],
        #                                 map_y_list[refIdx],
        #                                 ground_img_size],
        #                            Tout=[tf.dtypes.float32,
        #                                  tf.dtypes.float32])
        #         ,
        #         num_parallel_calls=tf.data.experimental.AUTOTUNE
        #     )

        pipe = pipe.prefetch(  # Overlap producer and consumer works
                tf.data.experimental.AUTOTUNE
            )
                
        return pipe

    # ----------------------------------------------------------------------------------------
    def map_instance_segmentation_data_generator(src_image,
                                                 label_lanes,
                                                 label_h_samples, 
                                                 net_input_img_size,
                                                 x_anchors,
                                                 y_anchors,
                                                 max_lane_count,
                                                 H,
                                                 map_x,
                                                 map_y,
                                                 groundSize):
        # transform image by perspective matrix
        height, width = src_image.shape[:2]
        if width != 1280 or height != 720:
            src_image = cv2.resize(src_image, (1280, 720))  

        gImg = cv2.remap(src_image, map_x, map_y,
                         interpolation=cv2.INTER_NEAREST,
                         borderValue=(125, 125, 125))
        imgf = np.float32(gImg) * (1.0/ 255.0)

        # create index for slice id mapping from  ground x anchor, and y anchor
        #   [y anchors,
        #    x anchors]
        #
        class_count = 2
        label = np.zeros((y_anchors, x_anchors, 1), dtype=np.float32)
        
    
        # transform "h_samples" & "lanes" to desired format
        anchor_scale_x = (float)(x_anchors) / (float)(groundSize[1])
        anchor_scale_y = (float)(y_anchors) / (float)(groundSize[0])

        # calculate anchor offsets
        for laneIdx in range(min(len(label_lanes), max_lane_count)):
            lane_data = label_lanes[laneIdx]

            prev_gx = None
            prev_gy = None
            prev_ax = None
            prev_ay = None
            for idx in range(len(lane_data)):
                dy = label_h_samples[idx]
                dx = lane_data[idx]

                if (dx < 0):
                    continue

                # do perspective transform at dx, dy
                gx, gy, gz = np.matmul(H, [[dx], [dy], [1.0]])
                if gz > 0:
                    continue

                # conver to anchor coordinate(grid)
                gx = int(gx / gz)
                gy = int(gy / gz)
                if gx < 0 or gy < 0 or gx >=(groundSize[1]-1) or gy >= (groundSize[0]-1):
                    continue

                ax = int(gx * anchor_scale_x)
                ay = int(gy * anchor_scale_y)

                if ax < 0 or ay < 0 or ax >=(x_anchors-1) or ay >= (y_anchors-1):
                    continue

                label_value = (laneIdx + 1.0) * 50
                label[ay][ax][0] = label_value

                if prev_gx is None:
                    prev_gx = gx
                    prev_gy = gy
                    prev_ax = ax
                    prev_ay = ay
                else:
                    if abs(ay - prev_ay) <= 1:
                        label[ay][ax][0] = label_value
                    else:
                        gA = np.array([prev_gx, prev_gy])
                        gB = np.array([gx, gy])
                        gLen = (float)(np.linalg.norm(gA - gB))
                        gV = (gA - gB) / gLen

                        inter_len = min(max((int)(abs(prev_gy - gy)), 1), 10)
                        for dy in range(inter_len):
                            gC = gB + gV * (float(dy) / float(inter_len)) * gLen

                            ax = np.int32(gC[0] * anchor_scale_x)
                            ay = np.int32(gC[1] * anchor_scale_y)

                            label[ay][ax][0] = label_value

                    prev_gx = gx
                    prev_gy = gy
                    prev_ax = ax
                    prev_ay = ay


        # label /= acc_count
        return (imgf, label)


    # ----------------------------------------------------------------------------------------
    def tttmap_projection_data_generator(src_image,
                                         label_lanes,
                                         label_h_samples, 
                                         net_input_img_size,
                                         x_anchors,
                                         y_anchors,
                                         max_lane_count,
                                         H,
                                         map_x,
                                         map_y,
                                         groundSize):
        # transform image by perspective matrix
        height, width = src_image.shape[:2]
        if width != 1280 or height != 720:
            src_image = cv2.resize(src_image, (1280, 720))  

        gImg = cv2.remap(src_image, map_x, map_y,
                         interpolation=cv2.INTER_NEAREST,
                         borderValue=(125, 125, 125))
        imgf = np.float32(gImg) * (1.0/ 255.0)


        # create label for class
        class_list= {'background'         : [0, 1],
                     'lane_marking'       : [1, 0]}
        class_count = len(class_list)     # [background, road]

        # create label for slice id mapping from  ground x anchor, and y anchor
        #   [y anchors,
        #    x anchors,
        #    class count + x offset]
        #
        class_count = 2
        offset_dim = 1
        instance_label_dim = 1
        label = np.zeros((y_anchors, x_anchors, class_count + offset_dim + instance_label_dim), dtype=np.float32)
        acc_count = np.zeros((y_anchors, x_anchors, class_count + offset_dim), dtype=np.float32)
        class_idx = 0
        x_offset_idx = class_count
        instance_label_idx = class_count + offset_dim

        # init values
        label[:,:,class_idx:class_idx+class_count] = class_list['background']
        label[:,:,x_offset_idx] = 0.0001
        
        # acc_count[:] = 1
        # acc_count.fill(1)
        
     
        # transform "h_samples" & "lanes" to desired format
        anchor_scale_x = (float)(x_anchors) / (float)(groundSize[1])
        anchor_scale_y = (float)(y_anchors) / (float)(groundSize[0])
        inv_anchor_scale_x = (float)(groundSize[1]) / (float)(x_anchors)
        inv_anchor_scale_y = (float)(groundSize[0]) / (float)(y_anchors)
        
        # calculate anchor offsets
        for laneIdx in range(min(len(label_lanes), max_lane_count)):
            lane_data = label_lanes[laneIdx]

            prev_gx = None
            prev_gy = None
            prev_ax = None
            prev_ay = None
            for idx in range(len(lane_data)):
                dy = label_h_samples[idx]
                dx = lane_data[idx]

                if (dx < 0):
                    continue

                # do perspective transform at dx, dy
                gx, gy, gz = np.matmul(H, [[dx], [dy], [1.0]])
                if gz > 0:
                    continue

                # conver to anchor coordinate(grid)
                gx = int(gx / gz)
                gy = int(gy / gz)
                if gx < 0 or gy < 0 or gx >=(groundSize[1]-1) or gy >= (groundSize[0]-1):
                    continue

                ax = int(gx * anchor_scale_x)
                ay = int(gy * anchor_scale_y)

                if ax < 0 or ay < 0 or ax >=(x_anchors-1) or ay >= (y_anchors-1):
                    continue

                instance_label_value = (laneIdx + 1.0) * 50                        
                label[ay][ax][class_idx:class_idx+class_count] = class_list['lane_marking']

                # offset = gx - (ax / anchor_scale_x)
                # label[ay][ax][x_offset_idx] += math.log(offset+0.0001)
                # acc_count [ay][ax][x_offset_idx] += 1
            
     
                if prev_gx is None:
                    prev_gx = gx
                    prev_gy = gy
                    prev_ax = ax
                    prev_ay = ay
                else:
                    if abs(ay - prev_ay) <= 1:
                        if acc_count [ay][ax][x_offset_idx] > 0:
                            continue
                        offset = gx - (ax / anchor_scale_x)
                        label[ay][ax][x_offset_idx] += math.log(offset+0.0001)
                        label[ay][ax][instance_label_idx] = instance_label_value
                        acc_count [ay][ax][x_offset_idx] = 1
                    else:
                        gA = np.array([prev_gx, prev_gy])
                        gB = np.array([gx, gy])
                        gLen = (float)(np.linalg.norm(gA - gB))
                        
                        gV = (gA - gB) / gLen


                        inter_len = min(max((int)(abs(prev_gy - gy)), 1), 10)
                        # print("gA {}, gB {}, gV{}, gLen{}".format(gA, gB, gV, gLen))
                        for dy in range(inter_len):
                            gC = gB + gV * (float(dy) / float(inter_len)) * gLen

                            ax = np.int32(gC[0] * anchor_scale_x)
                            ay = np.int32(gC[1] * anchor_scale_y)

                            # print("gC {}, ay {}, ax {}".format(gC, ay, ax))
                            if acc_count [ay][ax][x_offset_idx] > 0:
                                continue

                            offset = gC[0] - (ax / anchor_scale_x)
                            label[ay][ax][x_offset_idx] += math.log(offset+0.0001)
                            label[ay][ax][class_idx:class_idx+class_count] = class_list['lane_marking']
                            label[ay][ax][instance_label_idx] = instance_label_value
                            acc_count [ay][ax][x_offset_idx] = 1

                        # print("-----------------------------------------")
                    prev_gx = gx
                    prev_gy = gy
                    prev_ax = ax
                    prev_ay = ay

                # if prev_gx is None:
                #     prev_gx = gx
                #     prev_gy = gy
                #     prev_ax = ax
                #     prev_ay = ay
                # else:
                #     inter_len = min(max((int)(abs(prev_gy - gy)), 1), 3)
                #     ref_x = [float(prev_gx), float(gx)]
                #     ref_y = [float(prev_gy), float(gy)]
                #     f = interp1d(ref_y, ref_x)      # use y to predict x
                #     inter_gy_list = np.linspace(prev_gy, gy, num=inter_len, endpoint=True)
                #     inter_gx_list = f(inter_gy_list)

                #     for inter_gx, inter_gy in zip(inter_gx_list, inter_gy_list):
                #         ax = int(inter_gx * anchor_scale_x)
                #         ay = int(inter_gy * anchor_scale_y)

                #         if ax < 0 or ay < 0 or ax >=(x_anchors-1) or ay >= (y_anchors-1):
                #             continue
                    
                #         if acc_count [ay][ax][x_offset_idx] > 0:
                #             continue
                #         acc_count [ay][ax][x_offset_idx] = 1
                #         label[ay][ax][class_idx:class_idx+class_count] = class_list['lane_marking']

                #         offset = inter_gx - (ax / anchor_scale_x)
                #         label[ay][ax][x_offset_idx] += math.log(offset+0.0001)
                        

        # label /= acc_count
        return (imgf, label)



    def map_projection_data_generator(image,
                                      label_lanes,
                                      label_h_samples, 
                                      net_input_img_size,
                                      x_anchors,
                                      y_anchors,
                                      max_lane_count,
                                      refIdx,
                                      H,
                                      map_x,
                                      map_y):
        # get param value
        image = image.numpy()
        label_lanes = label_lanes.numpy()
        label_h_samples = label_h_samples.numpy()
        net_input_img_size = net_input_img_size.numpy()
        x_anchors = x_anchors.numpy()
        y_anchors = y_anchors.numpy()
        max_lane_count = max_lane_count.numpy()
        H = H.numpy()[0]

        # transform image by perspective matrix
        height, width = image.shape[:2]
        if width != 1280 or height != 720:
            image = cv2.resize(image, (1280, 720))  

        gImg = cv2.remap(image,
                         TusimpleLane.map_x[refIdx],
                         TusimpleLane.map_y[refIdx],
                         interpolation=cv2.INTER_NEAREST,
                         borderValue=(125, 125, 125))
        imgf = np.asarray(gImg) * (1.0/ 255.0)


        # transform "h_samples" & "lanes" to desired format
        x_scale = (float)(x_anchors) / (float)(TusimpleLane.groundSize[1])
        y_scale = (float)(y_anchors) / (float)(TusimpleLane.groundSize[0])
        
        # create label for class
        class_list= {'background'         : [0, 1],
                     'lane_marking'       : [1, 0]}
        class_count = len(class_list)     # [background, road]

        # create label for slice id mapping from  ground x anchor, and y anchor
        #   [y anchors,
        #    x anchors,
        #    class count + x offset]
        #
        label = np.zeros((y_anchors, x_anchors, class_count + 1), dtype=np.int8)


        for laneIdx in range(min(len(label_lanes), max_lane_count)):
            lane_data = label_lanes[laneIdx]

            px = -1
            py = -1
            for idx in range(len(lane_data)):
                dy = label_h_samples[idx]
                dx = lane_data[idx]

                if (dx < 0):
                    continue

                # roate dx, dy
                gx, gy, gz = np.matmul(H[refIdx], [[dx], [dy], [1.0]])
                if gz > 0:
                    continue
                gx /= gz
                gy /= gz

                # conver to anchor coordinate
                gx *= x_scale
                gy *= y_scale

                if gx < 0 or gy < 0 or gx >=(x_anchors-2) or gy >= (y_anchors-1):
                    continue

                # if px != -1 and py != -1:
                #     cv2.line(label[laneAnchorIdx], (px, py), (gx, gy), (1))
                #     cv2.line(label[laneAnchorIdx], (x_anchors-1, py), (x_anchors-1, gy), (0))
                px = gx
                py = gy


        # tf.print("label ", tf.convert_to_tensor(label), summarize=-1)
        # sys.exit(0)

        # # check empty column  
        # for laneIdx in range(max_lane_count):
        #     check_sum = np.sum(label[laneIdx], axis=1)

        #     for yIdx in range(len(check_sum)):
        #         if check_sum[yIdx] == 0:
        #             xIdx = x_anchors -1
        #             label[laneIdx][yIdx][xIdx] = 1
                

        # yield (imgf, [label, slice_confidences])
        # return [imgf, label]
        
        return (imgf, label)
 
    # ----------------------------------------------------------------------------------------
    def _data_reader(dataset_path,
                     label_data_name,
                     augmentation_deg):

        label_data_path = os.path.join(dataset_path, label_data_name)
        if (os.path.exists(label_data_path) == False):
            print("Label file doesn't exist, path : ", label_data_path)
            sys.exit(0)

        # load data
        count =0
        
        with open(label_data_path, 'r') as reader:
            for line in reader.readlines():
                raw_label = json.loads(line)
                # print("prefetch_list count ", len(prefetch_list))
                # print(">>>>>", jf["raw_file"])
                # print(">>>>>", jf["lanes"])
                # print(">>>>>", jf["h_samples"])
                image_path = os.path.join(str(dataset_path, "utf-8"), raw_label["raw_file"])
                label_lanes = raw_label["lanes"]
                label_h_samples = raw_label["h_samples"]


                # read image
                with Image.open(image_path) as image:
                    image_ary = np.asarray(image)
                
                if (count >=32):
                    break
                count += 1

                if augmentation_deg is None:
                    yield (image_ary, label_lanes, label_h_samples, 0)
                else:
                    for refIdx in range(len(augmentation_deg)):
                        yield (image_ary, label_lanes, label_h_samples, refIdx)


                

