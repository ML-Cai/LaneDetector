import tensorflow as tf
import time
import os
import json
import sys
import itertools
import random
import numpy as np
import tensorflow as tf
from PIL import Image
from cv2 import cv2

class TusimpleLane(tf.data.Dataset):

    groundSize = (256, 256)
    map_x = None
    map_y = None
    H = None

    def create_map(src_img_size, ground_size, augmentation_deg=[0.0]):
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
    def __new__(self, dataset_path, label_set, net_input_img_size, x_anchors, y_anchors, max_lane_count, augmentation=False):
        if (os.path.exists(dataset_path) == False):
            print("File doesn't exist, path : ", dataset_path)
            exit()

        # build map
        if augmentation:
            TusimpleLane.H, TusimpleLane.map_x, TusimpleLane.map_y = self.create_map(src_img_size=(720, 1280),
                                                                                    ground_size=TusimpleLane.groundSize,
                                                                                    augmentation_deg=[0.0, -15.0, 15.0, -30.0, 30.0])
        else:
            TusimpleLane.H, TusimpleLane.map_x, TusimpleLane.map_y = self.create_map(src_img_size=(720, 1280),
                                                                                    ground_size=TusimpleLane.groundSize)

        # build dataset
        label_set = tf.data.Dataset.from_tensor_slices(label_set)
        pipe = label_set.interleave(
            lambda label_file_name: tf.data.Dataset.from_generator(self._data_reader,
                                               output_types=(tf.dtypes.uint8,
                                                             tf.dtypes.int32,
                                                             tf.dtypes.int32,
                                                             tf.dtypes.bool,
                                                             tf.dtypes.int32),
                                               output_shapes=((None),
                                                              (None),
                                                              (None),
                                                              (None),
                                                              (None)),
                                               args=(dataset_path,
                                                     label_file_name,
                                                     net_input_img_size,
                                                     x_anchors,
                                                     y_anchors,
                                                     max_lane_count,
                                                     augmentation)
                                               )
            )
        
        # # non homography
        # pipe = pipe.map (
        #         # convert data to training label and norimalization
        #         lambda image, label_lanes, label_h_samples, augmentation, brightnessValue, saturationsValue, offsetX, offsetY : tf.py_function(func=self.map_data_read,
        #                            inp=[image,
        #                                 label_lanes,
        #                                 label_h_samples,
        #                                 net_input_img_size,
        #                                 x_anchors,
        #                                 y_anchors,
        #                                 max_lane_count,
        #                                 augmentation, brightnessValue, saturationsValue, offsetX, offsetY],
        #                            Tout=(tf.dtypes.float32,
        #                                  tf.dtypes.int32))
        #         ,
        #         num_parallel_calls=tf.data.experimental.AUTOTUNE
        #     )

        pipe = pipe.map (
                # convert data to training label and norimalization
                lambda image, label_lanes, label_h_samples, augmentation, refIdx : 
                    tf.py_function(func=self.map_projection_data_generator,
                                   inp=[image,
                                        label_lanes,
                                        label_h_samples,
                                        net_input_img_size,
                                        x_anchors,
                                        y_anchors,
                                        max_lane_count,
                                        refIdx],
                                   Tout=(tf.dtypes.float32,
                                         tf.dtypes.int8))
                ,
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )

        pipe = pipe.prefetch(  # Overlap producer and consumer works
                tf.data.experimental.AUTOTUNE
            )
                
        return pipe

    # ----------------------------------------------------------------------------------------
    def map_projection_data_generator(image, label_lanes, label_h_samples, 
                                      net_input_img_size, x_anchors, y_anchors, max_lane_count,
                                      refIdx):
        # get param value
        image = image.numpy()
        label_lanes = label_lanes.numpy()
        label_h_samples = label_h_samples.numpy()
        net_input_img_size = net_input_img_size.numpy()
        x_anchors = x_anchors.numpy()
        y_anchors = y_anchors.numpy()
        max_lane_count = max_lane_count.numpy()

        
        height, width = image.shape[:2]
        if width != 1280 or height != 720:
            image = cv2.resize(image, (1280, 720))  

        gImg = cv2.remap(image, TusimpleLane.map_x[refIdx], TusimpleLane.map_y[refIdx], interpolation=cv2.INTER_NEAREST, borderValue=(125, 125, 125))
        imgf = np.asarray(gImg) * (1.0/ 255.0)


        # transform "h_samples" & "lanes" to desired format
        label = np.zeros((max_lane_count, y_anchors, x_anchors), dtype=np.int8)
        x_scale = (float)(x_anchors) / (float)(TusimpleLane.groundSize[1])
        y_scale = (float)(y_anchors) / (float)(TusimpleLane.groundSize[0])
        lane_useage = np.zeros(shape=(max_lane_count), dtype=np.int32)

        for laneIdx in range(min(len(label_lanes), max_lane_count)):
            lane_data = label_lanes[laneIdx]
            
            ####################################################
            pA = None
            pB = None
            count = 0
            for idx in range(len(lane_data) -1, 0, -1):
                dy = label_h_samples[idx]
                dx = lane_data[idx]

                if dx == -2:
                    continue

                gx, gy, gz = np.matmul(TusimpleLane.H[refIdx], [[dx], [dy], [1.0]])
                if gz >= 0:
                    continue
                
                gx /= gz
                gy /= gz
                if gx < 0 or gy < 0 or gx>= TusimpleLane.groundSize[1] or gy >= TusimpleLane.groundSize[0]:
                    continue
                
                if pA is None:
                    pA = [gx, gy]
                else:
                    pB = [gx, gy]
                count += 1
                if (count > 5):
                    break
            
            if pA is None or pB is None:
                continue

            pA = np.array(pA)
            pB = np.array(pB)
            pV = (pA - pB) / (float)(np.linalg.norm(pA - pB))
            pC = pB + pV * (TusimpleLane.groundSize[0] - pB[1]) * (1.0 /pV[1])

            if pC[0] <0:
                laneAnchorIdx = 0
            elif pC[0] >= TusimpleLane.groundSize[1]:
                laneAnchorIdx = max_lane_count -1
            else:
                laneAnchorIdx = 1 + int((pC[0] * x_scale * (max_lane_count -2) ) / x_anchors)

            
            # discard lanes out of ground.
            if laneAnchorIdx < 0 or laneAnchorIdx >= max_lane_count:
                continue

            ####################################################
            lane_useage[laneAnchorIdx] += 1
            if lane_useage[laneAnchorIdx] > 1:
                if laneAnchorIdx == 0: # use new
                    label[laneAnchorIdx,:,:] = 0
                else:  # discard
                    continue

                    
                #tf.print("data alearday used ", lane_useage, " ------> ", len(label_lanes))
                # sys.exit(0)
                
            px = -1
            py = -1
            for idx in range(len(lane_data)):
                dy = label_h_samples[idx]
                dx = lane_data[idx]

                if (dx < 0):
                    continue

                # roate dx, dy
                gx, gy, gz = np.matmul(TusimpleLane.H[refIdx], [[dx], [dy], [1.0]])
                if gz > 0:
                    continue
                gx /= gz
                gy /= gz

                # conver to anchor coordinate
                gx *= x_scale
                gy *= y_scale

                if px != -1 and py != -1:
                    cv2.line(label[laneAnchorIdx], (px, py), (gx, gy), (1))
                px = gx
                py = gy

        # check empty column  
        for laneIdx in range(max_lane_count):
            check_sum = np.sum(label[laneIdx], axis=1)

            for yIdx in range(len(check_sum)):
                if check_sum[yIdx] == 0:
                    xIdx = x_anchors -1
                    label[laneIdx][yIdx][xIdx] = 1
                

        return [imgf, label]


        # # get param value
        # image = image.numpy()
        # label_lanes = label_lanes.numpy()
        # label_h_samples = label_h_samples.numpy()
        # net_input_img_size = net_input_img_size.numpy()
        # x_anchors = x_anchors.numpy()
        # y_anchors = y_anchors.numpy()
        # max_lane_count = max_lane_count.numpy()

        
        # height, width = image.shape[:2]
        # if width != 1280 or height != 720:
        #     image = cv2.resize(image, (1280, 720))  
        # # resized_img = cv2.resize(image, tuple(net_input_img_size))
        # gImg = cv2.remap(image, TusimpleLane.map_x[refIdx], TusimpleLane.map_y[refIdx], interpolation=cv2.INTER_NEAREST)
        # imgf = np.asarray(gImg) * (1.0/ 255.0)


        # # transform "h_samples" & "lanes" to desired format
        # label = np.zeros((max_lane_count, y_anchors, x_anchors), dtype=np.int8)
        # x_scale = (float)(x_anchors) / (float)(TusimpleLane.groundSize[1])
        # y_scale = (float)(y_anchors) / (float)(TusimpleLane.groundSize[0])
        # lane_useage = np.zeros(shape=(max_lane_count), dtype=np.int32)

        # for laneIdx in range(min(len(label_lanes), max_lane_count)):
        #     lane_data = label_lanes[laneIdx]
            
        #     ####################################################333
        #     laneAnchorIdx = 0
        #     for idx in range(len(lane_data)):
        #         dy = label_h_samples[idx]
        #         dx = lane_data[idx]

        #         if dx == -2:
        #             continue

        #         gx, gy, gz = np.matmul(TusimpleLane.H[refIdx], [[dx], [dy], [1.0]])
        #         if gz > 0:
        #             continue
                
        #         # conver to anchor coordinate
        #         gx *= x_scale

        #         laneAnchorIdx = gx / gz

        #     laneAnchorIdx = int((laneAnchorIdx * max_lane_count) / x_anchors)

        #     # discard lanes out of ground.
        #     # if (laneAnchorIdx < 0):
        #     #     laneAnchorIdx = 0
        #     # if (laneAnchorIdx >= max_lane_count):
        #     #     laneAnchorIdx = max_lane_count -1
        #     if laneAnchorIdx < 0 or laneAnchorIdx >= max_lane_count:
        #         continue
        #     ####################################################
        #     lane_useage[laneAnchorIdx] += 1

        #     # tf.print("lane_useage ", lane_useage)
        #     # if lane_useage[laneAnchorIdx] > 0:
        #     #     print("data alearday used")
        #     #     sys.exit(0)

        #     px = -1
        #     py = -1
        #     for idx in range(len(lane_data)):
        #         dy = label_h_samples[idx]
        #         dx = lane_data[idx]

        #         if (dx < 0):
        #             continue

        #         # roate dx, dy
        #         gx, gy, gz = np.matmul(TusimpleLane.H[refIdx], [[dx], [dy], [1.0]])
        #         if gz > 0:
        #             continue
        #         gx /= gz
        #         gy /= gz

        #         # conver to anchor coordinate
        #         gx *= x_scale
        #         gy *= y_scale

        #         if px != -1 and py != -1:
        #             cv2.line(label[laneAnchorIdx], (px, py), (gx, gy), (1))
        #         px = gx
        #         py = gy

        # # check empty column  
        # for laneIdx in range(max_lane_count):
        #     check_sum = np.sum(label[laneIdx], axis=1)

        #     for yIdx in range(len(check_sum)):
        #         if check_sum[yIdx] == 0:
        #             xIdx = x_anchors -1
        #             label[laneIdx][yIdx][xIdx] = 1
                

        # return [imgf, label]

    # ----------------------------------------------------------------------------------------
    def map_data_read(image, label_lanes, label_h_samples, net_input_img_size, x_anchors, y_anchors, max_lane_count,
                      augmentation, brightnessValue, saturationsValue, offsetX, offsetY ):
        # get param value
        label_lanes = label_lanes.numpy()
        label_h_samples = label_h_samples.numpy()
        net_input_img_size = net_input_img_size.numpy()
        x_anchors = x_anchors.numpy()
        y_anchors = y_anchors.numpy()
        max_lane_count = max_lane_count.numpy()

        
        if augmentation:
            # image = tf.image.adjust_brightness (image, brightnessValue)
            # image = tf.image.adjust_saturation (image, saturationsValue)
            coord_transform_matrix = np.array([[1.0, 0.0, offsetX], 
                                               [0.0, 1.0, offsetY]])
            image = image.numpy()
            height, width = image.shape[:2]
            image = cv2.warpAffine(image, coord_transform_matrix, (width, height))
        else:
            coord_transform_matrix = np.array([[1.0, 0.0, 0.0], 
                                               [0.0, 1.0, 0.0]])

            image = image.numpy()
            height, width = image.shape[:2]

        inv_w = 1.0 / float(width)
        inv_h = 1.0 / float(height)

        if augmentation:
            rotate_matrix = np.array([[1.0, 0.0, random.randrange(-200, 200) * 1.0], 
                                      [0.0, 1.0, 0.0]])
            image = cv2.warpAffine(image, rotate_matrix, (width, height) )
        else:
            rotate_matrix = np.array([[1.0, 0.0, 0.0], 
                                      [0.0, 1.0, 0.0]])

        resized_img = cv2.resize(image, tuple(net_input_img_size))
        ary = np.asarray(resized_img)
        imgf = ary * (1.0/ 255.0)

        # transform "h_samples" & "lanes" to desired format
        label = np.zeros((max_lane_count, y_anchors, x_anchors), dtype=np.int8)

        # for laneIdx in range(min(len(label_lanes), max_lane_count)):
        for laneIdx in range(len(label_lanes)):
            lane_data = label_lanes[laneIdx]

            

            for idx in range(len(lane_data)):
                dy = label_h_samples[idx]
                dx = lane_data[idx]

                # # roate dx, dy
                if augmentation:
                    pp = np.array([[dx], [dy], [1]])
                    pp = np.dot(rotate_matrix, pp)
                    dx, dy = pp
                    dx = dx[0]
                    dy = dy[0]
                

                # resample
                if (dx != -2):
                    xIdx = int((dx * inv_w) * (x_anchors - 2))
                else:
                    continue

                if (dy * inv_h) < 0.4:
                    continue
                
                yIdx = int((dy * inv_h) * y_anchors)

                # check laneIdx
                if (yIdx >= 0 and yIdx < y_anchors and xIdx >=0 and xIdx < x_anchors):
                    label[laneIdx][yIdx][xIdx] = 1
                    

        # check empty column  
        for laneIdx in range(max_lane_count):
            check_sum = np.sum(label[laneIdx], axis=1)

            for yIdx in range(len(check_sum)):
                if check_sum[yIdx] == 0:
                    xIdx = x_anchors -1
                    label[laneIdx][yIdx][xIdx] = 1
                

        return [imgf, label]

    # ----------------------------------------------------------------------------------------
    def _data_reader(dataset_path, label_data_name, net_input_img_size, x_anchors, y_anchors, max_lane_count, augmentation):
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

                for refIdx in range(len(TusimpleLane.H)):
                    yield (image_ary, label_lanes, label_h_samples, augmentation, refIdx)


                

