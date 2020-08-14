import tensorflow as tf
import time
import os
import json
import sys
import itertools
import random
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image


class TusimpleLane(tf.data.Dataset):
    # ----------------------------------------------------------------------------------------
    def __new__(self, dataset_path, label_set, net_input_img_size, x_anchors, y_anchors, max_lane_count, augmentation=False):
        if (os.path.exists(dataset_path) == False):
            print("File doesn't exist, path : ", dataset_path)
            exit()

        label_set = tf.data.Dataset.from_tensor_slices(label_set)
        pipe = label_set.interleave(
            lambda label_file_name: tf.data.Dataset.from_generator(self._generator,
                                               output_types=(tf.dtypes.uint8,
                                                             tf.dtypes.int32,
                                                             tf.dtypes.int32,
                                                             tf.dtypes.bool,
                                                             tf.dtypes.float32,
                                                             tf.dtypes.float32,
                                                             tf.dtypes.float32,
                                                             tf.dtypes.float32),
                                               output_shapes=((None),
                                                              (None),
                                                              (None),
                                                              (None),
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
        
        pipe = pipe.map (
                # convert data to training label and norimalization
                lambda image, label_lanes, label_h_samples, augmentation, brightnessValue, saturationsValue, offsetX, offsetY : tf.py_function(func=self.map_data_read,
                                   inp=[image,
                                        label_lanes,
                                        label_h_samples,
                                        net_input_img_size,
                                        x_anchors,
                                        y_anchors,
                                        max_lane_count,
                                        augmentation, brightnessValue, saturationsValue, offsetX, offsetY],
                                   Tout=(tf.dtypes.float32,
                                         tf.dtypes.int32))
                ,
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )

        pipe = pipe.prefetch(  # Overlap producer and consumer works
                tf.data.experimental.AUTOTUNE
            )
                
        return pipe

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

        for laneIdx in range(min(len(label_lanes), max_lane_count)):
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
    def _generator(dataset_path, label_data_name, net_input_img_size, x_anchors, y_anchors, max_lane_count, augmentation):
        label_data_path = os.path.join(dataset_path, label_data_name)
        if (os.path.exists(label_data_path) == False):
            print("Label file doesn't exist, path : ", label_data_path)
            exit()

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
                
                if (count >=30):
                    break
                count += 1

                if augmentation:
                    # for brightnessIdx in range(3):
                    #     for saturationsIdx in range(-3, 3, 3):
                    #         for dx in range(-200, 200, 100):
                    #             brightnessValue = brightnessIdx *0.05
                    #             saturationsValue = 1.0 + saturationsIdx * 0.025
                    #             # tf.print("> ", brightnessIdx, " , ", saturationsIdx, " , ", dx, " , ")
                    #             yield (image_ary, label_lanes, label_h_samples, augmentation, brightnessValue, saturationsValue, float(dx), float(0))
                    for dx in range(-200, 200, 100):
                        brightnessValue = 0.0
                        saturationsValue = 1.0 
                        # tf.print("> ", brightnessIdx, " , ", saturationsIdx, " , ", dx, " , ")
                        yield (image_ary, label_lanes, label_h_samples, augmentation, brightnessValue, saturationsValue, float(dx), float(0))
                else:
                    yield (image_ary, label_lanes, label_h_samples, augmentation, 0.0, 1.0, 0.0, 0.0)


                

