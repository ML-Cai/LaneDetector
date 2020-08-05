import tensorflow as tf
import time
import os
import json
import sys
import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
from PIL import Image


class TusimpleLane(tf.data.Dataset):
    def map_decorator(func, net_input_img_size, x_anchors, y_anchors, max_lane_count):
        def wrapper(image_path, label_lanes, label_h_samples):
            # Use a tf.py_function to prevent auto-graph from compiling the method
            return tf.py_function(
                func,
                inp=(image_path,
                     label_lanes,
                     label_h_samples,
                     net_input_img_size,
                     x_anchors,
                     y_anchors,
                     max_lane_count),
                Tout=(tf.dtypes.string,
                      tf.dtypes.int32,
                      tf.dtypes.int32,
                      tf.dtypes.int32,
                      tf.dtypes.int32,
                      tf.dtypes.int32,
                      tf.dtypes.int32)
            )
        return wrapper
    # ----------------------------------------------------------------------------------------
    def __new__(self, dataset_path, label_set, net_input_img_size, x_anchors, y_anchors, max_lane_count):
        if (os.path.exists(dataset_path) == False):
            print("File doesn't exist, path : ", dataset_path)
            exit()


        label_set = tf.data.Dataset.from_tensor_slices(label_set)
        pipe = label_set.interleave(
            lambda label_file_name: tf.data.Dataset.from_generator(self._generator,
                                               output_types=(tf.dtypes.uint8,
                                                             tf.dtypes.int32,
                                                             tf.dtypes.int32),
                                               output_shapes=((None),
                                                              (None),
                                                              (None)),
                                               args=(dataset_path, label_file_name, net_input_img_size, x_anchors, y_anchors, max_lane_count)
                                               )
            ).map (
                # convert data to training label and norimalization
                lambda image, label_lanes, label_h_samples : tf.py_function(func=self.map_data_read,
                                   inp=[image,
                                        label_lanes,
                                        label_h_samples,
                                        net_input_img_size,
                                        x_anchors,
                                        y_anchors,
                                        max_lane_count],
                                   Tout=(tf.dtypes.float32,
                                         tf.dtypes.int32))
                ,
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            ).prefetch(  # Overlap producer and consumer works
                tf.data.experimental.AUTOTUNE
            )
                
        return pipe

    # ----------------------------------------------------------------------------------------
    def map_data_read(image, label_lanes, label_h_samples, net_input_img_size, x_anchors, y_anchors, max_lane_count):
        # get param value
        label_lanes = label_lanes.numpy()
        label_h_samples = label_h_samples.numpy()
        net_input_img_size = net_input_img_size.numpy()
        x_anchors = x_anchors.numpy()
        y_anchors = y_anchors.numpy()
        max_lane_count = max_lane_count.numpy()


        # augment image
        # image = tf.image.random_brightness(image, max_delta=0.2)
        # image = tf.image.random_saturation(image, 0.8, 1.25)
        image = image.numpy()
        height, width = image.shape[:2]
        inv_w = 1.0 / float(width)
        inv_h = 1.0 / float(height)

        # rotate_matrix = np.array([[1.0, 0.0, random.randrange(-200, 200) * 1.0], 
        #                           [0.0, 1.0, 0.0]])
        # image = cv2.warpAffine(image, rotate_matrix, (width, height) )

        # convert image to float & desired training size
        # resized_img = image.resize(net_input_img_size, Image.ANTIALIAS)
        # tf.print("net_input_img_size ", tuple(net_input_img_size))
        resized_img = cv2.resize(image, tuple(net_input_img_size))
        ary = np.asarray(resized_img)
        imgf = ary * (1.0/ 255.0)
        # resized_img.close()


        # transform "h_samples" & "lanes" to desired format
        label = np.zeros((max_lane_count, y_anchors, x_anchors), dtype=np.int8)

        for laneIdx in range(min(len(label_lanes), max_lane_count)):
            lane_data = label_lanes[laneIdx]
            for idx in range(len(lane_data)):
                dy = label_h_samples[idx]
                dx = lane_data[idx]

                # # roate dx, dy
                # pp = np.array([[dx], [dy], [1]])
                # pp = np.dot(rotate_matrix, pp)
                # dx, dy = pp
                # dx = dx[0]
                # dy = dy[0]
                

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

    # # ----------------------------------------------------------------------------------------
    # def data_transform_map(image_path, label_lanes, label_h_samples, net_input_img_size, x_anchors, y_anchors, max_lane_count):
    #     image_path = image_path.numpy().decode('utf-8')
    #     label_lanes = label_lanes.numpy()
    #     label_h_samples = label_h_samples.numpy()
    #     net_input_img_size = net_input_img_size.numpy()
    #     x_anchors = x_anchors.numpy()
    #     y_anchors = y_anchors.numpy()
    #     max_lane_count = max_lane_count.numpy()

    #     # print("image_path ", image_path)
    #     # print("label_lanes ", label_lanes)
    #     # print("label_h_samples ", label_h_samples)
    #     # print("net_input_img_size ", net_input_img_size.numpy())
    #     # print("x_anchors ", x_anchors.numpy())
    #     # print("y_anchors ", y_anchors.numpy())
    #     # print("max_lane_count ", max_lane_count.numpy())
        
    #     # read image
    #     with Image.open(image_path) as img:
    #         width, height = img.size
    #         inv_w = 1.0 / float(width)
    #         inv_h = 1.0 / float(height)

    #     # convert image to float & desired training size
    #     resized_img = img.resize(net_input_img_size, Image.ANTIALIAS)
    #     ary = np.asarray(resized_img)
    #     imgf = ary * (1.0/ 255.0)
    #     resized_img.close()


    #     # transform "h_samples" & "lanes" to desired format
    #     label = np.zeros((max_lane_count, y_anchors, x_anchors), dtype=np.int8)

    #     for laneIdx in range(min(len(label_lanes), max_lane_count)):
    #         lane_data = label_lanes[laneIdx]
    #         for idx in range(len(lane_data)):
    #             # resample
    #             dy = label_h_samples[idx]
    #             dx = lane_data[idx]

    #             if (dx != -2):
    #                 xIdx = int((dx * inv_w) * (x_anchors - 2))
    #             else:
    #                 continue
    #             yIdx = int((dy * inv_h) * y_anchors)


    #             if (yIdx >= 0 and yIdx < y_anchors and xIdx >=0 and xIdx < x_anchors):
    #                 label[laneIdx][yIdx][xIdx] = 1

    #     # check empty column  
    #     for laneIdx in range(max_lane_count):
    #         check_sum = np.sum(label[laneIdx], axis=1)

    #         for yIdx in range(len(check_sum)):
    #             if check_sum[yIdx] == 0:
    #                 xIdx = x_anchors -1
    #                 label[laneIdx][yIdx][xIdx] = 1
                

    #     return [imgf, label]


    # ----------------------------------------------------------------------------------------
    def _generator(dataset_path, label_data_name, net_input_img_size, x_anchors, y_anchors, max_lane_count):
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
                
                    
                # if (count >=20):
                #     break
                # count += 1

                yield (image_ary, label_lanes, label_h_samples)

