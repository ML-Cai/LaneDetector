import tensorflow as tf
import time
import os
import json
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
from PIL import Image


class TusimpleLane(tf.data.Dataset):
    def __new__(self, dataset_path, net_input_img_size, x_anchors, y_anchors, max_lane_count):
        if (os.path.exists(dataset_path) == False):
            print("File doesn't exist, path : ", dataset_path)
            exit()
        # train_label_set = ["label_data_0313.json",
        #                    "label_data_0531.json",
        #                    "label_data_0601.json"]
        train_label_set = ["label_data_0313.json"]

        return tf.data.Dataset.from_generator(
            self._generator,
            output_types=(tf.dtypes.float32,
                          tf.dtypes.int8),
            output_shapes=((net_input_img_size[1], net_input_img_size[0], 3), 
                        #    (max_lane_count, y_anchors, x_anchors)),
                           (None)),
            args=(dataset_path, train_label_set, net_input_img_size, x_anchors, y_anchors, max_lane_count)
        )

    def _generator(dataset_path, train_label_set, net_input_img_size, x_anchors, y_anchors, max_lane_count):
        for label_data_name in train_label_set:
            # print("dataset_path ", dataset_path)
            # print("label_data_name ", label_data_name)
            label_data_path = os.path.join(dataset_path, label_data_name)

            if (os.path.exists(label_data_path) == False):
                print("Label file doesn't exist, path : ", label_data_path)

            # load all data (bad)
            
            # print("Prefetch ...")
            time1 = time.time()
            prefetch_list =[]
            with open(label_data_path, 'r') as reader:
                for line in reader.readlines():
                    jf = json.loads(line)
                    # print("prefetch_list count ", len(prefetch_list))
                    # print(">>>>>", jf["raw_file"])
                    # print(">>>>>", jf["lanes"])
                    # print(">>>>>", jf["h_samples"])
                    image_path = os.path.join(dataset_path.decode("utf-8"), jf["raw_file"])
                    # print("image_path ", image_path)
                    
                    with Image.open(image_path) as img:
                        width, height = img.size
                        resized_img = img.resize(net_input_img_size, Image.ANTIALIAS)
                        resized_img.save("aa.png")
                        
                        ary = np.asarray(resized_img)
                        imgf = ary / 255.0
                        resized_img.close()

                    inv_w = 1.0 / float(width)
                    inv_h = 1.0 / float(height)

                    # encode "h_samples" & "lanes" result label
                    # label = np.zeros((max_lane_count, y_anchors, x_anchors), dtype=np.int8)
                    label = np.zeros((max_lane_count, y_anchors, x_anchors), dtype=np.int8)

                    for laneIdx in range(min(len(jf["lanes"]), max_lane_count)):
                        lane_data = jf["lanes"][laneIdx]
                        for idx in range(len(lane_data)):
                            # resample
                            dy = jf["h_samples"][idx]
                            dx = lane_data[idx]
                            if (dx != -2):
                                xIdx = int((dx * inv_w) * (x_anchors - 2))
                            else:
                                continue

                            yIdx = int((dy * inv_h) * y_anchors)
                            label[laneIdx][yIdx][xIdx] = 1
                        
                    for laneIdx in range(max_lane_count):
                        # process
                        check_sum = np.sum(label[laneIdx], axis=1)
                        for yIdx in range(len(check_sum)):
                            if check_sum[yIdx] == 0:
                                xIdx = x_anchors -1
                                label[laneIdx][yIdx][xIdx] = 1

                        # np.set_printoptions(threshold=sys.maxsize)
                        # print(label[laneIdx])
                        

                    prefetch_list.append([imgf, label])
                
                    if (len(prefetch_list) >= 1):
                        break

            time2 = time.time()
            # print("Prefetch finish...", (time2-time1)*1000.0, "ms")
            
            for prefetch in prefetch_list:
                yield (prefetch[0], prefetch[1])

        