
import sys
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import numpy as np
from datasets import TusimpleLane
from models import LaneModel
from losses import LaneLoss

y_true = [[10.0, 10.0, 10.0, 10.0],
          [10.0, 0.1, 0.1, 0.1],
          [10.0, 0.1, 0.1, 0.1],
          [10.0, 0.1, 0.1, 0.1]]
# Using 'auto'/'sum_over_batch_size' reduction type.

# foo = tf.constant(y_true, dtype = tf.float32)
# tf.print(tf.shape(foo))
# print("-1-----------------------------------------------")
# print(tf.keras.layers.Softmax()(foo).numpy())
# print("0-----------------------------------------------")
# print(tf.keras.layers.Softmax(axis=0)(foo).numpy())
# print("1-----------------------------------------------")
# print(tf.keras.layers.Softmax(axis=1)(foo).numpy())
# sys.exit(0)

# cce = LaneLoss
# m.update_state(y_true, y_pred)
# print("crossentropy ", cce(y_true, y_pred).numpy())
# print("accuracy ", m.result().numpy())
# asd


# ---------------------------------------------------------------------------------------------------
# config tensorflow to prevent out of memory when training
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


net_input_img_size = (512, 288)
x_anchors = 128
y_anchors = 72
max_lane_count = 4
train_dataset_path = "/home/dana/Datasets/ML/TuSimple/train_set"
train_label_set = ["label_data_0313.json",
                   "label_data_0531.json",
                   "label_data_0601.json"]
test_dataset_path = "/home/dana/Datasets/ML/TuSimple/test_set"
test_label_set = ["test_label.json"]

full_dataset_path = "/home/dana/Datasets/ML/TuSimple/full_set"
full_label_set = ["label_data_0313.json",
                  "label_data_0531.json",
                  "label_data_0601.json",
                  "test_label.json"]
another_dataset_path = "/home/dana/Datasets/ML/TuSimple/another_test"
another_label_set = ["test.json"]

# dataset = TusimpleLane(train_dataset_path, train_label_set, net_input_img_size, x_anchors, y_anchors, max_lane_count)
# dataset = TusimpleLane(test_dataset_path, test_label_set, net_input_img_size, x_anchors, y_anchors, max_lane_count)
dataset = TusimpleLane(full_dataset_path, full_label_set, net_input_img_size, x_anchors, y_anchors, max_lane_count)
# dataset = TusimpleLane(another_dataset_path, another_label_set, net_input_img_size, x_anchors, y_anchors, max_lane_count)

# titanic_batches = dataset.batch(2).repeat(1)
# titanic_batches = dataset.prefetch(tf.data.experimental.AUTOTUNE).batch(200)
# titanic_batches = dataset.interleave(lambda parameter_list: dataset,
#                                      cycle_length=100,
#                                     block_length=1)
titanic_batches = dataset.batch(4).shuffle(100)
# titanic_batches = dataset.batch(1)

ii = 0
for elem in titanic_batches:
    ii+=1
    print(tf.shape(elem[0]))
    print(tf.shape(elem[1]))
    # tf.print(elem[1], summarize=-1)
    # print("-------------------------------------------------------")
    break
  
model = LaneModel(net_input_img_size, x_anchors, y_anchors, max_lane_count)
model.create()
model.load_weight()
# model.save() 
model.train(titanic_batches)
model.evaluate(titanic_batches)
