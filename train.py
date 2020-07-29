
import sys
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import numpy as np
from datasets import TusimpleLane
from models import LaneModel

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

y_pred = np.array ([[
          [
           [0.0, 0.0, 0.0, 1.0],
           [0.0, 0.0, 1.0, 0.0],
           [0.0, 1.0, 0.0, 0.0],
           [1.0, 0.0, 0.0, 0.0]
          ],
          [
           [1.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 1.0, 0.0],
           [0.0, 1.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 1.0]
          ],
        ] ])
        
print("y_pred.shape ", y_pred.shape)



# batch, lane_count, y_anchors, x_anchors = y_pred.shape
# loss_all = []
# for i in range(0, y_anchors-1):
#     print(y_pred[:,:,i,:])
#     print(" ")
#     print(y_pred[:,:,i+1,:])

#     HUBER_DELTA = 0.5
#     x  = tf.keras.backend.l2_normalize(y_pred[:,:,i,:])
#     tf.print(y_pred[:,:,i,:] + y_pred[:,:,i+1,:])
#     print("-------------------")
#     loss_all.append(y_pred[:,:,i,:] - y_pred[:,:,i+1,:])

# sys.exit(0)




# x_in = np.zeros((1, 224, 224, 3), dtype=np.int8)
# y_out = np.zeros((1, 1000), dtype=np.int8)
# model = tf.keras.applications.MobileNetV2()
# model.summary()

# result = model.predict(x_in)
# start = time.time()
# for i in range(1000):
#     result = model.predict(x_in)
# end = time.time()
# print("round ", i , " , cost : ", (end - start) / 1000.0, "s")
# sadasd



# y_true = [[[0, 0, 1, 0],
#            [0, 1, 0, 0]],
#           [[0, 1, 0, 0],
#            [0, 1, 0, 0]]]
# y_pred = [[[0.05, 0.95, 0, 0.01],
#            [0.1, 0.8, 0.1, 0.1]],
#           [[0.05, 0.95, 0, 0.01],
#            [0.1, 0.8, 0.1, 0.1]]]
# # Using 'auto'/'sum_over_batch_size' reduction type.
# cce = tf.keras.losses.CategoricalCrossentropy()

# m = tf.keras.metrics.CategoricalAccuracy()
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
# titanic_batches = dataset.batch(4).shuffle(100)
titanic_batches = dataset.batch(1)

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
# model.train(titanic_batches)
model.evaluate(titanic_batches)


# 