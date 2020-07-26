
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

# a = tf.constant([-20, -1.0, 0.0, 1.0, 2.0], dtype = tf.float32)
# b = tf.keras.activations.sigmoid(a)
# print(b.numpy())

# a = tf.constant([
#                  [[-20, -1.0, 0.0, 1.0, 2.0],
#                   [-20, -1.0, 0.0, 1.0, 2.0],
#                   [-20, -1.0, 0.0, 1.0, 2.0],
#                   [-20, -1.0, 0.0, 1.0, 2.0]],
#                  [[-20, -1.0, 0.0, 1.0, 2.0],
#                   [-20, -1.0, 0.0, 1.0, 2.0],
#                   [-20, -1.0, 0.0, 1.0, 2.0],
#                   [-20, -1.0, 0.0, 1.0, 2.0]]
#                 ] , dtype = tf.float32)
# b = tf.keras.activations.sigmoid(a)
# print(b.numpy())
# asdasd





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
# def benchmark(dataset, num_epochs=10):
#     start_time = time.perf_counter()
#     for epoch_num in range(num_epochs):
#         print("epoch_num ", epoch_num)
#         count = 0
#         for sample in dataset:
#             count += 1
#             # Performing a training step
#             # print(tf.shape(sample[0]))
#             # print(tf.shape(sample[1]))
#             print("train ............................", count)
#             # time.sleep(0.05)
#     tf.print("Execution time:", time.perf_counter() - start_time)


# ---------------------------------------------------------------------------------------------------
# config tensorflow to prevent out of memory when training
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


net_input_img_size = (512, 288)
x_anchors = 100
y_anchors = 64
max_lane_count = 4
dataset = TusimpleLane("/home/dana/Datasets/ML/TuSimple/train_set", net_input_img_size, x_anchors, y_anchors, max_lane_count)


# titanic_batches = dataset.batch(2).repeat(1)
# titanic_batches = dataset.prefetch(tf.data.experimental.AUTOTUNE).batch(200)
# titanic_batches = dataset.interleave(lambda parameter_list: dataset,
#                                      cycle_length=100,
#                                     block_length=1)
titanic_batches = dataset.batch(1)

for elem in titanic_batches:
    print(tf.shape(elem[0]))
    print(tf.shape(elem[1]))
    # tf.print(elem[1], summarize=-1)
    print("-------------------------------------------------------")
    break
    # asdasd

model = LaneModel(net_input_img_size, x_anchors, y_anchors, max_lane_count)
model.create()
model.load_weight()
model.train(titanic_batches)
model.evaluate(titanic_batches)
