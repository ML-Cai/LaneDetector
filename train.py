
import sys
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import numpy as np
from datasets import TusimpleLane
from models import LaneModel

# y_pred =[[[-10, -5, 0.0, 5, 10],
#           [-10, -5, 0.0, 5, 10],
#           [-10, -5, 0.0, 5, 10],
#           [-10, -5, 0.0, 5, 10]],
#          [[-10, -5, 0.0, 5, 10],
#           [-10, -5, 0.0, 5, 10],
#           [-10, -5, 0.0, 5, 10],
#           [-10, -5, 0.0, 5, 10]]]
# y_pred = [[-10, -5, 0.0, 5, 10],
#           [-10, -5, 0.0, 5, 10]]
# print("y_pred ", tf.shape(y_pred))
# a = tf.constant(y_pred, dtype = tf.float32)
# b = tf.keras.activations.softmax(a)
# print(b.numpy())
# # [4.539787e-05 6.692851e-03 5.000000e-01 9.933072e-01 9.999546e-01]
# asdasd


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



dataset = TusimpleLane("/home/dana/Datasets/ML/TuSimple/train_set", (640, 360), 100, 50, 4)


# titanic_batches = dataset.batch(2).repeat(1)
titanic_batches = dataset.prefetch(tf.data.experimental.AUTOTUNE).batch(3)


for elem in titanic_batches:
    print(tf.shape(elem[0]))
    print(tf.shape(elem[1]))
    # tf.print(elem[1], summarize=-1)
    print("-------------------------------------------------------")
    # asdasd

# model = LaneModel()
# model.create()
# model.load_weight()
# # model.train(titanic_batches)
# model.evaluate(titanic_batches)