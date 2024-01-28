
import sys
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import json
import cv2
import datasets
import datasets.cvat_dataset
from datasets.cvat_dataset.cvat_dataset_dataset_builder import Builder
import tensorflow_datasets as tfds
from models import AlphaLaneModel
from losses import LaneLoss
   

# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # read configs
    with open('add_ins/cvat_config2.json', 'r') as inf:
        config = json.load(inf)
    
    net_input_img_size  = config["model_info"]["input_image_size"]
    x_anchors           = config["model_info"]["x_anchors"]
    y_anchors           = config["model_info"]["y_anchors"]
    max_lane_count      = config["model_info"]["max_lane_count"]
    checkpoint_path     = config["model_info"]["checkpoint_path"]
    

    # enable memory growth to prevent out of memory when training
    physical_devices = tf.config.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # set path of training data
    train_dataset_path = "/mnt/c/Users/inf21034/source/IMG_ROOTS/1280x960_CVATROOT/train_set"
    # "/mnt/c/Users/inf21034/source/IMG_ROOTS/TUSIMPLEROOT/TUSimple"
    #
    train_label_set = ["train_set.json"]
    """["label_data_0313.json",
                       "label_data_0531.json",
                       "label_data_0601.json"]"""
    test_dataset_path = "/mnt/c/Users/inf21034/source/IMG_ROOTS/1280x960_CVATROOT/test_set"
    # "/mnt/c/Users/inf21034/source/IMG_ROOTS/TUSIMPLEROOT/TUSimple"
    #
    test_label_set = ["test_set.json"]

    # create dataset
    augmentation = True
    batch_size=32
    train_batches = tfds.load('cvat_dataset', split='train', shuffle_files=True, as_supervised=True)
    train_batches = train_batches.prefetch(tf.data.experimental.AUTOTUNE)
        # datasets.TusimpleLane(train_dataset_path, train_label_set, config, augmentation=augmentation).get_pipe()
    train_batches = train_batches.shuffle(1000).batch(batch_size)
    # print("Training batches: ", list(train_batches.as_numpy_iterator()))

    valid_batches = tfds.load('cvat_dataset', split='test', shuffle_files=True, as_supervised=True)
    valid_batches = valid_batches.prefetch(tf.data.experimental.AUTOTUNE)
        # datasets.TusimpleLane(test_dataset_path, test_label_set, config, augmentation=False).get_pipe()
    valid_batches = valid_batches.batch(1)

    # tf.debugging.disable_traceback_filtering()
    # create model
    model: tf.keras.Model = AlphaLaneModel(net_input_img_size, x_anchors, y_anchors,
                           training=True,
                           name='AlphaLaneNet',
                           input_batch_size=batch_size)
    model.summary()

    # Enable to load weights from previous training.
    # model.load_weights(tf.train.latest_checkpoint(checkpoint_path))     # load p/retrained

    # set path of checkpoint
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
                  loss=[LaneLoss()],
                  run_eagerly=False)

    checkpoint_path = os.path.join(checkpoint_path, "ccp-{epoch:04d}.ckpt")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
                                                     verbose=1, 
                                                     save_weights_only=True,
                                                     period=5)
    # start train
    history = model.fit(train_batches,
                        callbacks=[cp_callback],
                        epochs = 200)

    print("Training finish ...")