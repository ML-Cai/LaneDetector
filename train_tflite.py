
import sys
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import json
from cv2 import cv2
import datasets
from models import AlphaLaneModel
from losses import LaneLoss
   

# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # read configs
    with open('config.json', 'r') as inf:
        config = json.load(inf)
    
    net_input_img_size  = config["model_info"]["input_image_size"]
    x_anchors           = config["model_info"]["x_anchors"]
    y_anchors           = config["model_info"]["y_anchors"]
    max_lane_count      = config["model_info"]["max_lane_count"]
    checkpoint_path     = config["model_info"]["checkpoint_path"]
    

    # enable memory growth to prevent out of memory when training
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


    # set path of training data
    train_dataset_path = "/home/dana/Datasets/ML/TuSimple/train_set"
    train_label_set = ["label_data_0313.json",
                       "label_data_0531.json",
                       "label_data_0601.json"]
    test_dataset_path = "/home/dana/Datasets/ML/TuSimple/test_set"
    test_label_set = ["test_label.json"]

    # create dataset
    augmentation = True
    batch_size=32
    train_batches = datasets.TusimpleLane(train_dataset_path, train_label_set, config, augmentation=augmentation)
    train_batches = train_batches.shuffle(1000).batch(batch_size)

    valid_batches = datasets.TusimpleLane(test_dataset_path, test_label_set, config, augmentation=False)
    valid_batches = valid_batches.batch(1)


    # create model
    model = AlphaLaneModel(net_input_img_size, x_anchors, y_anchors,
                           training=True,
                           name='AlphaLaneNet',
                           input_batch_size=batch_size)
    model.summary()

    # Enable to load weights from previous training.
    # model.load_weights(tf.train.latest_checkpoint(checkpoint_path))     # load p/retrained

    # set path of checkpoint
    model.compile(optimizer=tf.keras.optimizers.Nadam(lr=0.001),
                  loss=[LaneLoss()])

    checkpoint_path = os.path.join(checkpoint_path, "ccp-{epoch:04d}.ckpt")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
                                                     verbose=1, 
                                                     save_weights_only=True,
                                                     period=5)
    # start train
    history = model.fit(train_batches,
                        # callbacks=[cp_callback],
                        epochs = 200)

    print("Training finish ...")