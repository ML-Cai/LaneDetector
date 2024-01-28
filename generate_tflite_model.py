import sys
import os
import tensorflow as tf
import json
import cv2
import datasets
from models import AlphaLaneModel
import datasets.cvat_dataset
import tensorflow_datasets as tfds


# --------------------------------------------------------------------------------------------------------------
def representative_data_gen(dataset):
    def _gen():
        for input_value in dataset:
            yield [input_value[0]]

    return _gen


# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # read configs
    with open('add_ins/cvat_config2.json', 'r') as inf:
        config = json.load(inf)

    net_input_img_size = config["model_info"]["input_image_size"]
    x_anchors = config["model_info"]["x_anchors"]
    y_anchors = config["model_info"]["y_anchors"]
    max_lane_count = config["model_info"]["max_lane_count"]
    checkpoint_path = config["model_info"]["checkpoint_path"]
    tflite_model_name = config["model_info"]["tflite_model_name"]

    if not os.path.exists(checkpoint_path):
        print("Checkpoint doesn't exist, please run \"train.py\" first to training model first.")
        sys.exit(0)

    # enable memory growth to prevent out of memory when training
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # set path of training data
    train_dataset_path = "/mnt/c/Users/inf21034/source/IMG_ROOTS/1280x960_CVATROOT/train_set"
    train_label_set = ["train_set.json"]
    """["label_data_0313.json",
                       "label_data_0531.json",
                       "label_data_0601.json"]"""
    test_dataset_path = "/mnt/c/Users/inf21034/source/IMG_ROOTS/1280x960_CVATROOT/test_set"
    test_label_set = ["test_set.json"]
        # ["test_label.json"]

    # create dataset
    train_batches = tfds.load('cvat_dataset', split='train', shuffle_files=True, as_supervised=True)
    # datasets.TusimpleLane(train_dataset_path, train_label_set, config, augmentation=False).get_pipe()
    representative_dataset = train_batches.batch(1)

    valid_batches = tfds.load('cvat_dataset', split='test', shuffle_files=True, as_supervised=True)
    # datasets.TusimpleLane(test_dataset_path, test_label_set, config, augmentation=False).get_pipe()
    valid_batches = valid_batches.batch(1)

    # create model and load weights
    model = AlphaLaneModel(net_input_img_size, x_anchors, y_anchors,
                           training=False,
                           name='AlphaLaneNet',
                           input_batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_path))  # load p/retrained

    # Convert the model.
    # Note : I use whole training dataset for the representative_dataset, 
    #        it will take a long time at convert(~20 minutes).
    print("---------------------------------------------------")
    print("Conver model (TF-Lite)")
    print("---------------------------------------------------")
    tf.lite.TFLiteConverter.from_concrete_functions
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen(representative_dataset)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32
    converter.experimental_new_converter = True
    tflite_model = converter.convert()

    print("---------------------------------------------------")
    print("Save tflite model")
    print("---------------------------------------------------")
    with tf.io.gfile.GFile(tflite_model_name, 'wb') as f:
        f.write(tflite_model)

    # test the conver loadable or not
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_name))
    interpreter.allocate_tensors()

    print("Generate finish, model path : ", tflite_model_name)
