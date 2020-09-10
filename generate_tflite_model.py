
import sys
import os
import tensorflow as tf
import json
from cv2 import cv2
import datasets
from models import AlphaLaneModel

# --------------------------------------------------------------------------------------------------------------
def representative_data_gen(dataset):
    def _gen():
        for input_value in dataset:
            yield [input_value[0]]
    
    return _gen
    

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
    tflite_model_name   = config["model_info"]["tflite_model_name"]

    if not os.path.exists(checkpoint_path):
        print("Checkpoint doesn't exist, please run \"train.py\" first to training model first.")
        sys.exit(0)

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
    train_batches = datasets.TusimpleLane(train_dataset_path, train_label_set, config, augmentation=False)
    representative_dataset = train_batches.batch(1)

    valid_batches = datasets.TusimpleLane(test_dataset_path, test_label_set, config, augmentation=False)
    valid_batches = valid_batches.batch(1)

    
    # create model and load weights
    model = AlphaLaneModel(net_input_img_size, x_anchors, y_anchors,
                           training=False,
                           name='AlphaLaneNet',
                           input_batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_path))     # load p/retrained

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
    converter.experimental_new_converter=False
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
