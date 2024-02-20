
import sys
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import json
import cv2
import datasets
import math
import datasets.cvat_dataset
import tensorflow_datasets as tfds
from datasets import TusimpleLane

# --------------------------------------------------------------------------------------------------------------
def tflite_image_test(tflite_model_quant_file,
                      dataset,
                      with_post_process=False):
    # load model from saved model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_quant_file),
                                      experimental_delegates=[tf.lite.experimental.load_delegate(
                                          "edgetpu.dll")]
                                      )
    interpreter.allocate_tensors()

    # get index of inputs and outputs
    input_index = interpreter.get_input_details()[0]
    output_index_instance = interpreter.get_output_details()[0]
    output_index_offsets = interpreter.get_output_details()[1]
    output_index_anchor_axis = interpreter.get_output_details()[2]
    
    # get part of data from output tensor
    _, _, _, max_instance_count = output_index_instance['shape']
    _, y_anchors, x_anchors, _ = output_index_anchor_axis['shape']
        

    COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), 
              (0, 255, 255), (255, 0, 255), (255, 255, 0) ]

    for k, elem in enumerate(dataset):
        test_img = elem[0]
        test_label = elem[1]

        if input_index['dtype']== np.uint8:
            test_img = np.uint8(test_img * 255)

        # inference
        interpreter.set_tensor(input_index["index"], test_img)
        interpreter.invoke()

        instance = interpreter.get_tensor(output_index_instance["index"])
        offsets = interpreter.get_tensor(output_index_offsets["index"])
        anchor_axis = interpreter.get_tensor(output_index_anchor_axis["index"])
    

        # convert image to gray 
        main_img = cv2.cvtColor(test_img[0], cv2.COLOR_BGR2GRAY)
        main_img = cv2.cvtColor(main_img, cv2.COLOR_GRAY2BGR)
        
        # post processing
        if not with_post_process:
            # rendering
            for instanceIdx in range(max_instance_count):
                for dy in range(y_anchors):
                    for dx in range(x_anchors):
                        instance_prob = instance[0, dy, dx, instanceIdx]
                        offset = offsets[0, dy, dx, 0]
                        gx = anchor_axis[0, dy, dx, 0] + offset
                        gy = anchor_axis[0, dy, dx, 1]
                        
                        if instance_prob > 0.5:
                            cv2.circle(main_img, (int(gx), int(gy)), 2, tuple(int(x) for x in COLORS[instanceIdx]))
        else:
            # Check the variance of anchors by row, ideally, we want each row of instance containt only 
            # zero or one valid anchor to identify instance of lane, but in some case, over one instance 
            # at same row would happened. In this step, we filter anchors at each row by the x variance.
            instance = tf.convert_to_tensor(instance)
            offsets = tf.convert_to_tensor(offsets)
            anchor_axis = tf.convert_to_tensor(anchor_axis)

            anchor_x_axis = anchor_axis[:,:,:,0]
            anchor_y_axis = anchor_axis[:,:,:,1]
            anchor_x_axis = tf.expand_dims(anchor_x_axis, axis=-1)
            anchor_y_axis = tf.expand_dims(anchor_y_axis, axis=-1)

            # create 0/1 mask by instance
            instance = tf.where(instance > 0.5,
                                tf.constant([1.0], tf.float32),
                                tf.constant([0.0], tf.float32))
       
            # mux x anchors and offsets by instance, the reason why y anchors doesn't need to
            # multiplied by instance is that y anchors doesn't join the calcuation of following
            # steps about "variance threshold by row"
            anchor_x_axis = tf.add(anchor_x_axis, offsets)
            anchor_x_axis = tf.multiply(anchor_x_axis, instance)    # [batch, y_anchors, x_anchors, max_instance_count]
            
            # get mean of x axis
            sum_of_instance_row = tf.reduce_sum(instance, axis=2)
            sum_of_x_axis = tf.reduce_sum(anchor_x_axis, axis=2)
            mean_of_x_axis = tf.math.divide_no_nan(sum_of_x_axis, sum_of_instance_row)
            mean_of_x_axis = tf.expand_dims(mean_of_x_axis, axis=2)
            mean_of_x_axis = tf.tile(mean_of_x_axis, [1, 1, x_anchors, 1])
            
            # create mask for threshold
            X_VARIANCE_THRESHOLD = 10.0
            diff_of_axis_x = tf.abs(tf.subtract(anchor_x_axis, mean_of_x_axis))
            mask_of_mean_offset = tf.where(diff_of_axis_x < X_VARIANCE_THRESHOLD,
                                           tf.constant ([1.0], tf.float32),
                                           tf.constant([0.0], tf.float32))
       
            # do threshold
            instance = tf.multiply(mask_of_mean_offset, instance)
            anchor_x_axis = tf.multiply(mask_of_mean_offset, anchor_x_axis)
            anchor_y_axis = tf.multiply(mask_of_mean_offset, anchor_y_axis)

            # average anchors by row
            sum_of_instance_row = tf.reduce_sum(instance, axis=2)
            sum_of_x_axis = tf.reduce_sum(anchor_x_axis, axis=2)
            mean_of_x_axis = tf.math.divide_no_nan(sum_of_x_axis, sum_of_instance_row)

            sum_of_y_axis = tf.reduce_sum(anchor_y_axis, axis=2)
            mean_of_y_axis = tf.math.divide_no_nan(sum_of_y_axis, sum_of_instance_row)
            
            # rendering
            for instanceIdx in range(max_instance_count):
                for dy in range(y_anchors):
                    instance_prob = sum_of_instance_row[0, dy, instanceIdx]
                    gx = mean_of_x_axis[0, dy, instanceIdx]
                    gy = mean_of_y_axis[0, dy, instanceIdx]
                        
                    if instance_prob > 0.5:
                        cv2.circle(main_img, (int(gx), int(gy)), 2, tuple(int(x) for x in COLORS[instanceIdx]))

        # redering output image
        target_szie = (1000, 1000)
        main_img = cv2.resize(main_img, target_szie)

        inv_dx = 1.0 / float(x_anchors)
        inv_dy = 1.0 / float(y_anchors)
        for dy in range(y_anchors):
            for dx in range(x_anchors):
                px = (inv_dx * dx) * target_szie[0]
                py = (inv_dy * dy) * target_szie[1]
                cv2.line(main_img, (int(px), 0), (int(px), target_szie[1]), (125, 125, 125))
                cv2.line(main_img, (0, int(py)), (target_szie[0], int(py)), (125, 125, 125))

        cv2.imwrite(f"images/outpt_imgs/frame{k:03d}.jpg", main_img)
        # plt.figure(figsize = (8,8))
        # plt.imshow(main_img)
        # plt.show()


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
    tflite_model_name   = config["model_info"]["tflite_model_name"]


    if not os.path.exists(tflite_model_name):
        print("tlite model doesn't exist, please run \"generate_tflite_nidel.py\" first to convert tflite model.")
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
    test_dataset_path = "C:/Users/inf21034/source/IMG_ROOTS/1280x960_CVATROOT/test_set"
    test_label_set = ["test_set.json"]

    # valid_batches = datasets.TusimpleLane(test_dataset_path,
    #                                       test_label_set,
    #                                       config,
    #                                       augmentation=False).get_pipe()
    valid_batches = tfds.load('cvat_dataset', split='test', shuffle_files=True, as_supervised=True)
    # TusimpleLane(test_dataset_path, test_label_set, config).get_pipe()
    #tfds.load('cvat_dataset', split='test', shuffle_files=True, as_supervised=True)
    valid_batches = valid_batches.batch(1)

    print("---------------------------------------------------")
    print("Load model as TF-Lite and test")
    print("---------------------------------------------------")
    tflite_image_test(tflite_model_name, valid_batches, True)
   