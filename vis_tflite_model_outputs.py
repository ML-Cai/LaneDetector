
import sys
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import json
import cv2
import datasets
import math
from models import AlphaLaneModel


# --------------------------------------------------------------------------------------------------------------
def render_cls_prob(net_input_img_size,
                    x_anchors,
                    y_anchors,
                    output_cls_prob):

    # get prob mask
    prob_vis = np.zeros(shape=(net_input_img_size[1], net_input_img_size[0], 3), dtype=np.uint8)
    anchor_width = net_input_img_size[0] / x_anchors
    anchor_height = net_input_img_size[1] / y_anchors

    for dy in range(y_anchors):
        for dx in range(x_anchors):
            prob = output_cls_prob[0, dy, dx, 0]
            ax = dx * anchor_width
            ay = dy * anchor_height
            
            if prob > 0.5:
                cv2.rectangle(prob_vis,
                              (int(ax), int(ay)),
                              (int(ax+anchor_width), int(ay+anchor_height)),
                              color=(255, 0, 0), thickness=-1)
    return prob_vis

# --------------------------------------------------------------------------------------------------------------
def render_anchor_data(main_img,
                       net_input_img_size,
                       x_anchors,
                       y_anchors,
                       output_cls_prob,
                       output_offsets):
    # get prob mask
    anchor_width = net_input_img_size[0] / x_anchors
    anchor_height = net_input_img_size[1] / y_anchors

    for dy in range(y_anchors):
        for dx in range(x_anchors):
            prob = output_cls_prob[0, dy, dx, 0]
            offset = output_offsets[0, dy, dx, 0]
            ax = dx * anchor_width
            ay = dy * anchor_height
            gx = ax + math.exp(offset)
            gy = ay
            
            if prob > 0.5:
                cv2.line(main_img, (int(ax), int(ay)), (int(gx), int(gy)), color=(255, 0, 0), thickness=1)
                cv2.circle(main_img, (int(gx), int(gy)), radius=3 ,color=(0, 255, 0))

    return main_img

# --------------------------------------------------------------------------------------------------------------
def render_embeddings(main_img,
                      net_input_img_size,
                      x_anchors,
                      y_anchors,
                      output_render_embeddings):

    # get prob mask
    _, _, _, max_instance_count = output_render_embeddings.get_shape().as_list()
    anchor_width = net_input_img_size[0] / x_anchors
    anchor_height = net_input_img_size[1] / y_anchors

    concat_img = None
    for instanceIdx in range(max_instance_count):
        sub_img = main_img.copy()
        for dy in range(y_anchors):
            for dx in range(x_anchors):
                embeddings = output_render_embeddings[0, dy, dx, instanceIdx]
                ax = dx * anchor_width
                ay = dy * anchor_height
                
                if embeddings > 0.5:
                    cv2.rectangle(sub_img,
                              (int(ax), int(ay)),
                              (int(ax+anchor_width), int(ay+anchor_height)),
                              color=(0, 0, 255), thickness=-1)

        # add boundary for recognizability
        cv2.rectangle(sub_img,
                      (0, 0),
                      (net_input_img_size[0], net_input_img_size[0]),
                      color=(255, 0, 0), thickness=5)
        cv2.putText(sub_img, str(instanceIdx), (10, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2.0, color=(255, 255, 0))

        # concat images
        if concat_img is None:
            
            concat_img = sub_img
            
        else:
            concat_img = cv2.hconcat([concat_img, sub_img])

    return concat_img
# --------------------------------------------------------------------------------------------------------------
def output_visualization(config,
                         model,
                         dataset):
    net_input_img_size  = config["model_info"]["input_image_size"]
    x_anchors           = config["model_info"]["x_anchors"]
    y_anchors           = config["model_info"]["y_anchors"]

    # get part of data from output tensor
    # _, _, _, max_instance_count = output_index_embeddings['shape']
    # _, y_anchors, x_anchors, _ = output_index_cls['shape']
        
    for elem in dataset:
        test_img = elem[0]

        prediction = model(test_img)

        # get output
        cls_prob, offsets, embeddings = prediction
        tf.print("shape of cls_prob ", tf.shape(cls_prob))
        tf.print("shape of offsets ", tf.shape(offsets))
        tf.print("shape of embeddings ", tf.shape(embeddings))

        # convert image to gray 
        main_img = np.uint8(test_img[0] * 255)
        main_img = cv2.cvtColor(main_img, cv2.COLOR_BGR2GRAY)
        main_img = cv2.cvtColor(main_img, cv2.COLOR_GRAY2BGR)

        # # get prob mask
        prob_vis =  render_cls_prob(net_input_img_size,
                                    x_anchors,
                                    y_anchors,
                                    cls_prob)
        prob_vis = cv2.addWeighted(main_img, 1.0, prob_vis, 0.5, 1.0)   # blending
        
       
        offset_vis =  render_anchor_data(main_img.copy(),
                                          net_input_img_size,
                                          x_anchors,
                                          y_anchors,
                                          cls_prob,
                                          offsets)

        embedding_vis = render_embeddings(main_img.copy(),
                                          net_input_img_size,
                                          x_anchors,
                                          y_anchors,
                                          embeddings)

        # show images
        # fig, axarr = plt.subplots(3, 1)
        # axarr[0].imshow(prob_vis)
        # axarr[1].imshow(offset_vis)
        # axarr[2].imshow(embedding_vis)
        # plt.show()
        plt.figure(figsize = (8,8))
        plt.imshow(embedding_vis)
        plt.show()
        

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
    test_dataset_path = "/mnt/c/Users/inf21034/source/IMG_ROOTS/1280x960_CVATROOT/test_set"
    test_label_set = ["test_set.json"]

    valid_batches = datasets.TusimpleLane(test_dataset_path,
                                          test_label_set, 
                                          config,
                                          augmentation=False).get_pipe()
    valid_batches = valid_batches.batch(1)

    # create model and load weights
    output_as_raw_data = True
    model = AlphaLaneModel(net_input_img_size, x_anchors, y_anchors,
                           training=False,
                           name='AlphaLaneNet',
                           input_batch_size=1,
                           output_as_raw_data=output_as_raw_data)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_path))     # load p/retrained

    # preview output of model
    output_visualization(config, model, valid_batches)
    