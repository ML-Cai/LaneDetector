import sys
import os
import numpy as np
import json
import cv2
import time
# from tflite_runtime.interpreter import Interpreter
# from tflite_runtime.interpreter import load_delegate
import tensorflow as tf


# --------------------------------------------------------------------------------------------------------------
def preprocess_image(image_path, input_size):
    """
    Preprocess the image (resize, normalize, etc.)
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, input_size)
    # Normalize or other preprocessing steps if required
    return image


def tflite_image_test(tflite_model_quant_file, folder_path, with_post_process=True):
    # Load the model onto the Edge TPU
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_quant_file),
                                      experimental_delegates=[tf.lite.experimental.load_delegate(
                                          "edgetpu.dll")])
    # interpreter = Interpreter(model_path=str(tflite_model_quant_file),
    #                           experimental_delegates=[load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()

    # Get index of inputs and outputs# Model input information
    input_details = interpreter.get_input_details()
    input_size = input_details[0]['shape'][1:3]  # Assuming input shape is in the form [1, height, width, 3]

    # Get part of data from output tensor

    COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
              (0, 255, 255), (255, 0, 255), (255, 255, 0)]

    frame_count = 0
    total_inference_time = 0

    for image_name in os.listdir(folder_path):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, image_name)
            image = preprocess_image(image_path, (input_size[1], input_size[0]))  # width, height

            # Prepare input data
            if input_details[0]['dtype'] == np.uint8:
                input_data = np.uint8(image * 255)
            else:
                input_data = image.astype(np.float32) / 255.0

            start_time = time.time()

            # Inference
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], [input_data])
            interpreter.invoke()
            inference_time = time.time() - start_time
            print(f"Inference time: {inference_time} seconds")
            total_inference_time += inference_time
            frame_count += 1

    avg_fps = frame_count / total_inference_time
    print(f"Average FPS: {avg_fps}")
    print(f"Average inference time per frame: {total_inference_time / frame_count} seconds")


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

    if not os.path.exists(tflite_model_name):
        print("tlite model doesn't exist, please run \"generate_tflite_nidel.py\" first to convert tflite model.")
        sys.exit(0)

    # set path of training data
    images = "C:/Users/inf21034/source/IMG_ROOTS/1280x960_CVATROOT/test_set/2023-10-02-12-59-12"
    # "/mnt/c/Users/inf21034/source/IMG_ROOTS/1280x960_CVATROOT/test_set"

    tflite_image_test(tflite_model_name, images)
