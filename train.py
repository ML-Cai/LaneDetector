
import sys
import time
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import numpy as np
import datetime
import cv2
from datasets import TusimpleLane
from models import AlphaLaneModel
from losses import LaneLoss


# y_true = [[10.0, 10.0, 10.0, 10.0],
#           [10.0, 0.1, 0.1, 0.1],
#           [10.0, 0.1, 0.1, 0.1],
#           [10.0, 0.1, 0.1, 0.1]]
# Using 'auto'/'sum_over_batch_size' reduction type.

# foo = tf.constant(y_true, dtype = tf.float32)
# tf.print(tf.shape(foo))
# print("-1-----------------------------------------------")
# print(tf.keras.layers.Softmax()(foo).numpy())
# print("0-----------------------------------------------")
# print(tf.keras.layers.Softmax(axis=0)(foo).numpy())
# print("1-----------------------------------------------")
# print(tf.keras.layers.Softmax(axis=1)(foo).numpy())
# sys.exit(0)

# cce = LaneLoss
# m.update_state(y_true, y_pred)
# print("crossentropy ", cce(y_true, y_pred).numpy())
# print("accuracy ", m.result().numpy())
# asd

# --------------------------------------------------------------------------------------------------------------
def train(model, train_dataset, valid_batches, checkpoint_path, train_epochs=200):
    # ---------------------------
    # recover point
    checkpoint_path = os.path.join(checkpoint_path, "ccp-{epoch:04d}.ckpt")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
                                                    verbose=1, 
                                                    save_weights_only=True,
                                                    period=2)
    log_dir = "/home/dana/tmp/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1,
                                                        #   profile_batch = '100,110'
                                                          )
    #----------------------
    # start train
    history = model.fit(train_dataset,
                        callbacks=[cp_callback, tensorboard_callback],
                        epochs = train_epochs,
                        validation_data=valid_batches,
                        validation_freq=10)
    
    return history

# -------------------------------------------------------------------
def evaluate(model, dataset):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
            loss=LaneLoss(),
            metrics=[tf.keras.metrics.CategoricalAccuracy()])

    print("-------------------------------------------------------------------")
    print("evaluate_model")
    test_loss, test_acc = model.evaluate(dataset, verbose=2)
    print('evaluate loss     :', test_loss)
    print('evaluate accuracy :', test_acc)

# --------------------------------------------------------------------------------------------------------------
def image_test(model, dataset, net_input_img_size, x_anchors, y_anchors, max_lane_count):
    print("-------------------------------------------------------------------")
    print("image_test")

    idx = 0
    W = 512
    H = 288
    for elem in dataset:
        # test_loss, test_acc =  model.test_on_batch (x=elem[0], y=elem[1])
        # print('test image [%d] loss: %s, accuracy %s', idx, test_loss, test_acc)
        print('test image', idx)
        prediction = model.predict(x=elem[0])
        main_img = np.uint8(elem[0] * 255)
        main_img = cv2.cvtColor(main_img[0], cv2.COLOR_BGR2GRAY)
        main_img = cv2.cvtColor(main_img, cv2.COLOR_GRAY2BGR)
        
        zeros = np.zeros((y_anchors,  x_anchors -1), dtype=np.uint8)
        raw_output= []
        mask = None
        for si in range(max_lane_count):
            # pred = prediction[0][si]
            pred = prediction[0,si,:, 0:127]
            output_int8 = np.uint8(pred * 255)
            if si == 0:
                img = cv2.merge([zeros, zeros, output_int8])
            elif si ==1:
                img = cv2.merge([zeros, output_int8, zeros])
            elif si ==2:
                img = cv2.merge([output_int8, zeros, zeros])
            elif si ==3:
                img = cv2.merge([zeros, output_int8, output_int8])
            else:
                img = cv2.merge([output_int8, zeros, output_int8])
            img = cv2.resize(img, (W, H))
            valu, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            if mask is None:
                mask = img
            else:
                mask = mask + img
        # main_img = main_img + mask
        # main_img = mask
        main_img = cv2.bitwise_or (main_img, mask)



        prefix = 'build/' + str(idx)
        main_img = cv2.resize(main_img, (1280, 720))

        inv_dx = 1.0 / float(x_anchors)
        inv_dy = 1.0 / float(y_anchors)
        for dy in range(y_anchors):
            for dx in range(x_anchors):
                px = (inv_dx * dx) * 1280
                py = (inv_dy * dy) * 720
                cv2.line(main_img, (int(px), 0), (int(px), 720), (125, 125, 125))
                cv2.line(main_img, (0, int(py)), (1280, int(py)), (125, 125, 125))
        

        # cv2.imwrite(prefix + "_aaw.png", main_img)
        cv2.imshow("preview", main_img)
        key = cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # idx += 1
        # if (idx >=100):
        #     break

# --------------------------------------------------------------------------------------------------------------
def time_test(x_in, model):
    loop_count = 200

    start = time.time()
    for i in range(loop_count):
        model.predict(x_in)
    end = time.time()
    time_predict = (end - start) / float(loop_count)
    
    start = time.time()
    for i in range(loop_count):
        model.predict_on_batch(x_in)
    end = time.time()
    time_predict_on_batch = (end - start) / float(loop_count)
    
    start = time.time()
    for i in range(loop_count):
        model(x_in, training=False)
    end = time.time()
    time_inference = (end - start) / float(loop_count)

    print("predict          avg time", time_predict, "s")
    print("predict_on_batch avg time", time_predict_on_batch, "s")
    print("inference        avg time", time_inference, "s")

# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':
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

    # titanic_batches = dataset.repeat(2).batch(8).shuffle(100)
    train_batches = dataset.batch(16).shuffle(60)
    # train_batches = dataset.batch(1)
    valid_batches = dataset.batch(1)

    for elem in train_batches:
        print(tf.shape(elem[0]))
        print(tf.shape(elem[1]))
        # tf.print(elem[1], summarize=-1)
        break
    
    flag_training=False
    checkpoint_path = "/home/dana/tmp"

    model = AlphaLaneModel(net_input_img_size, x_anchors, y_anchors, max_lane_count)
    model.summary()
    model.load_weights(tf.train.latest_checkpoint(checkpoint_path))
    model.save('model_result', save_format='tf')
    
    if flag_training:
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                loss=LaneLoss(),
                metrics=[tf.keras.metrics.CategoricalAccuracy()])

        train(model, train_batches, valid_batches, checkpoint_path=checkpoint_path, train_epochs=400)

    image_test(model, valid_batches, net_input_img_size, x_anchors, y_anchors, max_lane_count)

    print("mobilenetV2 time:")
    mobilenetV2 = tf.keras.applications.MobileNetV2()
    time_test(np.zeros((1, 224, 224, 3), dtype=np.int8), mobilenetV2)

    print("AlphaLaneNet time:")
    time_test(np.zeros((1, net_input_img_size[1], net_input_img_size[0], 3), dtype=np.int8), model)


