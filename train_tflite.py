
import sys
import time
import os
# import matplotlib.pyplot as plt
import tensorflow as tf
# import tensorflow_datasets.public_api as tfds
import tensorflow_model_optimization as tfmot
import numpy as np
import datetime
import cv2
import datasets
from models import AlphaLaneModel
from losses import LaneLoss
from losses import LaneAccuracy

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
def tflite_evaluate(tflite_model_quant_file, dataset):
    print("-------------------------------------------------------------------")
    print("TF-Lite evaluate")
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]
    output_index = interpreter.get_output_details()[0]
    lane_accuracy = LaneAccuracy()

    acc_count = 0
    acc_acc = 0
    for elem in dataset:
        test_img = elem[0]
        test_label = elem[1]

        if input_index['dtype']== np.uint8:
            test_img = np.uint8(test_img * 255)

        interpreter.set_tensor(input_index["index"], test_img)
        
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_index["index"])
        # prediction = tf.keras.activations.softmax(tf.convert_to_tensor(prediction, dtype=tf.float32)).numpy()
        
        y_pred =prediction
        y_true =test_label

        # tf.print(tf.shape(tf.convert_to_tensor(y_true)))
        # tf.print(tf.shape(tf.convert_to_tensor(y_pred, dtype=tf.float32)))
        acc = lane_accuracy(tf.convert_to_tensor(y_true),
                            tf.convert_to_tensor(y_pred, dtype=tf.float32))
        print("acc ", acc)
        acc_count+=1
        acc_acc += acc

    print('avg acc ', acc_acc / acc_count)

# --------------------------------------------------------------------------------------------------------------
def image_test(model, dataset, net_input_img_size, x_anchors, y_anchors, max_lane_count):
    print("-------------------------------------------------------------------")
    print("image_test")

    idx = 0
    W = net_input_img_size[0]
    H = net_input_img_size[1]
    for elem in dataset:
        # test_loss, test_acc =  model.test_on_batch (x=elem[0], y=elem[1])
        # print('test image [%d] loss: %s, accuracy %s', idx, test_loss, test_acc)
        prediction = model.predict(x=elem[0])
        prediction = tf.keras.activations.softmax( tf.convert_to_tensor(prediction)).numpy()


        main_img = np.uint8(elem[0] * 255)
        main_img = cv2.cvtColor(main_img[0], cv2.COLOR_BGR2GRAY)
        main_img = cv2.cvtColor(main_img, cv2.COLOR_GRAY2BGR)
        
        zeros = np.zeros((y_anchors,  x_anchors -1), dtype=np.uint8)
        raw_output= []
        mask = None
        for si in range(max_lane_count):
            # pred = prediction[0][si]
            pred = prediction[0,si,:, 0:x_anchors-1]

            # for dy in range(y_anchors):
            #     max = np.max(pred[dy:])
            #     pred[dy:]

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
            # valu, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
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
def tflite_image_homography_test(tflite_model_quant_file, dataset, net_input_img_size, x_anchors, y_anchors, max_lane_count):
    print("-------------------------------------------------------------------")
    print("image_test")

    #############################################
    groundSize = (288, 288)
    imgP = [[501, 362], [781, 357],  [924, 499], [337, 510]]
    groundP = [[-180, 1000], [180, 1000],  [180, 300], [-180, 300]]

    for i in range(4):
        imgP[i][0] *= 512.0 / 1280.0
        imgP[i][1] *= 288.0 / 720.0
        groundP[i][0] = groundP[i][0] * 0.2 + groundSize[0] / 2.0
        groundP[i][1] = groundSize[1] - groundP[i][1] * 0.075
    
    
    H, mask = cv2.findHomography(np.float32(imgP), np.float32(groundP))


    for elem in dataset:
        test_img = elem[0]
        test_label = elem[1]
        test_img = np.uint8(test_img * 255)
        test_img = np.reshape(test_img, newshape=(288, 512, 3))
       
        img = cv2.warpPerspective(test_img, H, dsize=groundSize)

        img = cv2.hconcat([test_img, img])
        # cv2.imshow("test_img", test_img)
        cv2.imshow("preview", img)
        key = cv2.waitKey(0)

    #############################################

# --------------------------------------------------------------------------------------------------------------
def tflite_image_test(tflite_model_quant_file, dataset, net_input_img_size, x_anchors, y_anchors, max_lane_count):
    print("-------------------------------------------------------------------")
    print("image_test")

    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
    interpreter.allocate_tensors()

    idx = 0
    W = net_input_img_size[0]
    H = net_input_img_size[1]
    
    input_index = interpreter.get_input_details()[0]
    output_index = interpreter.get_output_details()[0]
    
    tf.print("Input :  ", interpreter.get_input_details())
    tf.print("Output : ", interpreter.get_output_details())
    lane_accuracy = LaneAccuracy()

    for elem in dataset:
        test_img = elem[0]
        test_label = elem[1]

        if input_index['dtype']== np.uint8:
            test_img = np.uint8(test_img * 255)

        # inference
        interpreter.set_tensor(input_index["index"], test_img)
        interpreter.invoke()
        np.set_printoptions(threshold=sys.maxsize)
        prediction = interpreter.get_tensor(output_index["index"])

        # check accuracy
        y_pred = prediction
        y_true = test_label
        acc = lane_accuracy(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred, dtype=tf.float32))

        prediction = tf.keras.activations.softmax(tf.convert_to_tensor(prediction, dtype=tf.float32)).numpy()


        main_img = test_img
        main_img = cv2.cvtColor(main_img[0], cv2.COLOR_BGR2GRAY)
        main_img = cv2.cvtColor(main_img, cv2.COLOR_GRAY2BGR)
        
        zeros = np.zeros((y_anchors,  x_anchors -1), dtype=np.uint8)
        raw_output= []
        mask = None
        for si in range(max_lane_count):
            # pred = prediction[0][si]
            pred = prediction[0,si,:, 0:x_anchors-1]
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

            # tt = cv2.resize(img, (1280, 720))
            # cv2.imwrite("tmp/preview_"+str(si)+".png", tt)

            valu, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
            if mask is None:
                mask = img
            else:
                mask = mask + img

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

        cv2.putText(main_img, 'accuracy '+str(acc), org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 0, 255))
        cv2.imshow("preview", main_img)
        key = cv2.waitKey(0)


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

# --------------------------------------------------------------------------------------------------------------
def representative_data_gen(dataset):
    def _gen():
        for input_value in dataset.take(10):
            # Model has only one input so each data point has one element.
            yield [input_value[0]]
    
    return _gen


# --------------------------------------------------------------------------------------------------------------
def train(model, train_dataset, valid_batches, checkpoint_path, train_epochs=200):
    # ---------------------------
    # recover point
    checkpoint_path = os.path.join(checkpoint_path, "ccp-{epoch:04d}.ckpt")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
                                                    verbose=1, 
                                                    save_weights_only=True,
                                                    period=5)
    log_dir = "tmp/tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1,
                                                          profile_batch = '100,110'
                                                          )
    #----------------------
    # start train
    history = model.fit(train_dataset,
                        callbacks=[cp_callback
                                #    , tensorboard_callback
                                   ],
                        epochs = train_epochs,
                        validation_data=valid_batches,
                        validation_freq=5)
    
    return history


# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # config tensorflow to prevent out of memory when training
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


    net_input_img_size = (512, 288)
    x_anchors = 128
    y_anchors = 72
    # x_anchors = 96
    # y_anchors = 54
    max_lane_count = 4
    train_dataset_path = "/home/dana/Datasets/ML/TuSimple/train_set"
    train_label_set = ["label_data_0313.json", "label_data_0531.json", "label_data_0601.json"]

    test_dataset_path = "/home/dana/Datasets/ML/TuSimple/test_set"
    test_label_set = ["test_label.json"]

    full_dataset_path = "/home/dana/Datasets/ML/TuSimple/full_set"
    full_label_set = ["label_data_0313.json", "label_data_0531.json", "label_data_0601.json", "test_label.json"]

    another_dataset_path = "/home/dana/Datasets/ML/TuSimple/another_test"
    another_label_set = ["test.json"]

    train_batches = datasets.TusimpleLane(full_dataset_path, full_label_set, net_input_img_size, x_anchors, y_anchors, max_lane_count, False)
    # train_batches = datasets.TusimpleLane(train_dataset_path, train_label_set, net_input_img_size, x_anchors, y_anchors, max_lane_count, False)
    train_batches = train_batches.shuffle(500).batch(32)
    # train_batches = train_batches.take(100).batch(1)

    
    valid_batches = datasets.TusimpleLane(test_dataset_path, test_label_set, net_input_img_size, x_anchors, y_anchors, max_lane_count, False)
    valid_batches = datasets.TusimpleLane(train_dataset_path, train_label_set, net_input_img_size, x_anchors, y_anchors, max_lane_count, False)
    # valid_batches = datasets.TusimpleLane(another_dataset_path, another_label_set, net_input_img_size, x_anchors, y_anchors, max_lane_count, False)
    valid_batches = valid_batches.batch(1)



    tflite_model_quant_file = 'model_int8.tflite'
    # tflite_image_test(tflite_model_quant_file, valid_batches, net_input_img_size, x_anchors, y_anchors, max_lane_count)
    tflite_image_homography_test(tflite_model_quant_file, valid_batches, net_input_img_size, x_anchors, y_anchors, max_lane_count)
    # tflite_evaluate(tflite_model_quant_file, valid_batches)
    sys.exit(0)


    # checkpoint_path = "tmp/fake-quantized-checkpoint/"
    checkpoint_path = "tmp/non-QAT-checkpoint/"
    flag_training=True
    flag_training=False

    if flag_training:
        model = AlphaLaneModel(net_input_img_size, x_anchors, y_anchors, max_lane_count, quantization_aware_training=False, input_batch_size=None)
    else:
        model = AlphaLaneModel(net_input_img_size, x_anchors, y_anchors, max_lane_count, quantization_aware_training=False, input_batch_size=1)
    
    model.summary()
    if not flag_training:
        model.load_weights(tf.train.latest_checkpoint(checkpoint_path))     # load pretrained
    model.load_weights(tf.train.latest_checkpoint(checkpoint_path))     # load pretrained
    # tf.keras.utils.plot_model(model, 'model.png')

    if flag_training:
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                      loss=LaneLoss(),
                      metrics=[LaneAccuracy()])

        train(model, train_batches, valid_batches, checkpoint_path=checkpoint_path, train_epochs=100)


    # Convert the model.
    print("---------------------------------------------------")
    print("Conver model (int8)")
    print("---------------------------------------------------")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # ------------------------------------------------
    ## at QAT, with these flags will cause many error.
    converter.representative_dataset = representative_data_gen(valid_batches)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    # ------------------------------------------------
    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    input_type = interpreter.get_input_details()[0]['dtype']
    print('input: ', input_type)
    output_type = interpreter.get_output_details()[0]['dtype']
    print('output: ', output_type)

    # # Save the TF Lite model.
    print("---------------------------------------------------")
    print("Save model")
    print("---------------------------------------------------")
    tflite_model_quant_file = 'model_int8.tflite'
    with tf.io.gfile.GFile(tflite_model_quant_file, 'wb') as f:
        f.write(tflite_model)

    # print("---------------------------------------------------")
    # print("Evulate accuracy")
    # print("---------------------------------------------------")
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
    #                   loss=LaneLoss(),
    #                   metrics=[LaneAccuracy()])
    # test_loss, test_acc = model.evaluate(valid_batches, verbose=2)
    # print('evaluate loss     :', test_loss)
    # print('evaluate accuracy :', test_acc)

    # image_test(model, valid_batches, net_input_img_size, x_anchors, y_anchors, max_lane_count)
    
    print("---------------------------------------------------")
    print("Load model as TF-Lite and test")
    print("---------------------------------------------------")
    # tflite_model_quant_file = '/home/dana/Download/mobilenet_v2_1.0_224_quant.tflite'
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    

    for ten in interpreter.get_tensor_details():
        # tf.print('dtype ', ten['dtype'], '      name ', ten['name'][:100])
        tf.print('dtype ', ten['dtype'], '      name ', ten['name'])
        # tf.print('dtype ', ten['dtype'])
    # tf.print(interpreter.get_input_details())
    # tf.print(interpreter.get_output_details())


    # tflite_image_test(tflite_model_quant_file, valid_batches, net_input_img_size, x_anchors, y_anchors, max_lane_count)
    tflite_evaluate(tflite_model_quant_file, valid_batches)

    # start = time.time()
    # for i in range(100):
    #     interpreter.set_tensor(input_index, test_image)
    #     interpreter.invoke()
    #     predictions = interpreter.get_tensor(output_index)
    # end = time.time()
    # time_inference = (end - start) / float(100)

    # print("predict          avg time", time_inference, "s")

    sys.exit(0)
