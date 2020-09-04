
import sys
import time
import os
# import matplotlib.pyplot as plt
import tensorflow as tf
# import tensorflow_datasets.public_api as tfds
import tensorflow_model_optimization as tfmot
import numpy as np
import datetime
from cv2 import cv2
import datasets
import math
from models import AlphaLaneModel
from losses import LaneLoss
from losses import ConfidenceLoss
from losses import LaneAccuracy
from losses import ConfidenceAccuracy
from losses import DiscriminativeLoss

from scipy.interpolate import interp1d

# input_shape = [4, 3, 5, 2]
# batch_size, H, W, channels = input_shape
# xx_channels = tf.range(0.0, float(W), dtype=tf.float32)
# xx_channels = tf.expand_dims(xx_channels, axis=0)
# xx_channels = tf.tile(xx_channels, [H, 1])
# xx_channels = xx_channels / (W - 1.0)
# xx_channels = (xx_channels * 2.0) - 1.0
# xx_channels = tf.expand_dims(xx_channels, axis=0)
# xx_channels = tf.expand_dims(xx_channels, axis=-1)
# xx_channels = tf.tile(xx_channels, [batch_size, 1, 1, 1])
# tf.print("xx_range ", xx_channels, " shape ", tf.shape(xx_channels), xx_channels.dtype)
# sys.exit(0)

# xx_channels = tf.range(0, dim2, dtype=tf.float32)
# xx_channels = tf.expand_dims(xx_channels, axis=0)
# xx_channels = tf.tile(xx_channels, [dim1, 1])
# xx_channels = xx_channels / (dim2 - 1.0)
# xx_channels = (xx_channels * 2.0) - 1.0

# xx_channels = tf.expand_dims(xx_channels, axis=0)
# xx_channels = tf.expand_dims(xx_channels, axis=-1)
# xx_channels = tf.tile(xx_channels, [batch_shape, 1, 1, 1])
# tf.print("xx_range ", xx_channels, " shape ", tf.shape(xx_channels), xx_channels.dtype)


# # yy_channels = tf.range(0, dim1, dtype=tf.float32)
# # yy_channels = tf.expand_dims(yy_channels, axis=1)
# # yy_channels = tf.tile(yy_channels, [1, dim2])
# # yy_channels = yy_channels / (dim1 - 1.0)
# # yy_channels = yy_channels * 2.0 -1.0
# # yy_channels = tf.expand_dims(yy_channels, axis=0)
# # yy_channels = tf.expand_dims(yy_channels, axis=-1)
# # yy_channels = tf.tile(yy_channels, [batch_shape, 1, 1, 1])

# tf.print("yy_channels ", yy_channels, " shape ", tf.shape(yy_channels))

# sys.exit(0)



# y_pred = tf.constant([[[10, 20, 30, 40],
#                        [10, 20, 30, 10],
#                        [10, 20, 30, 40]],
#                       [[10, 20, 30, 40],
#                        [20, 20, 30, 40],
#                        [10, 20, 30, 40]],
#                       [[30, 20, 30, 40],
#                        [20, 20, 30, 40],
#                        [30, 20, 30, 40]]], dtype=np.float32)
# y_true = tf.constant([[[1],
#                        [1],
#                        [1]],
#                       [[2],
#                        [2],
#                        [2]],
#                       [[3],
#                        [3],
#                        [3]]], dtype=np.float32)
# DD = DiscriminativeLoss()
# v = DD(y_true, y_pred)
# print(v)
# sys.exit(0)


# label =  tf.constant([[0, 0, 0],
#                       [1, 1, 1],
#                       [2, 2, 2]])
# label = tf.reshape(label, [3*3])            
# unique_labels, unique_id, counts = tf.unique_with_counts(label)
# num_instances = tf.size(unique_labels)
# print("unique_labels ", unique_labels)
# print("unique_id ", unique_id)
# print("counts ", counts)
# # sys.exit(0)

# src_img = tf.constant([[[10, 20, 30, 40],
#                         [10, 20, 30, 10],
#                         [10, 20, 30, 40]],
#                        [[10, 20, 30, 40],
#                         [20, 20, 30, 40],
#                         [10, 20, 30, 40]],
#                        [[30, 20, 30, 40],
#                         [20, 20, 30, 40],
#                         [30, 20, 30, 40]]])
# print("src_img",src_img.shape)
# src_img = tf.reshape(src_img, [3*3, 4])
# print("src_img",src_img)

# segmented_sum = tf.math.unsorted_segment_sum(src_img, tf.constant(unique_id), num_segments=num_instances)
# print("unsorted_segment_sum ", segmented_sum)

# mu = tf.math.divide(segmented_sum, tf.reshape(counts, (-1, 1)))
# print("mu ", mu)

# mu_expand = tf.gather(mu, unique_id)


# print("mu_expand", mu_expand)

# delta_v = 0.1
# src_img = tf.cast(src_img, tf.float64)
# distance = tf.norm(tf.subtract(mu_expand, src_img), axis=1, ord=1)
# distance = tf.subtract(distance, delta_v)
# distance = tf.clip_by_value(distance, 0., distance)
# distance = tf.square(distance)
# print("distance", distance)

# mu_interleaved_rep = tf.tile(mu, [num_instances, 1])
# print("mu_interleaved_rep", mu_interleaved_rep)

# mu_band_rep = tf.tile(mu, [1, num_instances])
# mu_band_rep = tf.reshape( mu_band_rep, (num_instances * num_instances, 4))
# print("mu_band_rep", mu_band_rep)

# mu_diff = tf.subtract(mu_band_rep, mu_interleaved_rep)
# print("mu_diff", mu_diff)

# intermediate_tensor = tf.reduce_sum(tf.math.abs(mu_diff), axis=1)
# zero_vector = tf.zeros(1, dtype=tf.float64)
# bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
# mu_diff_bool = tf.boolean_mask(mu_diff, bool_mask)
# print("mu_diff_bool", mu_diff_bool)
# sys.exit(0)


# x = [0, 10]
# y = [0, 100]

# f = interp1d(x, y)
# xnew = np.linspace(0, 10, num=10, endpoint=True)

# print(f(xnew))
# sys.exit(0)

# # (1, 10, 32, 64)
# # batch = tf.keras.backend.shape(x)[0]
# batch = 1
# x_anchors = 64
# y_anchors = 32
# cx = np.linspace(1, x_anchors, x_anchors)
# cy = np.linspace(1, y_anchors, y_anchors)
# cx_grid, cy_grid = np.meshgrid(cx, cy)
# cx_grid = np.expand_dims(cx_grid, 0) # This is necessary for np.tile() to do what we want further down
# cy_grid = np.expand_dims(cy_grid, 0) # This is necessary for np.tile() to do what we want further down
# print("cx_grid ", cx_grid)
# print("cy_grid ", cy_grid)
# print("cx_grid ", cx_grid.shape)
# print("cy_grid ", cy_grid.shape)
# # sys.exit(0)

# n_boxes = 10
# boxes_tensor = np.zeros((batch, n_boxes, y_anchors, x_anchors))
# boxes_tensor[:, 0:n_boxes, :, :] = np.tile(cx_grid, (n_boxes, 1, 1)) # Set cx
# # boxes_tensor[:, 1, :, :] = np.tile(cy_grid, (n_boxes, 1, 1)) # Set c
# boxes_tensor = tf.keras.backend.tile(tf.keras.backend.constant(boxes_tensor, dtype='float32'), (batch, 1, 1, 1))
# print("boxes_tensor ", boxes_tensor)
# sys.exit(0)

# cy = np.linspace(1, 10, 10)
# cx = np.linspace(1, 10, 10)
# print("cx ", cx)
# print("cy ", cy)
# cx_grid, cy_grid = np.meshgrid(cx, cy)

# print("cx_grid ", cx_grid)
# print("cx ", cx_grid[..., -8])
# print("cx ", cx_grid[..., -7])
# sys.exit(0)

# print("cy_grid ", cy_grid)
# print("cx_grid ", cx_grid.shape)
# print("cy_grid ", cy_grid.shape)

# cx_grid = np.expand_dims(cx_grid, -1) # This is necessary for np.tile() to do what we want further down
# cy_grid = np.expand_dims(cy_grid, -1) # This is necessary for np.tile() to do what we want further down
# print("cx_grid ", cx_grid)
# print("cy_grid ", cy_grid)
# print("cx_grid ", cx_grid.shape)
# print("cy_grid ", cy_grid.shape)

# feature_map_height = 10
# feature_map_width = 10
# n_boxes = 4
# boxes_tensor = np.zeros((feature_map_height, feature_map_width, n_boxes, 4))
# boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes)) # Set cx
# boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes)) # Set cy
# # boxes_tensor[:, :, :, 2] = wh_list[:, 0] # Set w
# # boxes_tensor[:, :, :, 3] = wh_list[:, 1] # Set h

# print("boxes_tensor ", boxes_tensor)
# sys.exit(0)

# groundP = [[-180, 1000],
#            [180, 1000], 
#            [180, 300], 
#            [-180, 300]]
# print(np.average(groundP, axis=0))
# sys.exit(0)
# y_true = [[0, 1, 0],
#           [0, 1, 0],
#           [0, 1, 0],
#           [0, 1, 0]]
# y_pred = [[0.95, 0.05, 0],
#           [0.05, 0.95, 0],
#           [0.05, 0.95, 0],
#           [0.05, 0.95, 0]]
# cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
# print(cce(y_true, y_pred).numpy())
# sys.exit(0)

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
def tflite_image_homography_test(dataset, net_input_img_size, x_anchors, y_anchors, max_lane_count):
    print("-------------------------------------------------------------------")
    print("image_test")

    def create_map(srcImgSize, groundSize):
        h, w = srcImgSize
        gh, gw = groundSize

        # calc homography
        imgP = [[501, 362], [781, 357],  [924, 499], [337, 510]]
        groundP = [[-180, 1000], [180, 1000],  [180, 300], [-180, 300]]
        for i in range(4):
            imgP[i][0] *= w / 1280.0
            imgP[i][1] *= h / 720.0
            groundP[i][0] = groundP[i][0] * 0.125 + gw / 2.0
            groundP[i][1] = gh - groundP[i][1] * 0.1
        
        groud_center = tuple(np.average(groundP, axis=0))
        R = cv2.getRotationMatrix2D(groud_center, -20.0, 1.0)
        rotate_groupP = []
        for gp in groundP:
            pp = np.matmul(R, [[gp[0]], [gp[1]], [1.0]])
            rotate_groupP.append([pp[0], pp[1]])
            

        H, _ = cv2.findHomography(np.float32(imgP), np.float32(rotate_groupP))
        _, invH = cv2.invert(H)
            
        map_x = np.zeros((gh, gw), dtype=np.float32)
        map_y = np.zeros((gh, gw), dtype=np.float32)
            
        for gy in range(gh):
            for gx in range(gw):
                nx, ny, nz = np.matmul(invH, [[gx], [gy], [1.0]])
                nx /= nz
                ny /= nz
                if (nx >= 0 and nx < w and ny >= 0 and ny < h):
                    map_x[gy][gx] = nx
                    map_y[gy][gx] = ny
                else:
                    map_x[gy][gx] = -1
                    map_y[gy][gx] = -1
        
        return H, map_x, map_y
       

    groundSize = (256, 256)
    map_x = None
    map_y = None

    for elem in dataset:
        test_img = elem[0]
        label_lanes = elem[1][0].numpy()
        label_h_samples = elem[2][0].numpy()
        b, h, w, c = test_img.shape
        
        # build map
        if map_x is None or map_y is None:
            H, map_x, map_y = create_map((h, w), groundSize)
       
        # project img
        test_img = np.reshape(test_img, newshape=(h, w, 3))
        gImg = cv2.remap(test_img, map_x, map_y, interpolation=cv2.INTER_NEAREST)
        lane_useage = np.zeros(shape=(max_lane_count), dtype=np.int32)


        # reprojrect anchors
        show = False
        for laneIdx in range(min(len(label_lanes), max_lane_count)):
            lane_data = label_lanes[laneIdx]
            
            px = -1
            py = -1
            pA = None
            pB = None
            count = 0
            for idx in range(len(lane_data) -1, 0, -1):
                dy = label_h_samples[idx]
                dx = lane_data[idx]

                if (dx < 0):
                    continue

                # roate dx, dy
                gx, gy, gz = np.matmul(H, [[dx], [dy], [1.0]])
                if gz > 0:
                    continue
                gx /= gz
                gy /= gz


                if gx < 0 or gy < 0 or gx>= groundSize[1] or gy >= groundSize[0]:
                    continue
                
                if pA is None:
                    pA = (gx, gy)
                elif count < 10:
                    pB = (gx, gy)
                count += 1
                

                cv2.circle(test_img, (dx, dy), 10, (0, 0, 255))
                if px != -1 and py != -1:
                    cv2.line(gImg, (px, py), (gx, gy), (0, 0, 255))
                px = gx
                py = gy

            if pA is None or pB is None:
                continue

            pA = np.array(pA)
            pB = np.array(pB)
            pV = (pA - pB) / (float)(np.linalg.norm(pA - pB))
            pC = pB + pV * (groundSize[0] - pB[1]) * (1.0 /pV[1])
            cv2.line(gImg, tuple(pA), tuple(pC), (0, 255, 0))


            x_scale = (float)(x_anchors) / (float)(groundSize[1])
            laneAnchorIdx = 0
            if pC[0] <0:
                laneAnchorIdx = 0
            elif pC[0] >= groundSize[1]:
                laneAnchorIdx = max_lane_count -1
            else:
                laneAnchorIdx = 1 + int((pC[0] * x_scale * (max_lane_count -2) ) / x_anchors)

            lane_useage[laneAnchorIdx] += 1
            if lane_useage[laneAnchorIdx] > 1:
                tf.print("data alearday used ", lane_useage, " ------> ", len(label_lanes))
                show = True

            print("pA ", tuple(pA), " , pB ", tuple(pB), " , pC ", tuple(pC), "---->", laneAnchorIdx)    
        print("-------------------------------------------------")
        resized_img = cv2.resize(test_img, (568, 256))

        if show:
            img = cv2.hconcat([resized_img, gImg])
            img = cv2.resize(img, (0, 0), fx=2, fy=2)
            cv2.imshow("preview", img)
            cv2.waitKey(0)


# --------------------------------------------------------------------------------------------------------------
def tflite_image_test_multi_lane(tflite_model_quant_file, dataset, net_input_img_size, x_anchors, y_anchors, max_lane_count):
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
    COLORS = np.random.randint(70, 255, [max_lane_count+1, 3], dtype=np.int32)

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

        # prediction = tf.keras.activations.softmax(tf.convert_to_tensor(prediction, dtype=tf.float32)).numpy()
        prediction = (prediction > 0.5)
            

        main_img = test_img
        main_img = cv2.cvtColor(main_img[0], cv2.COLOR_BGR2GRAY)
        main_img = cv2.cvtColor(main_img, cv2.COLOR_GRAY2BGR)
        
        zeros = np.zeros((y_anchors,  x_anchors -1), dtype=np.uint8)
        raw_output= []
        mask = None
        all_pred = None
        
        for si in range(max_lane_count):
        # for si in range(2):
            # if si == max_lane_count:
            #     continue
            # pred = prediction[0][si]
            pred = prediction[0,si,:, 0:x_anchors-1]
            img = cv2.merge([pred * COLORS[si][0], pred * COLORS[si][1], pred * COLORS[si][2]])
            img = np.uint8(img)

            img = cv2.resize(img, (W, H))
            if all_pred is None:
                all_pred = img
            else:
                cv2.line(all_pred, (all_pred.shape[1] -2, 0), (all_pred.shape[1] -2, 800), (255, 255, 255), 2)
                all_pred = cv2.hconcat([all_pred, img])
            
            # _, img = cv2.threshold(img, 50, 255, cv2.THRESH_TOZERO)
            if mask is None:
                mask = img
            else:
                mask = mask + img

        _, mask_bin = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY_INV)
        main_img = cv2.bitwise_and(main_img, mask_bin)
        main_img = cv2.bitwise_or (main_img, mask)

        prefix = 'build/' + str(idx)
        # target_szie = (1280, 720)
        target_szie = (800, 800)
        main_img = cv2.resize(main_img, target_szie)

        inv_dx = 1.0 / float(x_anchors)
        inv_dy = 1.0 / float(y_anchors)
        for dy in range(y_anchors):
            for dx in range(x_anchors):
                px = (inv_dx * dx) * target_szie[0]
                py = (inv_dy * dy) * target_szie[1]
                cv2.line(main_img, (int(px), 0), (int(px), target_szie[1]), (125, 125, 125))
                cv2.line(main_img, (0, int(py)), (target_szie[0], int(py)), (125, 125, 125))

        cv2.putText(main_img, 'accuracy '+str(acc), org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 0, 255))
        
        all_pred = cv2.resize(all_pred, (800, 180))
        main_img = cv2.vconcat((all_pred, main_img))
        # main_img = all_pred

        cv2.imshow("preview", main_img)
        key = cv2.waitKey(0)


# --------------------------------------------------------------------------------------------------------------
def tflite_image_test_instance_seg(tflite_model_quant_file, dataset, net_input_img_size, x_anchors, y_anchors, max_lane_count):
    print("-------------------------------------------------------------------")
    print("tflite_image_test_instance_seg")

    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
    interpreter.allocate_tensors()

    idx = 0
    W = net_input_img_size[0]
    H = net_input_img_size[1]
    
    input_index = interpreter.get_input_details()[0]
    output_index = interpreter.get_output_details()[0]
    
    lane_accuracy = LaneAccuracy()
    COLORS = np.random.randint(70, 255, [max_lane_count+1, 3], dtype=np.int32)

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
        # prediction = np.float32(test_label)

        main_img = test_img
        main_img = cv2.cvtColor(main_img[0], cv2.COLOR_BGR2GRAY)
        main_img = cv2.cvtColor(main_img, cv2.COLOR_GRAY2BGR)
        
        zeros = np.zeros((y_anchors,  x_anchors -1), dtype=np.uint8)
        raw_output= []
        mask = None

        groundSize = (256, 256)
        inv_anchor_scale_x = (float)(groundSize[1]) / (float)(x_anchors)
        inv_anchor_scale_y = (float)(groundSize[0]) / (float)(y_anchors) 
        line_mask = main_img.copy()
        for dy in range(y_anchors):
            for dx in range(x_anchors):
                offset = prediction[0,dy,dx,2]
                offset = math.exp(offset)

                ggx = dx * inv_anchor_scale_x
                ggy = dy * inv_anchor_scale_y

                pgx = ggx + offset
                pgy = ggy
                if prediction[0,dy,dx,0] > 0.5:
                    cv2.line(line_mask, (int(ggx), int(ggy)), (int(pgx), int(pgy)), (0, 0, 255))
                    cv2.circle(line_mask, (int(pgx), int(pgy)), 2, (0, 255, 0))
        line_mask = cv2.hconcat([line_mask, line_mask])

        pred_0 = prediction[0,:,:,3]
        pred_1 = prediction[0,:,:,4]
        pred_2 = prediction[0,:,:,5]
        pred_3 = prediction[0,:,:,6]
        pred_4 = prediction[0,:,:,7]
        pred_5 = prediction[0,:,:,8]

        # print("pred_0 ", pred_0)

        img = cv2.merge([pred_2 * 255, pred_1 * 255, pred_0 * 255])
        img = np.uint8(img)
        
        img2 = cv2.merge([pred_5 * 255, pred_4 * 255, pred_3 * 255])
        img2 = np.uint8(img2)

        img = cv2.hconcat([img, img2])
        cv2.line(img, (W, int(0)), (W, int(H)), (255, 255, 255), 5)


        mask = cv2.resize(img, (W  * 2, H))
        mask = cv2.bitwise_or(mask, line_mask)

        # main_img = cv2.bitwise_or (main_img, mask)
        main_img = mask
        

        prefix = 'build/' + str(idx)
        # target_szie = (1280, 720)
        target_szie = (800 *2, 800)
        main_img = cv2.resize(main_img, target_szie)

        inv_dx = 1.0 / float(x_anchors)
        inv_dy = 1.0 / float(y_anchors)
        for dy in range(y_anchors):
            for dx in range(x_anchors):
                px = (inv_dx * dx) * target_szie[0]
                py = (inv_dy * dy) * target_szie[1]
                cv2.line(main_img, (int(px), 0), (int(px), target_szie[1]), (125, 125, 125))
                cv2.line(main_img, (0, int(py)), (target_szie[0], int(py)), (125, 125, 125))

        cv2.imshow("preview", main_img)
        key = cv2.waitKey(0)


# --------------------------------------------------------------------------------------------------------------
def tflite_image_test_instance_seg_post_process(tflite_model_quant_file, dataset, net_input_img_size, x_anchors, y_anchors, max_lane_count):
    print("-------------------------------------------------------------------")
    print("tflite_image_test_instance_seg_post_process")

    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
    interpreter.allocate_tensors()

    idx = 0
    W = net_input_img_size[0]
    H = net_input_img_size[1]
    
    input_index = interpreter.get_input_details()[0]
    output_index = interpreter.get_output_details()[0]
    
    lane_accuracy = LaneAccuracy()
    COLORS = [(0, 0, 255), 
              (0, 255, 0), 
              (255, 0, 0), 
              (0, 255, 255), 
              (255, 0, 255), 
              (255, 255), ]

    for elem in dataset:
        test_img = elem[0]
        test_label = elem[1]

        if input_index['dtype']== np.uint8:
            test_img = np.uint8(test_img * 255)

        # inference
        interpreter.set_tensor(input_index["index"], test_img)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_index["index"])
        prediction = tf.convert_to_tensor(prediction)

        main_img = test_img
        main_img = cv2.cvtColor(main_img[0], cv2.COLOR_BGR2GRAY)
        main_img = cv2.cvtColor(main_img, cv2.COLOR_GRAY2BGR)
        
        # ==================================================================
        # post processing
        embedding_count = 6
        pred_class  = prediction[:,:,:,0:2]
        pred_offset = prediction[:,:,:,2:3]
        pred_embeddings = prediction[:,:,:,3:]
        
        # do threshold to crop class prob and create a bool mask 
        PROB_THRESHOLD = 0.5
        lane_prob = prediction[:,:,:,0]
        # lane_prob = tf.clip_by_value(lane_prob, PROB_THRESHOLD, 1.0)     # set prob > 0.5 as valid  anchor
        threshold_vector = tf.constant([PROB_THRESHOLD], dtype=tf.float32)
        lane_prob_mask = tf.greater_equal(lane_prob, threshold_vector)

        
        ones = tf.ones(tf.shape(lane_prob_mask), dtype=tf.float32)
        zeros = tf.zeros(tf.shape(lane_prob_mask), dtype=tf.float32)
        lane_prob_mask = tf.where(lane_prob_mask, ones, zeros)
        lane_prob_mask = tf.expand_dims(lane_prob_mask, axis=-1)
        lane_prob_mask = tf.tile(lane_prob_mask, multiples=[1, 1, 1, 6])
        lane_prob_mask = tf.cast(lane_prob_mask, tf.float32)

        # do threshold to crop embeddings and create a bool mask 
        PROB_THRESHOLD = 0.5
        pred_embeddings = tf.clip_by_value(pred_embeddings, PROB_THRESHOLD, 1.0)     # set prob > 0.5 as valid  anchor
        threshold_vector = tf.constant([PROB_THRESHOLD], dtype=tf.float32)
        embedding_mask = tf.not_equal(pred_embeddings, threshold_vector)
        embedding_mask = tf.cast(embedding_mask, tf.float32)    #[batch, anchor_height, anchor_width, mask(0, 1)]

        # remove embeddings by prob mask
        embeddings = tf.multiply(embedding_mask, lane_prob_mask)     #[batch, anchor_height, anchor_width, embeddings]


        # create anchor coordinate maps
        groundSize = (256, 256)
        inv_anchor_scale_x = (float)(groundSize[1]) / (float)(x_anchors)
        inv_anchor_scale_y = (float)(groundSize[0]) / (float)(y_anchors) 

        anchor_x_axis = tf.range(0, x_anchors, dtype=tf.float32)
        anchor_x_axis = tf.expand_dims(anchor_x_axis, axis=0)
        anchor_x_axis = tf.tile(anchor_x_axis, [y_anchors, 1])
        anchor_x_axis = tf.multiply(anchor_x_axis, tf.constant(inv_anchor_scale_x, dtype=tf.float32))
        anchor_x_axis = tf.expand_dims(anchor_x_axis, axis=-1)  # expand as same dim as embeddings
        anchor_x_axis = tf.expand_dims(anchor_x_axis, axis=0)   # expand batch
        anchor_x_axis = tf.tile(anchor_x_axis, multiples=[1, 1, 1, embedding_count])
        # tf.print("anchor_x_axis ", anchor_x_axis, summarize=-1)
        # tf.print("anchor_x_axis ", tf.shape(anchor_x_axis))


        # filter data by embeddings and  decode offsets by exp
        offsets = tf.exp(pred_offset)
        anchor_x_axis = tf.add(anchor_x_axis, offsets) 
        anchor_x_axis = tf.multiply(embeddings, anchor_x_axis)

        # check the variance of embeddings by row, ideally, we want each row of embeddings containt only one embedding 
        # to identify instance of lane, but in some case, over one embedding at same row would happened.
        # In this step, we filter embeddings at each row by the position variance.
        sum_of_embedding_count_x = tf.reduce_sum(embeddings, axis=2)
        sum_of_embedding_count_x = tf.clip_by_value(sum_of_embedding_count_x, 1.0, x_anchors)
        sum_of_axis_x = tf.reduce_sum(anchor_x_axis, axis=2)
        mean_of_axis_x = tf.divide(sum_of_axis_x, sum_of_embedding_count_x)

        mean_of_axis_x = tf.expand_dims(mean_of_axis_x, axis=2)
        mean_of_axis_x = tf.tile(mean_of_axis_x, [1, 1, x_anchors, 1])
        # tf.print("mean_of_axis_x ", tf.shape(mean_of_axis_x))
        
        # threshold
        dpulicated_threshold = 5.0
        diff_of_axis_x = tf.abs(tf.subtract(anchor_x_axis, mean_of_axis_x))
        diff_of_axis_x = tf.less_equal(diff_of_axis_x, tf.constant(dpulicated_threshold, dtype=tf.float32))

        ones = tf.ones(tf.shape(diff_of_axis_x))
        zeros = tf.zeros(tf.shape(diff_of_axis_x))
        mask_of_mean_offset = tf.where(diff_of_axis_x, ones, zeros)

        embeddings = tf.multiply(mask_of_mean_offset, embeddings)
        anchor_x_axis = tf.multiply(mask_of_mean_offset, anchor_x_axis)

        # recalcuate average lane markings
        sum_of_embedding_count_x = tf.reduce_sum(embeddings, axis=2)
        count_of_valid_point = tf.transpose(sum_of_embedding_count_x, perm=[0, 2, 1])
        count_of_valid_point = tf.expand_dims(count_of_valid_point, axis=-1)
        sum_of_embedding_count_x = tf.clip_by_value(sum_of_embedding_count_x, 1.0, x_anchors)
        sum_of_axis_x = tf.reduce_sum(anchor_x_axis, axis=2)
        mean_of_axis_x = tf.divide(sum_of_axis_x, sum_of_embedding_count_x)
        tf.print("mean_of_axis_x ", tf.shape(mean_of_axis_x))

        mean_of_axis_x = tf.transpose(mean_of_axis_x, perm=[0, 2, 1])   # [batch, height, instance] -> [batch, instance, height]
        mean_of_axis_x = tf.expand_dims(mean_of_axis_x, axis=-1)
        
        # generate y axis data
        anchor_y_axis = tf.range(0, y_anchors, dtype=tf.float32)
        anchor_y_axis = tf.multiply(anchor_y_axis, tf.constant(inv_anchor_scale_y, dtype=tf.float32))
        anchor_y_axis = tf.expand_dims(anchor_y_axis, axis=-1)
        anchor_y_axis = tf.expand_dims(anchor_y_axis, axis=0)
        anchor_y_axis = tf.expand_dims(anchor_y_axis, axis=0)
        anchor_y_axis = tf.tile(anchor_y_axis, [1, embedding_count, 1, 1])

        # generate confidence data
        
        result = tf.concat([mean_of_axis_x, anchor_y_axis, count_of_valid_point], axis=-1)
        
        tf.print("mean_of_axis_x ", tf.shape(mean_of_axis_x))
     
        # rendering
        for laneIdx in range(embedding_count):
            for dy in range(y_anchors):
                for dx in range(x_anchors):
                    # offset = offsets[0,dy,dx,laneIdx]
                    # offset = math.exp(offset)

                    # ggx = anchor_x_axis[0,dy,dx,laneIdx]
                    # ggx = mean_of_axis_x[0, laneIdx, dy, 0]
                    # ggy = anchor_y_axis[0, laneIdx, dy,0]
                    gx = result[0, laneIdx, dy, 0]
                    gy = result[0, laneIdx, dy, 1]
                    gb = result[0, laneIdx, dy, 2]

                    if gb > 0.5:
                        cv2.circle(main_img, (int(gx), int(gy)), 2, tuple(int(x) for x in COLORS[laneIdx]))


        
        # for laneIdx in range(embedding_count):
        #     for dy in range(y_anchors):
        #         for dx in range(x_anchors):
        #             offset = offsets[0,dy,dx,laneIdx]
        #             # offset = math.exp(offset)

        #             ggx = anchor_x_axis[0,dy,dx,laneIdx]
        #             ggy = anchor_y_axis[0,dy,dx,laneIdx]
        #             # ggx = dx * inv_anchor_scale_x
        #             # ggy = dy * inv_anchor_scale_y

        #             pgx = ggx + offset
        #             pgy = ggy
        #             if embeddings[0,dy,dx,laneIdx] > 0.5:
        #                 cv2.line(main_img, (int(ggx), int(ggy)), (int(pgx), int(pgy)), (0, 0, 255))
        #                 cv2.circle(main_img, (int(pgx), int(pgy)), 2, tuple(int(x) for x in COLORS[laneIdx]))
        
        # for dy in range(y_anchors):
        #     for dx in range(x_anchors):
        #         offset = prediction[0,dy,dx,2]
        #         offset = math.exp(offset)
        #         ggx = dx * inv_anchor_scale_x
        #         ggy = dy * inv_anchor_scale_y
        #         pgx = ggx + offset
        #         pgy = ggy
        #         if prediction[0,dy,dx,0] > 0.5:
        #             cv2.line(main_img, (int(ggx), int(ggy)), (int(pgx), int(pgy)), (0, 0, 255))
        #             cv2.circle(main_img, (int(pgx), int(pgy)), 2, (0, 255, 0))



        # remove the embeddings where prob less than threshold
        # clip_embeddings = tf.boolean_mask(pred_embeddings, prob_bool_mask)
        # tf.multiply()
        # tf.print("clip_embeddings ", clip_embeddings)
        # tf.print("clip_embeddings", tf.shape(clip_embeddings))
        
        # mask = None
        # groundSize = (256, 256)
        # inv_anchor_scale_x = (float)(groundSize[1]) / (float)(x_anchors)
        # inv_anchor_scale_y = (float)(groundSize[0]) / (float)(y_anchors) 
        # line_mask = main_img.copy()
        # for dy in range(y_anchors):
        #     for dx in range(x_anchors):
        #         offset = prediction[0,dy,dx,2]
        #         offset = math.exp(offset)

        #         ggx = dx * inv_anchor_scale_x
        #         ggy = dy * inv_anchor_scale_y

        #         pgx = ggx + offset
        #         pgy = ggy
        #         if prediction[0,dy,dx,0] > 0.5:
        #             cv2.line(line_mask, (int(ggx), int(ggy)), (int(pgx), int(pgy)), (0, 0, 255))
        #             cv2.circle(line_mask, (int(pgx), int(pgy)), 2, (0, 255, 0))
        # line_mask = cv2.hconcat([line_mask, line_mask])

        # pred_0 = prediction[0,:,:,3]
        # pred_1 = prediction[0,:,:,4]
        # pred_2 = prediction[0,:,:,5]
        # pred_3 = prediction[0,:,:,6]
        # pred_4 = prediction[0,:,:,7]
        # pred_5 = prediction[0,:,:,8]

        # # print("pred_0 ", pred_0)

        # img = cv2.merge([pred_2 * 255, pred_1 * 255, pred_0 * 255])
        # img = np.uint8(img)
        
        # img2 = cv2.merge([pred_5 * 255, pred_4 * 255, pred_3 * 255])
        # img2 = np.uint8(img2)

        # img = cv2.hconcat([img, img2])
        # cv2.line(img, (W, int(0)), (W, int(H)), (255, 255, 255), 5)


        # mask = cv2.resize(img, (W  * 2, H))
        # mask = cv2.bitwise_or(mask, line_mask)

        # # main_img = cv2.bitwise_or (main_img, mask)
        # main_img = mask
        

        # prefix = 'build/' + str(idx)
        target_szie = (800, 800)
        # target_szie = (800 *2, 800)
        main_img = cv2.resize(main_img, target_szie)

        inv_dx = 1.0 / float(x_anchors)
        inv_dy = 1.0 / float(y_anchors)
        for dy in range(y_anchors):
            for dx in range(x_anchors):
                px = (inv_dx * dx) * target_szie[0]
                py = (inv_dy * dy) * target_szie[1]
                cv2.line(main_img, (int(px), 0), (int(px), target_szie[1]), (125, 125, 125))
                cv2.line(main_img, (0, int(py)), (target_szie[0], int(py)), (125, 125, 125))

        cv2.imshow("preview", main_img)
        key = cv2.waitKey(0)


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
    COLORS = np.random.randint(70, 255, [max_lane_count+1, 3], dtype=np.int32)

    for elem in dataset:
        test_img = elem[0]
        test_label = elem[1]

        if input_index['dtype']== np.uint8:
            test_img = np.uint8(test_img * 255)

        # # inference
        interpreter.set_tensor(input_index["index"], test_img)
        interpreter.invoke()
        np.set_printoptions(threshold=sys.maxsize)
        prediction = interpreter.get_tensor(output_index["index"])
        # prediction = np.float32(test_label)

        # check accuracy
        # y_pred = prediction
        # y_true = test_label
        # acc = lane_accuracy(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred, dtype=tf.float32))

        # prediction = tf.keras.activations.softmax(tf.convert_to_tensor(prediction, dtype=tf.float32)).numpy()
        # prediction = (prediction > 0.5)
            
        main_img = test_img[0]
        main_img = cv2.cvtColor(main_img, cv2.COLOR_BGR2GRAY)
        main_img = cv2.cvtColor(main_img, cv2.COLOR_GRAY2BGR)
        
        all_pred = None

        pred = prediction[0,:,:,0]
        img = cv2.merge([pred * COLORS[0][0], pred * COLORS[0][1], pred * COLORS[0][2]])
        img = np.uint8(img)
        img = cv2.resize(img, (W, H))

        _, mask_bin = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
        # main_img = cv2.bitwise_and(main_img, mask_bin)
        # main_img = cv2.bitwise_or (main_img, img)

        groundSize = (256, 256)
        inv_anchor_scale_x = (float)(groundSize[1]) / (float)(x_anchors)
        inv_anchor_scale_y = (float)(groundSize[0]) / (float)(y_anchors) 
        
        for dy in range(y_anchors):
            for dx in range(x_anchors):
                offset = prediction[0,dy,dx,2]
                offset = math.exp(offset)

                ggx = dx * inv_anchor_scale_x
                ggy = dy * inv_anchor_scale_y

                pgx = ggx + offset
                pgy = ggy
                if prediction[0,dy,dx,0] > 0.5:
                    cv2.line(main_img, (int(ggx), int(ggy)), (int(pgx), int(pgy)), (0, 0, 255))
                    cv2.circle(main_img, (int(pgx), int(pgy)), 2, (0, 255, 0))
                

        prefix = 'build/' + str(idx)
        # target_szie = (1280, 720)
        target_szie = (800, 800)
        main_img = cv2.resize(main_img, target_szie)

        inv_dx = 1.0 / float(x_anchors)
        inv_dy = 1.0 / float(y_anchors)
        for dy in range(y_anchors):
            for dx in range(x_anchors):
                px = (inv_dx * dx) * target_szie[0]
                py = (inv_dy * dy) * target_szie[1]
                cv2.line(main_img, (int(px), 0), (int(px), target_szie[1]), (125, 125, 125))
                cv2.line(main_img, (0, int(py)), (target_szie[0], int(py)), (125, 125, 125))

        # cv2.putText(main_img, 'accuracy '+str(acc), org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 0, 255))
        
        # all_pred = cv2.resize(all_pred, (800, 180))
        # main_img = cv2.vconcat((all_pred, main_img))
        # main_img = all_pred

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
                        # validation_data=valid_batches,
                        # validation_freq=5
                        )
    return history
    


# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':


    # config tensorflow to prevent out of memory when training
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # net_input_img_size = (512, 288)
    net_input_img_size = (256, 256)
    # x_anchors = 128
    # y_anchors = 72
    # x_anchors = 96
    # y_anchors = 54
    x_anchors = 32
    y_anchors = 32
    max_lane_count = 10
    train_dataset_path = "/home/dana/Datasets/ML/TuSimple/train_set"
    train_label_set = ["label_data_0313.json", "label_data_0531.json", "label_data_0601.json"]

    test_dataset_path = "/home/dana/Datasets/ML/TuSimple/test_set"
    test_label_set = ["test_label.json"]

    full_dataset_path = "/home/dana/Datasets/ML/TuSimple/full_set"
    full_label_set = [
        "label_data_0313.json",
        "label_data_0531.json",
        "label_data_0601.json",
        "test_label.json"
        ]

    another_dataset_path = "/home/dana/Datasets/ML/TuSimple/another_test"
    another_label_set = ["test.json"]
    
    setA_dataset_path = "/home/dana/Datasets/ML/TuSimple/set_A"
    setA_label_set = ["test.json"]

    setB_dataset_path = "/home/dana/Datasets/ML/TuSimple/set_B"
    setB_label_set = ["test.json"]

    augmentation = True
    batch_size=32
    train_batches = datasets.TusimpleLane(full_dataset_path, full_label_set, net_input_img_size, x_anchors, y_anchors, max_lane_count, augmentation=augmentation)
    representative_dataset = train_batches.batch(1)
    train_batches = train_batches.shuffle(1000).batch(batch_size)

    # valid_batches = datasets.TusimpleLane(test_dataset_path, test_label_set, net_input_img_size, x_anchors, y_anchors, max_lane_count, False)
    valid_batches = datasets.TusimpleLane(full_dataset_path, full_label_set, net_input_img_size, x_anchors, y_anchors, max_lane_count, augmentation=augmentation)
    valid_batches = datasets.TusimpleLane(setA_dataset_path, setA_label_set, net_input_img_size, x_anchors, y_anchors, max_lane_count, augmentation=False)
    # valid_batches = datasets.TusimpleLane(setB_dataset_path, setB_label_set, net_input_img_size, x_anchors, y_anchors, max_lane_count, augmentation=False)
    valid_batches = valid_batches.batch(1)

    # for elem in valid_batches:
    #     test_img = elem[0]
    #     test_label = elem[1]
    #     test_img = np.uint8(test_img[0] * 255)
    #     cv2.imshow("preview", test_img)
    #     # print(test_label.shape)
    #     key = cv2.waitKey(0)

    # tflite_model_quant_file = 'model_int8.tflite'
    # tflite_image_test(tflite_model_quant_file, valid_batches, net_input_img_size, x_anchors, y_anchors, max_lane_count)
    # tflite_image_test_instance_seg(tflite_model_quant_file, valid_batches, net_input_img_size, x_anchors, y_anchors, max_lane_count)
    # tflite_image_test_instance_seg_post_process(tflite_model_quant_file, valid_batches, net_input_img_size, x_anchors, y_anchors, max_lane_count)
    # sys.exit(0)

    # checkpoint_path = "tmp/fake-quantized-checkpoint/"
    # checkpoint_path = "tmp/non-QAT-checkpoint/"
    checkpoint_path = "tmp/perspective-checkpoint/"
    flag_training=True
    flag_training=False

    if flag_training:
        model = AlphaLaneModel(net_input_img_size, x_anchors, y_anchors, max_lane_count,
                               training=flag_training,
                               quantization_aware_training=False, input_batch_size=None)
    else:
        model = AlphaLaneModel(net_input_img_size, x_anchors, y_anchors, max_lane_count,
                               training=flag_training,
                               quantization_aware_training=False, input_batch_size=1)

    model.summary()
    # tf.keras.utils.plot_model(model, 'model.png')
    
    if not flag_training:
        model.load_weights(tf.train.latest_checkpoint(checkpoint_path))     # load pretrained
    model.load_weights(tf.train.latest_checkpoint(checkpoint_path))     # load p/retrained
    # sys.exit(0)

    if flag_training:
        # model.compile(
        #               optimizer=tf.keras.optimizers.Nadam(lr=0.001),
        #               loss_weights=[1.0,
        #                             1.0],
        #               loss={'label_output' : LaneLoss(),
        #                     'confidence_output' : ConfidenceLoss()},
        #               metrics={'label_output' : LaneAccuracy(),
        #                        'confidence_output' : ConfidenceAccuracy()})

        model.compile(optimizer=tf.keras.optimizers.Nadam(lr=0.001),
                      loss=[LaneLoss()],
                    #   metrics=[LaneAccuracy()]
                      )

        # model.compile(
        #               optimizer=tf.keras.optimizers.Adam(lr=0.0001),
        #               loss=[DiscriminativeLoss()],
        #             #   metrics=[LaneAccuracy()]
        #               )

        train(model, train_batches, valid_batches, checkpoint_path=checkpoint_path, train_epochs=200)

    # Convert the model.
    print("---------------------------------------------------")
    print("Conver model (int8)")
    print("---------------------------------------------------")
    tf.lite.TFLiteConverter.from_concrete_functions
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # ------------------------------------------------
    ## at QAT, with these flags will cause many error.
    converter.representative_dataset = representative_data_gen(representative_dataset)
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32
    converter.experimental_new_converter=False
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

    print("---------------------------------------------------")
    print("Evulate accuracy")
    print("---------------------------------------------------")
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
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
    interpreter.allocate_tensors()
    # tflite_image_test(tflite_model_quant_file, valid_batches, net_input_img_size, x_anchors, y_anchors, max_lane_count)
    tflite_image_test_instance_seg_post_process(tflite_model_quant_file, valid_batches, net_input_img_size, x_anchors, y_anchors, max_lane_count)

    # start = time.time()
    # for i in range(100):
    #     interpreter.set_tensor(input_index, test_image)
    #     interpreter.invoke()
    #     predictions = interpreter.get_tensor(output_index)
    # end = time.time()
    # time_inference = (end - start) / float(100)

    # print("predict          avg time", time_inference, "s")

    sys.exit(0)
