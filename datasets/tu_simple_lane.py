import tensorflow as tf
import time
import os
import json
import sys
import itertools
import random
import numpy as np
import math
from PIL import Image
import cv2
import xml.etree.ElementTree as ET
from scipy.interpolate import interp1d
import datasets.cvat_dataset
import tensorflow_datasets as tfds


class TusimpleLane:

    # ----------------------------------------------------------------------------------------
    def __init__(self,
                 dataset_path,
                 label_set,
                 config,
                 augmentation=False):
        if not os.path.exists(dataset_path):
            print("File doesn't exist, path : ", dataset_path)
            exit()

        # get data from config
        x_anchors = config["model_info"]["x_anchors"]
        y_anchors = config["model_info"]["y_anchors"]
        max_lane_count = config["model_info"]["max_lane_count"]
        ground_img_size = config["model_info"]["input_image_size"]

        # build map
        augmentation_deg = None
        if augmentation:
            augmentation_deg = [0.0, -15.0, 15.0, -30.0, 30.0]
        else:
            augmentation_deg = [0.0]
        dates = list(config["perspective_info"].keys())
        perspective_infos = {}
        for date in dates:
            H_list, map_x_list, map_y_list = create_map(config, date, augmentation_deg=augmentation_deg)
            # H_list = tf.constant(H_list)
            # map_x_list = tf.constant(map_x_list)
            # map_y_list = tf.constant(map_y_list)
            perspective_infos[date] = (H_list, map_x_list, map_y_list)

        # pipe = [tf.data.Dataset.from_tensors(i) for i in self._data_reader(dataset_path, label_set[0], augmentation_deg)][0]
        # pipe = tf.data.Dataset.from_tensor_slices(label_set, "CVAT")
        # print(list(pipe.as_numpy_iterator()))
        # pipe = tf.data.Dataset.from_tensor_slices(pipe.as_numpy_iterator(), "CVAT")
        # pipe = tf.data.Dataset.from_generator(self._data_reader,
        #                                        output_types=(tf.dtypes.uint8,
        #                                                      tf.dtypes.int32,
        #                                                      tf.dtypes.int32,
        #                                                      tf.dtypes.int32),
        #                                        output_shapes=((None),
        #                                                       (None),
        #                                                       (None),
        #                                                       (None)),
        #                                        args=(dataset_path,
        #                                              label_set[0],
        #                                              augmentation_deg)
        #                                        )
        # build dataset
        label_set = tf.data.Dataset.from_tensor_slices(label_set)
        pipe = label_set.interleave(
            lambda label_file_name: tf.data.Dataset.from_generator(_data_reader,
                                                                   output_types=(tf.dtypes.uint8,
                                                                                 tf.dtypes.int32,
                                                                                 tf.dtypes.int32,
                                                                                 tf.dtypes.int32),
                                                                   output_shapes=(None,
                                                                                  None,
                                                                                  None,
                                                                                  None),
                                                                   args=(dataset_path,
                                                                         label_file_name,
                                                                         augmentation_deg,
                                                                         x_anchors,
                                                                         y_anchors,
                                                                         ground_img_size,
                                                                         perspective_infos,
                                                                         max_lane_count)
                                                                   )
        )

        pipe = pipe.map(
            # convert data to training label and norimalization
            lambda image, label_lanes, label_h_samples, refIdx:
            tf.numpy_function(func=_map_projection_data_generator,
                              inp=[image,
                                   label_lanes,
                                   label_h_samples,
                                   x_anchors,
                                   y_anchors,
                                   max_lane_count,
                                   H_list[refIdx],
                                   map_x_list[refIdx],
                                   map_y_list[refIdx],
                                   ground_img_size],
                              Tout=[tf.dtypes.float32,
                                    tf.dtypes.float32])
            ,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        pipe = pipe.prefetch(  # Overlap producer and consumer works
            tf.data.experimental.AUTOTUNE
        )
        self.pipe = pipe

    def get_pipe(self):
        return self.pipe

    # ----------------------------------------------------------------------------------------


def _data_reader_cvat(dataset_path: str,
                      label_data_name: str,
                      augmentation_deg: float,
                      ground_img_size: int,
                      tranformation_params):
    print("Load data from ", dataset_path, label_data_name)
    label_data_path = os.path.join(dataset_path, label_data_name)
    if not os.path.exists(label_data_path):
        print("Label file doesn't exist, path : ", label_data_path)
        sys.exit(0)

    tree = ET.parse(label_data_path)
    root = tree.getroot()

    for image in root.findall('image'):
        y_coordinates = []
        x_coordinates = []

        image_path = str(os.path.join(str(dataset_path, 'utf-8'), str(image.attrib['name'])))
        with Image.open(image_path) as img:
            image_ary = np.asarray(img)
        for polyline in image.findall('polyline'):
            points = polyline.attrib['points'].split(";")
            for point in points:
                point = point.split(',')
                x_coordinates.append(int(point[0]))
                y_coordinates.append(int(point[1]))

        if augmentation_deg is None:
            yield image_ary, x_coordinates, y_coordinates, 0
        else:
            for refIdx in range(len(augmentation_deg)):
                yield _map_projection_data_generator(image_ary, x_coordinates, y_coordinates, refIdx)
                yield image_ary, x_coordinates, y_coordinates, refIdx

    # load data


def _data_reader(dataset_path,
                 label_data_name: str,
                 augmentation_deg: list,
                 x_anchors: int,
                 y_anchors: int,
                 ground_img_size: list,
                 tranformation_params,
                 max_lane_count: int):
    print("Load data from ", dataset_path, label_data_name)
    label_data_path = os.path.join(dataset_path, label_data_name)
    if not os.path.exists(label_data_path):
        print("Label file doesn't exist, path : ", label_data_path)
        sys.exit(0)

    count = 0

    # load data
    with open(label_data_path, 'r') as reader:
        for line in reader.readlines():
            raw_label = json.loads(line)
            image_path = os.path.join(str(dataset_path, 'utf-8'), raw_label["raw_file"])
            label_lanes = raw_label["lanes"]
            label_h_samples = raw_label["h_samples"]

            # read image
            with Image.open(image_path) as image:
                image_ary = np.asarray(image)

            count += 1

            H, map_x, map_y = tranformation_params[raw_label["raw_file"][0:10]]
            if augmentation_deg is None:
                yield _map_projection_data_generator(image_ary,
                                                     label_lanes,
                                                     label_h_samples,
                                                     x_anchors,
                                                     y_anchors,
                                                     max_lane_count,
                                                     H[0],
                                                     map_x[0],
                                                     map_y[0],
                                                     ground_img_size,)
            else:
                for refIdx in range(len(augmentation_deg)):
                    yield _map_projection_data_generator(image_ary,
                                                         label_lanes,
                                                         label_h_samples,
                                                         x_anchors,
                                                         y_anchors,
                                                         max_lane_count,
                                                         H[refIdx],
                                                         map_x[refIdx],
                                                         map_y[refIdx],
                                                         ground_img_size,)


def create_map(config, day_of_recording: str,
               augmentation_deg=None):
    transformation_settings = config["perspective_info"].get(day_of_recording, None)
    if transformation_settings is None:
        print("No transformation settings found for day of recording: ", day_of_recording, file=sys.stderr)
        return None
    src_img_size = transformation_settings["image_size"]
    ground_size = config["model_info"]["input_image_size"]

    w, h = src_img_size
    gw, gh = ground_size

    # calc homography (TuSimple fake)
    imgP = [transformation_settings["image_p0"],
            transformation_settings["image_p1"],
            transformation_settings["image_p2"],
            transformation_settings["image_p3"]]
    groundP = [transformation_settings["ground_p0"],
               transformation_settings["ground_p1"],
               transformation_settings["ground_p2"],
               transformation_settings["ground_p3"]]
    ground_scale_width = config["model_info"]["ground_scale_width"]
    ground_scale_height = config["model_info"]["ground_scale_height"]

    # We only use one perspective matrix for image transform, therefore, all images
    # at dataset must have same size, or the perspective transormation may fail.
    # In default, we assume the camera image input size is 1280x720, so the following
    # step will resize the image point for size fitting.
    # for i in range(len(imgP)):
    #     imgP[i][0] *= w / 1280.0
    #     imgP[i][1] *= h / 720.0

    # Scale the ground points, we assume the camera position is center of perspectived image,
    # as shown at following codes :
    #     (perspectived image with ground_size)
    #     ################################
    #     #             +y               #
    #     #              ^               #
    #     #              |               #
    #     #              |               #
    #     #              |               #
    #     #          p0 --- p1           #
    #     #          |      |            #
    #     #          p3 --- p2           #
    #     #              |               #
    #     # -x ----------C------------+x #
    #     ################################
    #
    # for i in range(len(groundP)):
    #     groundP[i][0] = groundP[i][0] * gw / w  # ground_scale_width + gw / 2.0
    #     groundP[i][1] = groundP[i][1] * gh / h  # gh - groundP[i][1] * ground_scale_height

    list_H = []
    list_map_x = []
    list_map_y = []

    groud_center = tuple(np.average(groundP, axis=0))
    if augmentation_deg is None:
        augmentation_deg = [0.0]

    for deg in augmentation_deg:
        R = cv2.getRotationMatrix2D(groud_center, deg, 1.0)
        rotate_groupP = []
        for gp in groundP:
            pp = np.matmul(R, [[gp[0]], [gp[1]], [1.0]])
            rotate_groupP.append([pp[0], pp[1]])

        # H, _ = cv2.findHomography(np.float32(imgP), np.float32(rotate_groupP))
        H = cv2.getPerspectiveTransform(np.float32(imgP), np.float32(rotate_groupP))
        _, invH = cv2.invert(H)

        map_x = np.zeros((gh, gw), dtype=np.float32)
        map_y = np.zeros((gh, gw), dtype=np.float32)

        for gy in range(gh):
            for gx in range(gw):
                nx, ny, nz = np.matmul(invH, [[gx], [gy], [1.0]])
                nx /= nz
                ny /= nz
                if 0 <= nx < gw and 0 <= ny < gh:
                    map_x[gy][gx] = nx
                    map_y[gy][gx] = ny
                else:
                    map_x[gy][gx] = -1
                    map_y[gy][gx] = -1

        list_H.append(H)
        list_map_x.append(map_x)
        list_map_y.append(map_y)

    return list_H, list_map_x, list_map_y


def _map_projection_data_generator(src_image,
                                   label_lanes,
                                   label_h_samples,
                                   net_input_img_size,
                                   x_anchors,
                                   y_anchors,
                                   max_lane_count,
                                   H,
                                   map_x,
                                   map_y,
                                   groundSize,
                                   cutoffs):
    # transform image by perspective matrix
    height, width = src_image.shape[:2]
    # if width != 1280 or height != 720:
    #     src_image = cv2.resize(src_image, (1280, 720))
    # src_img_cpy = src_image.copy()
    # for l in label_lanes:
    #     for sz in range(len(label_h_samples)):
    #         cv2.circle(src_img_cpy, (int(l[sz]), int(label_h_samples[sz])), 2, (0, 255, 0), -1)

    # gImg = cv2.remap(src_image, np.array(map_x), np.array(map_y),
    #                  interpolation=cv2.INTER_NEAREST,
    #                  borderValue=(125, 125, 125))

    # Multiplying the transformation matrix with an identity matrix in numpy transforms the "Eager Tensor" to a numpy array
    # cv2.warpPerspective() requires a numpy array as input and will fail on an "Eager Tensor"
    gImg = cv2.warpPerspective(src_image, np.matmul(H, np.eye(3, 3)), (1280, 960), flags=cv2.INTER_LINEAR,)

    l_c, r_c, u_c, d_c = cutoffs
    gImg = gImg[u_c:d_c, l_c:r_c]
    gImg = cv2.resize(gImg, (net_input_img_size[1], net_input_img_size[0]))

    #################################
    # t_height, t_width = gImg.shape[:2]
    grid_size = 32
    # for x in range(0, t_width, t_width // grid_size):
    #     cv2.line(gImg, (x, 0), (x, t_height), (255, 0, 0), 1)  # Blue lines
    #
    # for y in range(0, t_height, t_height // grid_size):
    #     cv2.line(gImg, (0, y), (t_width, y), (255, 0, 0), 1)  # Blue lines

    #################################

    imgf = np.float32(gImg) * (1.0 / 255.0)
    # cv2.imwrite("test.jpg", gImg)

    # create label for class
    class_list = {'background': [0, 1],
                  'lane_marking': [1, 0]}
    class_count = len(class_list)  # [background, road]

    # create label for slice id mapping from  ground x anchor, and y anchor
    #   [y anchors,
    #    x anchors,
    #    class count + x offset]
    #
    class_count = 2
    offset_dim = 1
    instance_label_dim = 1
    label = np.zeros((y_anchors, x_anchors, class_count + offset_dim + instance_label_dim), dtype=np.float32)
    acc_count = np.zeros((y_anchors, x_anchors, class_count + offset_dim), dtype=np.float32)
    class_idx = 0
    x_offset_idx = class_count
    instance_label_idx = class_count + offset_dim

    # init values
    label[:, :, class_idx:class_idx + class_count] = class_list['background']
    label[:, :, x_offset_idx] = 0.0001

    # transform "h_samples" & "lanes" to desired format
    anchor_scale_x = float(x_anchors) / float(groundSize[1])
    anchor_scale_y = float(y_anchors) / float(groundSize[0])


    # bool to indicate if any lane has been added. if no lane has been added, return none
    lane_added = False
    # calculate anchor offsets
    for laneIdx in range(min(len(label_lanes), max_lane_count)):
        lane_data = label_lanes[laneIdx]

        prev_gx = None
        prev_gy = None
        prev_ax = None
        prev_ay = None
        for idx in range(len(lane_data)):
            dy = label_h_samples[idx]
            dx = lane_data[idx]

            if dx < 0:
                continue

            # do perspective transform at dx, dy
            gx, gy, gz = np.matmul(H, [[dx], [dy], [1.0]])
            if gz == 0:
                continue

            # conver to anchor coordinate(grid)
            gx = int(gx / gz)
            gy = int(gy / gz)
            resized = _resize_transformed_labels((gx, gy),
                                                 (width, height),
                                                 (400, 400),
                                                 (net_input_img_size[0], net_input_img_size[1]),
                                                 cutoffs=cutoffs)
            if resized is None:
                continue
            gx, gy = resized
            cv2.circle(gImg, (int(gx), int(gy)), 2, (0, 255, 0), -1)
            if gx < 0 or gy < 0 or gx >= (groundSize[1] - 1) or gy >= (groundSize[0] - 1):
                continue

            ax = int(gx * anchor_scale_x)
            ay = int(gy * anchor_scale_y)

            if ax < 0 or ay < 0 or ax >= (x_anchors - 1) or ay >= (y_anchors - 1):
                continue

            instance_label_value = (laneIdx + 1.0) * 50
            label[ay][ax][class_idx:class_idx + class_count] = class_list['lane_marking']

            # do line interpolation to padding label data for perspectived coordinate.
            if prev_gx is None:
                prev_gx = gx
                prev_gy = gy
                prev_ax = ax
                prev_ay = ay
            else:
                if abs(ay - prev_ay) <= 1:
                    if acc_count[ay][ax][x_offset_idx] > 0:
                        continue
                    offset = gx - (ax / anchor_scale_x)
                    label[ay][ax][x_offset_idx] += math.log(offset + 0.0001)
                    label[ay][ax][instance_label_idx] = instance_label_value
                    acc_count[ay][ax][x_offset_idx] = 1
                    lane_added = True
                else:
                    gA = np.array([prev_gx, prev_gy])
                    gB = np.array([gx, gy])
                    gLen = float(np.linalg.norm(gA - gB))

                    gV = (gA - gB) / gLen

                    inter_len = min(max(int(abs(prev_gy - gy)), 1), 10)
                    for dy in range(inter_len):
                        gC = gB + gV * (float(dy) / float(inter_len)) * gLen

                        ax = np.int32(gC[0] * anchor_scale_x)
                        ay = np.int32(gC[1] * anchor_scale_y)

                        if acc_count[ay][ax][x_offset_idx] > 0:
                            continue

                        offset = gC[0] - (ax / anchor_scale_x)
                        label[ay][ax][x_offset_idx] += math.log(offset + 0.0001)
                        label[ay][ax][class_idx:class_idx + class_count] = class_list['lane_marking']
                        label[ay][ax][instance_label_idx] = instance_label_value
                        acc_count[ay][ax][x_offset_idx] = 1
                        lane_added = True

                prev_gx = gx
                prev_gy = gy
                prev_ax = ax
                prev_ay = ay
    # global imagecnt
    # with open(f"images/outpt_imgs/{imagecnt:03d}.jpg", "wb") as f:
    #     f.write(cv2.imencode('.jpg', gImg)[1])
    # imagecnt += 1
    if not lane_added:
        return None, None
    return imgf, label


def _resize_transformed_labels(label, previous_shape, cutoff_shape, resize_shape, cutoffs):
    c_width, c_height  = cutoff_shape[:2]
    r_width, r_height  = resize_shape[:2]
    p_width, p_height = previous_shape[:2]
    x_l_cutoff, x_r_cutoff, y_u_cutoff, y_d_cutoff = cutoffs

    x_l_cutoff = x_l_cutoff if x_l_cutoff >= 0 else p_width + x_l_cutoff
    x_r_cutoff = x_r_cutoff if x_r_cutoff > 0 else p_width + x_r_cutoff
    y_u_cutoff = y_u_cutoff if y_u_cutoff >= 0 else p_height + y_u_cutoff
    y_d_cutoff = y_d_cutoff if y_d_cutoff > 0 else p_height + y_d_cutoff

    if x_l_cutoff < label[0] < x_r_cutoff and y_u_cutoff < label[1] < y_d_cutoff:
        label = (label[0] - x_l_cutoff, label[1] - y_u_cutoff)
    else:
        return None

    label = (label[0] * r_width // c_width ,
             label[1] * r_height // c_height )
    return label