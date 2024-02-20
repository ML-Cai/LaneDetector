"""cvat_dataset dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import json
from PIL import Image
import numpy as np
import sys
import cv2
import os
import math
import matplotlib.pyplot as plt

cnt_unique = 0


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for cvat_dataset dataset."""

    VERSION = tfds.core.Version('1.4.1')
    RELEASE_NOTES = {
        '1.4.1': 'Updated Matrices',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            # builder=self,

            description="CVat Lane Dataset",

            features=tfds.features.FeaturesDict({
                "image": tfds.features.Tensor(dtype=tf.float32, shape=(None, None, 3), encoding="bytes"),
                "label": tfds.features.Sequence(
                    tfds.features.Tensor(shape=(None, None), dtype=tf.float32, encoding="bytes")),
                # "h_samples": tfds.features.Sequence(tfds.features.Tensor(shape=(None,), dtype=tf.int32)),
                # "augmentation_index": tf.int32,
            }),
            supervised_keys=("image", "label"),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = "/mnt/c/Users/inf21034/source/IMG_ROOTS/1280x960_CVATROOT/"
        return {
            'train': self._generate_examples(os.path.join(path, "train_set"),
                                             'train_set.json',
                                             [0.0, -15.0, 15.0, -30.0, 30.0]),
            'test': self._generate_examples(os.path.join(path, "test_set"), 'test_set.json', None),
        }

    def _generate_examples(self, path, label_data_name, augmentation_deg):
        """Yields examples."""
        # DONE(cvat_dataset): Yields (key, example) tuples from the dataset
        global cnt_unique
        with open(
                "/mnt/c/Users/inf21034/PycharmProjects/edge_tpu_lane_detection/add_ins/cvat_config2.json") as json_file:
            config = json.load(json_file)
        net_input_img_size = config["model_info"]["input_image_size"]
        x_anchors = config["model_info"]["x_anchors"]
        y_anchors = config["model_info"]["y_anchors"]
        max_lane_count = config["model_info"]["max_lane_count"]
        ground_img_size = config["model_info"]["input_image_size"]
        dates = list(config["perspective_info"].keys())
        perspective_infos = {}
        for date in dates:
            H_list, map_x_list, map_y_list = create_map(config, date, augmentation_deg=augmentation_deg)
            H_list = tf.constant(H_list)
            map_x_list = tf.constant(map_x_list)
            map_y_list = tf.constant(map_y_list)
            perspective_infos[date] = (H_list, map_x_list, map_y_list)


        for image_ary, label_lanes, label_h_samples, refIdx, day_of_recording in _data_reader(path, label_data_name,
                                                                            augmentation_deg):
            # Generate a unique key for each example
            key = cnt_unique
            cnt_unique += 1
            H_list, map_x_list, map_y_list = perspective_infos[day_of_recording]
            img, label = _map_projection_data_generator(image_ary,
                                                        label_lanes,
                                                        label_h_samples,
                                                        net_input_img_size,
                                                        x_anchors,
                                                        y_anchors,
                                                        max_lane_count,
                                                        H_list[refIdx],
                                                        map_x_list[refIdx],
                                                        map_y_list[refIdx],
                                                        ground_img_size
                                                        )
            yield key, {
                "image": img,
                "label": label,
            }
        # for f in path.glob('*.jpeg'):
        #   yield 'key', {
        #       'image': f,
        #       'label': 'yes',
        #   }


def _data_reader(dataset_path,
                 label_data_name,
                 augmentation_deg):
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
            image_path = os.path.join(str(dataset_path), raw_label["raw_file"])
            day_of_recording = raw_label["raw_file"][0:10]
            label_lanes = raw_label["lanes"]
            label_h_samples = raw_label["h_samples"]

            # read image
            with Image.open(image_path) as image:
                image_ary = np.asarray(image)

            # enable this for small dataset test
            # if count >=32:
            #     break
            count += 1

            if augmentation_deg is None:
                yield image_ary, label_lanes, label_h_samples, 0, day_of_recording
            else:
                for refIdx in range(len(augmentation_deg)):
                    yield image_ary, label_lanes, label_h_samples, refIdx, day_of_recording


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
    for i in range(len(groundP)):
        groundP[i][0] = groundP[i][0] * gw / w  # ground_scale_width + gw / 2.0
        groundP[i][1] = groundP[i][1] * gh / h  # gh - groundP[i][1] * ground_scale_height

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

        H, _ = cv2.findHomography(np.float32(imgP), np.float32(rotate_groupP))
        _, invH = cv2.invert(H)

        map_x = np.zeros((gh, gw), dtype=np.float32)
        map_y = np.zeros((gh, gw), dtype=np.float32)

        for gy in range(gh):
            for gx in range(gw):
                nx, ny, nz = np.matmul(invH, [[gx], [gy], [1.0]])
                nx /= nz
                ny /= nz
                if 0 <= nx < w and 0 <= ny < h:
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
                                   groundSize):
    # transform image by perspective matrix
    # height, width = src_image.shape[:2]
    # if width != 1280 or height != 720:
    #     src_image = cv2.resize(src_image, (1280, 720))
    src_img_cpy = src_image.copy()
    for l in label_lanes:
        for sz in range(len(label_h_samples)):
            cv2.circle(src_img_cpy, (int(l[sz]), int(label_h_samples[sz])), 2, (0, 255, 0), -1)

    gImg = cv2.remap(src_image, np.array(map_x), np.array(map_y),
                     interpolation=cv2.INTER_NEAREST,
                     borderValue=(125, 125, 125))

    gImg = gImg[100:-1, 30:-30]
    gImg = cv2.resize(gImg, (net_input_img_size[1], net_input_img_size[0]))

    #################################
    height, width = gImg.shape[:2]
    grid_size = 32
    for x in range(0, width, width // grid_size):
        cv2.line(gImg, (x, 0), (x, height), (255, 0, 0), 1)  # Blue lines

    for y in range(0, height, height // grid_size):
        cv2.line(gImg, (0, y), (width, y), (255, 0, 0), 1)  # Blue lines

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
            resized = _resize_transformed_labels((gx, gy), net_input_img_size, (30, -30, 100, -1))
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

                prev_gx = gx
                prev_gy = gy
                prev_ax = ax
                prev_ay = ay

    return imgf, label


def _resize_transformed_labels(label, image_shape, cutoffs):
    height, width = image_shape[:2]
    x_l_cutoff, x_r_cutoff, y_u_cutoff, y_d_cutoff = cutoffs
    if x_l_cutoff:
        if label[0] > x_l_cutoff:
            label = (label[0] - x_l_cutoff, label[1])
        else:
            return None
    if x_r_cutoff:
        if label[0] > x_r_cutoff + width:
            return None
    if y_u_cutoff:
        if label[1] > y_u_cutoff:
            label = (label[0], label[1] - y_u_cutoff)
        else:
            return None
    if y_d_cutoff:
        if label[1] > y_d_cutoff + height:
            return None
    label = (label[0] * width / (width - x_l_cutoff + x_r_cutoff),
             label[1] * height / (height - y_u_cutoff + y_d_cutoff))
    return label