import xml.etree.ElementTree as ET
import os
import glob
import numpy as np
import cv2
import argparse
import json

# Path: scripts/convert_cvat_tusimple.py

"""
TODO: 
    - [ ] Randextrapolation
    - [ ] lane cut-off
"""


class CVATDataset:
    def __init__(self):
        self.images = []
        self.min_y = -1
        self.max_y = 0

    def add_image(self, image):
        self.images.append(image)

    def to_tusimple(self):
        for image in self.images:
            if not image.annotations:
                print("No annotations for image")
                continue
            min_y, max_y = image.get_minmax_y()
            if min_y < self.min_y or self.min_y < 0:
                self.min_y = min_y
            if max_y > self.max_y:
                self.max_y = max_y
        steps = (self.max_y - self.min_y) // 10
        y_samples = np.linspace(self.min_y, self.max_y, steps + 1, dtype=np.int32)
        for image in self.images:
            image.to_tusimple(y_samples)

    def write_to_json(self, out_path):
        with open(out_path, 'w') as f:
            for image in self.images:
                if not image.annotations:
                    continue
                f.write(str(image) + '\n')


class CVATImage:
    def __init__(self, image_id, image_width, image_height, image_path):
        self.image_id = image_id
        self.image_width = image_width
        self.image_height = image_height
        self.image_path = image_path
        self.min_y = -1
        self.max_y = 0
        self.annotations: list[CvatOneLane] = []

    def add_annotation(self, annotation):
        self.annotations.append(annotation)

    def get_minmax_y(self):
        for annotation in self.annotations:
            if annotation.min_y < self.min_y or self.min_y < 0:
                self.min_y = annotation.min_y
            if annotation.max_y > self.max_y:
                self.max_y = annotation.max_y
        # Auf 10 runden; für alle gelabelten Bilder gleich (wie in TUSimple)
        self.max_y = self.max_y + 10 - self.max_y % 10
        self.min_y = self.min_y - self.min_y % 10
        return self.min_y, self.max_y

    def to_tusimple(self, y_samples: np.ndarray):
        self.annotations = sorted(self.annotations, key=lambda x: x.label, reverse=True)
        for annotation in self.annotations:
            print(annotation.label)
            annotation.to_tusimple(y_samples)

    def __str__(self):
        json_img = {
            "lanes": [lane.x_val.tolist() for lane in self.annotations],
            "h_samples": self.annotations[0].y_samples.tolist(),
            "raw_file": self.image_path,
        }
        return json.dumps(json_img)

    def scale_image(self, image, scale):
        width = int(image.shape[1] * scale / 100)
        height = int(image.shape[0] * scale / 100)
        dim = (width, height)
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


class CvatOneLane:
    def __init__(self, label, points_str, image_width, image_height):
        self.label = label  # "name of lane (left_lane, right_lane or similar); currently unused
        self.max_y: int = 0
        self.min_y: int = -1
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.points_str: str = points_str
        self.points: [(int, int)] = []
        self.y_samples = None
        self.x_val: np.ndarray | None = None
        self.extract_points()

    def scale_points(self, scale):
        for k, point in enumerate(self.points):
            point[0] = int(point[0] * scale / self.image_width)
            point[1] = int(point[1] * scale / self.image_height)
            self.points[k] = point

    def add_border_point(self, point):
        x, y = point
        # Distances to each edge
        smallest_distance = 0
        distances = {
            "top": y,
            "bottom": self.image_height - y,
            "left": x,
            "right": self.image_width - x
        }

        # Find the minimum distance and corresponding edge
        closest_edge = min(distances, key=distances.get)
        closest_distance = distances[closest_edge]
        if closest_distance == 0:
            return None
        elif closest_edge == "bottom":
            return x, self.image_height
        elif closest_edge == "left":
            return 0, y
        elif closest_edge == "right":
            return self.image_width, y
        return None

    def extract_points(self):
        points = self.points_str.split(';')
        for point in points:
            x, y = point.split(',')
            if int(float(y)) > self.max_y:
                self.max_y = int(float(y))
            if int(float(y)) < self.min_y or self.min_y < 0:
                self.min_y = int(float(y))
            self.points.append((int(float(x)), int(float(y))))

        border_point = self.add_border_point(self.points[0])
        # # display the orignal points[0] and the new border point on a white image ( 1280 x 720 )
        # image = np.zeros((720, 1280, 3), np.uint8)
        # cv2.circle(image, self.points[0], 5, (0, 255, 0), -1)
        # if border_point:
        #     # color red
        #     cv2.circle(image, border_point, 5, (0, 0, 255), -1)
        # # show the image in a window
        # cv2.imshow("image", image)
        # # wait for the user to press a key
        # cv2.waitKey(0)
        # # destroy the window
        # cv2.destroyAllWindows()

        if border_point:
            self.points = [border_point] + self.points

    def to_tusimple(self, y_samples: np.ndarray):
        self.cut_off_lane()
        x_samples = np.zeros_like(y_samples, dtype=np.int32)
        x_samples = x_samples - 2
        self.y_samples = y_samples
        for k, point in enumerate(self.points):
            # interpolate between points so I have for every y_sample a x_value
            x1, y1 = point

            if k == len(self.points) - 1:
                # round it to clostest ten
                y1 = round(y1 / 10) * 10
                # check how often 10 can get into y1
                how_often = y1 // 10
                if x_samples[how_often]:
                    break
                else:
                    x_samples[how_often] = x1
                    break

            if k == 0:
                if y1 == self.image_height:
                    x_samples[-1] = x1
                else:
                    # round y1 to closest tens
                    y1 = round(y1 / 10) * 10
                    # check how often 10 can get into y1
                    how_often = y1 // 10
                    # how_often = len(y_samples) - how_often - 1
                    x_samples[how_often] = x1

            x2, y2 = self.points[k + 1]

            if not y2 < y1:
                break

            if str(y1)[:-1] == str(y2)[:-1]:
                # skip points becauuse they dont cross a y_sample
                continue
            else:
                # use polyfit
                # y = mx + b
                x_values = [x1, x2]
                y_values = [y1, y2]
                m, b = np.polyfit(y_values, x_values, 1)

                for y in range(y2+1, y1+1):
                    if y % 10 == 0:
                        # calculate x value for y
                        x = m * y + b
                        how_often = y // 10
                        # how_often = len(y_samples) - how_often - 1
                        x_samples[how_often] = x

        self.x_val = x_samples

    def cut_off_lane(self):
        # 1. Finde den Index der Punkte die in y-werten zurückgehen
        # 2. Schneide dann die obere hälfte ab
        smallest_loc_y = self.max_y + 1
        local_points = []
        for k, point in enumerate(self.points):
            if point[1] < smallest_loc_y:
                smallest_loc_y = point[1]
                local_points.append(point)
        self.points = local_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', required=True, help='The path to the CVAT annotation file')
    parser.add_argument('--image_path', required=False, help='The path to the images')
    args = parser.parse_args()
    annotation_path = args.annotation_path
    annotation_folder = os.path.dirname(annotation_path)
    image_path = args.image_path

    tree = ET.parse(annotation_path)
    root = tree.getroot()
    # get all images with annotations
    dataset = CVATDataset()
    # image_list = []
    for image in root.findall('image'):
        image_id = image.attrib['id']
        image_width = image.attrib['width']
        image_height = image.attrib['height']
        image_path = image.attrib['name']
        cvat_image = CVATImage(image_id, image_width, image_height, image_path)
        for polyline in image.findall('polyline'):
            label = polyline.attrib['label']
            points = polyline.attrib['points']
            annotation = CvatOneLane(label, points, image_width, image_height)
            cvat_image.add_annotation(annotation)
        dataset.add_image(cvat_image)
    dataset.to_tusimple()
    dataset.write_to_json(os.path.join(annotation_folder, 'tusimple.json'))


if __name__ == "__main__":
    main()
