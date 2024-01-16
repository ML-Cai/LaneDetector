# resize "../datasets/original_task_bags 2023..." to 1280, 960
# not only the images but also the annotations

# lets start with resizing the images and by that creating a new folder for the dataset

# use argparse to get the path to the dataset: !!
# Path includes the folder: "images" and "annoations.xml" from cvat (images 1.1)

import argparse
import os
import sys
import cv2
import shutil
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description='Resize images and annotations from a dataset')
parser.add_argument('dataset_path', type=str, help='path to the dataset')
args = parser.parse_args()
path = args.dataset_path

og_image_width = 0
og_image_height = 0

# check if the path is valid
if not os.path.exists(path):
    print("Path does not exist")
    sys.exit()

# check if the path contains the folder "images" and "annotations.xml"
if not os.path.exists(os.path.join(path, "images")):
    print("Path does not contain the folder 'images'")
    sys.exit()
if not os.path.exists(os.path.join(path, "annotations.xml")):
    print("Path does not contain the file 'annotations.xml'")
    sys.exit()

# create a new folder for the resized dataset, put it in the same folder as the original path
old_path_split = os.path.split(path)
new_path = os.path.join(old_path_split[0], "1280x960_" + old_path_split[1])

# create the new path
# os.mkdir(new_path)

# resize recursively the images in all their subfolders and store them in the new path but also in subfolders
# with the same name as the original subfolders
# first copy everything to the new path and then resize overwrite the images
if not os.path.exists(new_path):
    shutil.copytree(path, new_path, dirs_exist_ok=True)

    # resize the images from all subfolders and overwrite them
    # go into every subfolder of the new path

    for root, dirs, files in os.walk(new_path):
        # go through all files in the subfolder
        for file in files:
            # check if the file is an image
            if file.endswith(".png") or file.endswith(".jpg"):
                # resize the image
                image = cv2.imread(os.path.join(root, file))
                if og_image_width == 0:
                    og_image_width = image.shape[1]
                    og_image_height = image.shape[0]
                    print("Height: " + str(og_image_height))
                    print("Width: " + str(og_image_width))
                resized_image = cv2.resize(image, (1280, 960))
                # overwrite the image
                cv2.imwrite(os.path.join(root, file), resized_image)
                print("imwrite")
else:
    og_image_width = 2064
    og_image_height = 1544
    # delete current annotations.xml and copy it again
    os.remove(os.path.join(new_path, "annotations.xml"))
    shutil.copyfile(os.path.join(path, "annotations.xml"), os.path.join(new_path, "annotations.xml"))



# resize the annotations.xml file under path/annotations.xml

# under <annotations> there are <image> tags
# under <image> tags there are <polyline> tags
# under <polyline> tags there are points="x1,y1;x2,y2;..." attributes
# the points are the coordinates of the lane lines
# the coordinates are relative to the image size
# therefore now are being resized

print("test")
# parse the xml file
tree = ET.parse(os.path.join(new_path, "annotations.xml"))
root = tree.getroot()

print("Find all")
# go through all <image> tags
for image in root.findall("image"):
    print("Annotation downsizing")
    # change witdh and height
    image.set("width", "1280")
    image.set("height", "960")
    # go through all <polyline> tags
    for polyline in image.findall("polyline"):
        # get the points attribute
        points = polyline.get("points")
        # split the points attribute into a list of points
        points_list = points.split(";")
        # go through all points
        for i in range(len(points_list)):
            # split the point into x and y coordinate
            x, y = points_list[i].split(",")
            # convert the coordinates to int
            x = int(float(x))
            y = int(float(y))
            # resize the coordinates
            x = int(x * 1280 / og_image_width)
            y = int(y * 960 / og_image_height)
            # convert the coordinates back to string
            x = str(x)
            y = str(y)
            # put the coordinates back together
            points_list[i] = x + "," + y
        # put the points back together
        points = ";".join(points_list)
        # overwrite the points attribute
        polyline.set("points", points)

        # write changes
        tree.write(os.path.join(new_path, "annotations.xml"))



        # display the images and the annotations until the user wants to end
        # get the image path
        # image_path = image.get("name")
        # # get the image
        # image_to_show = cv2.imread(os.path.join(new_path+"/images", image_path))
        # # draw the points as green circles
        # for point in points_list:
        #     x, y = point.split(",")
        #     x = int(x)
        #     y = int(y)
        #     cv2.circle(image_to_show, (x, y), 5, (0, 255, 0), -1)
        # print("New path: " + new_path)
        # print("Showing image: " + image_path)
        # # show the image in a window
        # cv2.imshow("image", image_to_show)
        # # wait for the user to press a key
        # cv2.waitKey(0)
        # # destroy the window
        # cv2.destroyAllWindows()


# write the changes to the xml file
tree.write(os.path.join(new_path, "annotations.xml"))








