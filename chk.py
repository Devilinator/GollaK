from load_data import load_data_from_npz, load_batch, load_data_names, load_batch_from_names_random, load_batch_from_names
import numpy as np
import cv2
import os
import glob
from os.path import join
import json

train_path = "E:\gig-dl.ir\Dataset/SETrain"
val_path = "E:\gig-dl.ir\Dataset/SEValidation"
test_path = "E:\gig-dl.ir\Dataset/SETest"
dataset_path = "F:\GazeCapture"

all_path = "E:\gig-dl.ir\Dataset/SEAll"

path = "E:\gig-dl.ir\Dataset"

# batch_size = 2000000
        
img_cols = 64
img_rows = 64
img_ch = 3
    
train_names = load_data_names(train_path)
# validation data
val_names = load_data_names(val_path)
# test data
test_names = load_data_names(test_path)

names = load_data_names(all_path)
        
#x, y = load_batch([l[0:batch_size] for l in train_data], img_ch, img_cols, img_rows)
# x, y = load_batch_from_names(train_names[0:batch_size], dataset_path, img_ch, img_cols, img_rows)

# print(train_names[0:batch_size])


for i, img_name in enumerate(names):

    # directory
    dir = img_name[:5]

    # frame name
    frame = img_name[6:]

    # index of the frame inside the sequence
    idx = int(frame[:-4])

    # open json files
    face_file = open(join(path, dir, "appleFace.json"))
    left_file = open(join(path, dir, "appleLeftEye.json"))
    right_file = open(join(path, dir, "appleRightEye.json"))
    dot_file = open(join(path, dir, "dotInfo.json"))
    grid_file = open(join(path, dir, "faceGrid.json"))

    # load json content
    face_json = json.load(face_file)
    left_json = json.load(left_file)
    right_json = json.load(right_file)
    dot_json = json.load(dot_file)
    grid_json = json.load(grid_file)
    
    print("No ERR: {}".format(join(path, dir, "frames", frame)))
    
    if int(dir is None):
            print("Error with coordinates: {}".format(join(path, dir, "frames", frame)))
            # continue
    
    # print(frame)
    if int(frame is None):
            print("Error with coordinates: {}".format(join(path, dir, "frames", frame)))
            # continue
    
    if int(face_json["X"][idx]) is None or int(face_json["Y"][idx]) is None or \
            int(left_json["X"][idx]) is None or int(left_json["Y"][idx]) is None or \
            int(right_json["X"][idx]) is None or int(right_json["Y"][idx]) is None:
            print("Error with coordinates: {}".format(join(path, dir, "frames", frame)))
            # continue
        
    # if int(dot_json["DotNum"][idx]) is None or int(dot_json["XCam"][idx]) is None or int(dot_json["YCam"][idx]) is None or \
    #     int(grid_json["X"][idx]) is None or int(grid_json["Y"][idx]) is None :
    #     print("Error with coordinates: {}".format(join(path, dir, "frames", frame)))
    #     continue

    # open image
    img = cv2.imread(join(path, dir, "frames", frame))

    # debug stuff
    if img is None:
        print("Error opening image: {}".format(join(path, dir, "frames", frame)))
        # continue
    
    # print("No Error: {}".format(join(path, dir, "frames", frame)))