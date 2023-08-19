import numpy as np
import cv2
import os
import glob
from os.path import join
import json

def load_data_names(path):

    seq_list = []
    seqs = sorted(glob.glob(join(path, "0*")))

    for seq in seqs:

        file = open(seq, "r")
        content = file.read().splitlines()
        for line in content:
            seq_list.append(line)

    return seq_list
    
train_path = "F:\GazeCapture/train"
names = load_data_names(train_path)

for i in range(len(names)):
    # get the lucky one
    img_name = names[i]
    
    # directory
    dir = img_name[:5]
    
    # frame name
    frame = img_name[6:]
    
    print("Checked Directory: {}".format(join(train_path, dir, "frames", frame)))
    
    # index of the frame into a sequence
    idx = int(frame[:-4])