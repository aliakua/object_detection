import tqdm
import torch, cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os

from PIL import Image
from torchvision.ops import box_iou
import subprocess

import warnings
warnings.filterwarnings( 'ignore' )

import glob 
IMAGE_PATHS = np.loadtxt('/kaggle/input/cub2002011/CUB_200_2011/images.txt', dtype=str) # all image paths
BBOXES_FILE_PATH = "/kaggle/input/cub2002011/CUB_200_2011/bounding_boxes.txt"
BBOXES = np.loadtxt(BBOXES_FILE_PATH, dtype=int, delimiter=" ", ndmin=2)[:, 1:]
TRAIN_TEST_PATH  = np.loadtxt('/kaggle/input/cub2002011/CUB_200_2011/train_test_split.txt', dtype=int)
CSV_FILE = '/kaggle/working/data_anno.csv'
IMG_DIR_FULL = '/kaggle/input/cub2002011/CUB_200_2011/images'
IMG_DIR = '/kaggle/working/images_30'

def create_dict(img_dir):
        label_dict = {}
        value = 1
        for folder in glob.glob(img_dir + '/*'):
            key = folder.split('/')[-1].split('.')[-1]
            label = int(folder.split('/')[-1].split('.')[-2])
            if label<31 :
                label_dict.update({key: label})
        return label_dict

def data_to_yolo(x1, y1, w, h, image_w, image_h):
    return [((2*x1 + w)/(2*image_w)) , ((2*y1 + h)/(2*image_h)), w/image_w, h/image_h]

def img_label(img_dir , img_paths, bboxes, train_test_path):
    list_names = []
    column_name_assos = ['ind', 'label','x','y','w','h','image','train_flag']
    df_assos = pd.DataFrame(columns=column_name_assos)
    for row in img_paths:
        image_open = Image.open(img_dir + '/' +row[1]) # from full to 30
        ind = int(row[0])
        label = row[1].split('/')[-2].split('.')[-1]
        prelabel = int(row[1].split('/')[-2].split('.')[-2])
        image = row[1]
        image_w = image_open.size[0]
        image_h = image_open.size[1]
        x = bboxes[ind-1][0]
        y = bboxes[ind-1][1]
        w = bboxes[ind-1][2]
        h = bboxes[ind-1][3]
        x_center_norm, y_center_norm, width_norm, height_norm = data_to_yolo(x, y, w, h, image_w, image_h)
        train_flag = train_test_path[ind-1][1]
        if prelabel<31: # dataset has 200 classes, which is too much for detection, and ok for GAN - will decrease till 30
            value = (ind, label, x_center_norm, y_center_norm, width_norm, height_norm , image, train_flag)
            list_names.append(value)
    df_img_label = pd.DataFrame(list_names, columns=column_name_assos)
    df_img_label.to_csv('data_anno.csv', index=None)
    print('Successfully create data_anno.csv.')
    return df_img_label

def create_data_folder(image_paths, img_dir_old, img_dir_new, bboxes, train_test_path):
    df = img_label(img_dir_old, image_paths, bboxes, train_test_path)
    os.makedirs(img_dir_new)
    annotations = pd.read_csv('/kaggle/working/data_anno.csv')
    names = set(annotations['label'])
    names_folder =[]
    for row in image_paths:
        image_open = Image.open(img_dir_old + '/' +row[1])
        ind = int(row[0])
        label = row[1].split('/')[-2]
        prelabel = int(row[1].split('/')[-2].split('.')[-2])
        if prelabel <31:
            names_folder.append(f'{label}')
    names_folder = set(names_folder)

    FOLDER_FORMAT = "{dir}/{name_folder}"
    for name in names_folder:
        os.makedirs(FOLDER_FORMAT.format(dir = img_dir_new, name_folder = name))

    for name in names_folder:
        for dirname, _, filenames in os.walk(img_dir_old+name):
            for filename in filenames:
                inp = FOLDER_FORMAT.format(dir = img_dir_old, name_folder = name) +'/'+ filename
                out = FOLDER_FORMAT.format(dir = img_dir_new, name_folder = name) +'/'
                subprocess.call(f'cp {inp} {out}', shell=True)
    print('Successfully was created data_anno.csv with annotations AND images_30 folder with images')