import pandas as pd
import numpy as np 
import glob
import os
import math
import xml.etree.ElementTree as ET
import subprocess

# Convert Pascal_Voc bb to Yolo
def pascal_voc_to_yolo(x1, y1, x2, y2, image_w, image_h):
    x_center_norm = ((x2 + x1)/(2*image_w))
    y_center_norm = ((y2 + y1)/(2*image_h))
    width_norm = (x2 - x1)/image_w
    height_norm = (y2 - y1)/image_h
    return x_center_norm,  y_center_norm, width_norm, height_norm


def xml_to_csv(path):
    xml_list = []
    column_name = ['image_name', 'label']
    df = pd.DataFrame(columns=column_name)
    dict_label = {'aeroplane':0
                  ,'bicycle':1
                  ,'bird':2
                  ,'boat':3
                  ,'bottle':4
                  ,'bus':5
                  ,'car':6
                  ,'cat':7
                  ,'chair':8
                  ,'cow':9
                  ,'diningtable':10
                  ,'dog':11
                  ,'horse':12
                  ,'motorbike':13
                  ,'person':14
                  ,'pottedplant':15
                  ,'sheep':16
                  ,'sofa':17
                  ,'train':18
                  ,'tvmonitor':19}
    for xml_file in glob.glob(path + '/*.xml'):
        xml_label = []
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('.//size')
        width = int(size.find('.//width').text)
        height = int(size.find('.//height').text)
        for object in root.findall('.//object'):
            label = object.find(".//name").text
            bndbox = object.find(".//bndbox")
            xmin = bndbox.find(".//xmin").text
            ymin = bndbox.find(".//ymin").text
            xmax = bndbox.find(".//xmax").text
            ymax = bndbox.find(".//ymax").text
            x_center_norm,  y_center_norm, width_norm, height_norm = pascal_voc_to_yolo(float(xmin), float(ymin), float(xmax), float(ymax), float(width), float(height))

            image_name = root.find('.//filename').text
            name = image_name.replace('.jpg','')
            value_df = (image_name,
                        image_name.replace('.jpg','.txt'))
            value_label = (str(dict_label[label]) +' '+ str(x_center_norm)  +' '+ str(y_center_norm)  +' '+ str(width_norm)  +' '+ str(height_norm))
            
            xml_list.append(value_df)
            if xml_label ==[]:
                xml_label = value_label
            else:
                xml_label = xml_label+ os.linesep + value_label
        folder = path.split('/')[-3].replace('VOC2012_','')
        file = open(f'/kaggle/working/labels/{folder}/{name}.txt', 'w')
        file.write(xml_label)
        file.close()
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    df = pd.concat([df, xml_df], axis = 0)
    return df


def img_transfer(path, sample):
    for image in glob.glob(path + '/*.jpg'):
        inp = image
        out = '/kaggle/working/images/'+ sample + '/'+ image.split('/')[-1]
        subprocess.call(f'cp {inp} {out}', shell=True)
    print(f'Images of {sample} successfully were copied in working directory')

def prepare_data(folder_path= '/kaggle/input/pascal-voc-2012-dataset'):
    datasets = ['train_val', 'test']
    column_name = ['filename', 'width', 'height']
    for ds in datasets:
        ann_path = os.path.join(folder_path , 'VOC2012_' + ds,'VOC2012_' + ds, 'Annotations')
        img_txt_csv = xml_to_csv(ann_path)

        image_path = os.path.join(folder_path , 'VOC2012_' + ds, 'VOC2012_' + ds, 'JPEGImages')
        img_transfer(image_path , ds)
        img_txt_csv.to_csv('{}.csv'.format(ds), index=None)
        print(f'Successfully converted xml to csv and was created  {ds}.csv.')