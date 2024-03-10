import pandas as pd
import glob
import os
import math
import xml.etree.ElementTree as ET

# Convert Pascal_Voc bb to Yolo
def pascal_voc_to_yolo(x1, y1, x2, y2, image_w, image_h):
    x_center_norm = ((x2 + x1)/(2*image_w))
    y_center_norm = ((y2 + y1)/(2*image_h))
    width_norm = (x2 - x1)/image_w
    height_norm = (y2 - y1)/image_h
    return x_center_norm,  y_center_norm, width_norm, height_norm


def xml_to_csv(path):
    xml_list = []
    column_name = ['filename', 'width', 'height']
    df = pd.DataFrame(columns=column_name)
    for xml_file in glob.glob(path + '/*.xml'):
        xml_label = []
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('.//size')
        width = int(size.find('.//width').text)
        height = int(size.find('.//height').text)
        for object in root.findall('.//object'):
        #object = xroot.find(".//object")
            label = object.find(".//name").text
            bndbox = object.find(".//bndbox")
            xmin = bndbox.find(".//xmin").text
            ymin = bndbox.find(".//ymin").text
            xmax = bndbox.find(".//xmax").text
            ymax = bndbox.find(".//ymax").text

            x_center_norm,  y_center_norm, width_norm, height_norm = pascal_voc_to_yolo(float(xmin), float(ymin), float(xmax), float(ymax), float(width), float(height))

            value_name = root.find('.//filename').text
            name = value_name.replace('.jpg','')
            value_df = (value_name,
                    width,
                    height)
            value_label = (label +' '+ str(x_center_norm)  +' '+ str(y_center_norm)  +' '+ str(width_norm)  +' '+ str(height_norm))
            #print('value_label:',value_label)

            xml_list.append(value_df)
            if xml_label ==[]:
                xml_label = value_label
            else:
                xml_label = xml_label+ os.linesep + value_label
        folder = path.split('/')[-3].replace('VOC2012_','')
        file = open(f'/content/labels/{folder}/{name}.txt', 'w')
        file.write(xml_label)
        file.close()
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    df = pd.concat([df, xml_df], axis = 0)
    return df


def img_label(path):
    list_names = []
    column_name_assos = ['image', 'label']
    df_assos = pd.DataFrame(columns=column_name_assos)
    for image in glob.glob(path + '/*.jpg'):
        image = image.split('/')[-1]
        label_path = image.replace('jpg', 'txt')
        value = (image, label_path)
        list_names.append(value)
    df_img_label = pd.DataFrame(list_names, columns=column_name_assos)
    return df_img_label

def prepare_data():
    datasets = ['train_val', 'test']
    column_name = ['filename', 'width', 'height']
    for ds in datasets:
        ann_path = os.path.join(os.getcwd(), 'VOC2012_' + ds,'VOC2012_' + ds, 'Annotations')
        xml_df = xml_to_csv(ann_path)

        image_path = os.path.join(os.getcwd(), 'VOC2012_' + ds, 'VOC2012_' + ds, 'JPEGImages')
        df_img_lbl = img_label(image_path)
        df_img_lbl.to_csv('{}.csv'.format(ds), index=None)
        print('Successfully converted xml to csv and create img_label.csv for train and test.')