"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import pandas as pd
from PIL import Image


class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        self.dict_label = {'aeroplane':0
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

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if (('.') in x or (',') in x) else int(self.dict_label[x])
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5* self.B )) #* self.B)) ## S*S*25
        for box in boxes:
            class_label, x, y, width, height = box #.tolist()
            #class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)  # INT - is defined like ceil !!
            #print(f'i:{i},j:{j}')
            x_cell, y_cell = self.S * x - j, self.S * y - i

            width_cell, height_cell = (
                width * self.S, # should be = it is label
                height * self.S, # should be = it is label
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, 20] == 0:
                # Set that there exists an object
                label_matrix[i, j, 20] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates

                # Set one hot encoding for class_label
                #print(f'label_matrix[i, j, class_label]:{label_matrix[i, j, class_label]}')
                label_matrix[i, j, class_label] = 1

        boxes = torch.tensor(boxes)

        if self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)

        return image, label_matrix