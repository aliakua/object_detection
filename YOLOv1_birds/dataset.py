import torch
import pandas as pd
import os
from PIL import Image
import numpy as np
import glob

class BirdDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, S=7, B=2, C=30, rescale_size = 448, transform=None,flag_aug = 0
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dict = self._create_dict(self.img_dir)
        self.transform = transform
        self.flag_aug = flag_aug
        self.rescale_size = rescale_size
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        box = []
        class_label = self.label_dict[self.annotations.iloc[index,1]]
        x = self.annotations.iloc[index,2]
        y = self.annotations.iloc[index,3]
        width = self.annotations.iloc[index,4]
        height = self.annotations.iloc[index,5]
        box.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, str(self.annotations.iloc[index, 6]))
        image = Image.open(img_path).convert("RGB")

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5* self.B )) #* self.B)) ## S*S*25
        #print(f'label_matrix.shape:{label_matrix.shape}')
        for box in box:
            #print(f'class_label:{class_label}')
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
            if label_matrix[i, j, 30] == 0:
                # Set that there exists an object
                label_matrix[i, j, 30] = 1

                # Box coordinates
                
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 31:35] = box_coordinates

                # Set one hot encoding for class_label
                #print(f'label_matrix[i, j, class_label]:{label_matrix[i, j, class_label]}')
                label_matrix[i, j, class_label] = 1

        box = torch.tensor(box)
        image = self._prepare_sample(image) # PIL-> np
        image = np.array(image / 255, dtype='float32') # np # standartization
       #print(image)
       # print(type(image))
        if self.flag_aug == 0:
            # image = self.transform(image)
            image, box = self.transform(image, box)
        else:
            aug_image = self.transform(image=image)
            image = aug_image['image']

        return image, label_matrix
    
    def _create_dict(self, img_dir):
        label_dict = {}
        for folder in glob.glob(img_dir + '/*'):
            key = folder.split('/')[-1].split('.')[-1]
            label = int(folder.split('/')[-1].split('.')[-2])
            if label<31 :
                label_dict.update({key: label})
        return label_dict
    
    def _prepare_sample(self, image):
        image = image.resize((self.rescale_size, self.rescale_size))
        return np.array(image)