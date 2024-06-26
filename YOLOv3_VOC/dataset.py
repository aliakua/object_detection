"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""
import torch
import os

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import Image
from utils import iou_width_height

class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        C=20,
        transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        # apply augmentations with albumentations 
        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
       
        # Building the targets below:
        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]  #6  = 4box coordicates +5th objectness score +6th class - 3x13x13x6, 3x26x26x6, 3x52x52x6
        for box in bboxes:
            iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors) #relative width and height VS anchor w and h
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                #print(scale_idx)
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S_ = self.S[scale_idx]
                i, j = int(S_ * y), int(S_ * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S_ * x - j, S_ * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S_,
                        height * S_,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction
        #target_tensors = torch.Tensor(targets)
#         print(f'len(targets): {len(targets)}')
#         print(f'len(targets[0]): {len(targets[0])}')#for 13x13
#         print(f'len(targets[0][0]): {len(targets[0][0])}')
#         print(f'len(targets[0][0][0]): {len(targets[0][0][0])}')
        
#         print(f'len(targets[1]): {len(targets[1])}') #for 26x26
#         print(f'len(targets[1][0]): {len(targets[1][0])}')
#         print(f'len(targets[1][0][0]): {len(targets[1][0][0])}')
                
        return image, tuple(targets)
