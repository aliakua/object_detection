"""
Main file for training Yolo model on Bird dataset

"""
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import glob
import os
from model import Yolov1
from dataset import BirdDataset
from get_data import create_data_folder
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
    create_dict,
    get_key,
    inference
)
from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc.

IMAGE_PATHS = np.loadtxt('/kaggle/input/cub2002011/CUB_200_2011/images.txt', dtype=str) # all image paths
BBOXES_FILE_PATH = "/kaggle/input/cub2002011/CUB_200_2011/bounding_boxes.txt"
BBOXES = np.loadtxt(BBOXES_FILE_PATH, dtype=int, delimiter=" ", ndmin=2)[:, 1:]
TRAIN_TEST_PATH  = np.loadtxt('/kaggle/input/cub2002011/CUB_200_2011/train_test_split.txt', dtype=int)
IMG_DIR_FULL = '/kaggle/input/cub2002011/CUB_200_2011/images'

LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 64 # 64 in original paper but I don't have that much vram
WEIGHT_DECAY = 0
EPOCHS = 25
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "/kaggle/working/overfit_2.pth.tar"
CSV_FILE = '/kaggle/working/data_anno.csv'
IMG_DIR = '/kaggle/working/images_30'

import albumentations as A
from albumentations.pytorch import ToTensorV2

create_data_folder(image_paths = IMAGE_PATHS, 
                   img_dir_old = IMG_DIR_FULL, 
                   img_dir_new = IMG_DIR, 
                   bboxes = BBOXES, 
                   train_test_path = TRAIN_TEST_PATH)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bbox):
        for t in self.transforms:
            img, bbox = t(img), bbox

        return img, bbox


transform = Compose([#transforms.Resize((448, 448)), 
                     transforms.ToTensor(),
])

transform_aug1 = A.Compose([A.ColorJitter( brightness=0.1, contrast=0.6, saturation=0.2, hue=0.1),
                            ToTensorV2() # convert the image to PyTorch tensor
    ])

transform_aug2 = A.Compose([A.Rotate(limit=15, p=1),
                            ToTensorV2() # convert the image to PyTorch tensor
    ])
transform_aug3 = A.Compose([A.Rotate(limit=-15, p=1),
                            ToTensorV2() # convert the image to PyTorch tensor
    ])
transform_aug4 = A.Compose([A.ColorJitter( brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1),
                            ToTensorV2() # convert the image to PyTorch tensor
    ])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=30).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    bird_dict = create_dict(IMG_DIR)

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset1 = BirdDataset(
        CSV_FILE,IMG_DIR,
        transform=transform
    )
    train_dataset_aug1 = BirdDataset(
        CSV_FILE,IMG_DIR,
        transform=transform_aug1,
        flag_aug = 1,
    )
    train_dataset_aug2 = BirdDataset(
        CSV_FILE,IMG_DIR,
        transform=transform_aug2,
        flag_aug = 1,
    )
    train_dataset_aug3 = BirdDataset(
        CSV_FILE,IMG_DIR,
        transform=transform_aug3,
        flag_aug = 1,
    )
    train_dataset_aug4 = BirdDataset(
        CSV_FILE,IMG_DIR,
        transform=transform_aug4,
        flag_aug = 1,
    )
    train_dataset = train_dataset1 + train_dataset_aug1 + train_dataset_aug2 + train_dataset_aug3 + train_dataset_aug4
    
    # получаем рандомные индексы, для разделения на train/val
    train_indices, rem_indices = train_test_split(
        list(range(len(train_dataset))), train_size = 0.9)
    val_indices, test_indices = train_test_split(
        list(range(len(rem_indices))), train_size = 0.5)

    train_loader = DataLoader(
        dataset=torch.utils.data.Subset(train_dataset, train_indices),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=torch.utils.data.Subset(train_dataset, val_indices),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )
    
    test_loader = DataLoader(
        dataset=torch.utils.data.Subset(train_dataset, test_indices),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    if LOAD_MODEL==False:
        for epoch in range(EPOCHS):
            print(f'epoch: {epoch+1}')
            pred_boxes, target_boxes = get_bboxes(
                train_loader, model, iou_threshold=0.5, threshold=0.4
            )

            mean_avg_prec = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
            )
            print(f"Train mAP: {mean_avg_prec}")

            if mean_avg_prec > 0.85:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
                import time
                time.sleep(10)

            train_fn(train_loader, model, optimizer, loss_fn)
    else: 
        #inference 
        model, optimizer = load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
        inference(test_loader, model, bird_dict )


if __name__ == "__main__":
    #train
    main()
    #inference
    main(LOAD_MODEL = True)