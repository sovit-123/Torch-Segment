import glob
import albumentations
import cv2
import numpy as np
import torch

from .utils.helpers import get_label_mask
from .utils.helpers import set_class_values
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def get_images(root_path):
    train_images = glob.glob(f"{root_path}/train/*")
    train_images.sort()
    train_segs = glob.glob(f"{root_path}/train_labels/*")
    train_segs.sort()
    valid_images = glob.glob(f"{root_path}/val/*")
    valid_images.sort()
    valid_segs = glob.glob(f"{root_path}/val_labels/*")
    valid_segs.sort()

    return train_images, train_segs, valid_images, valid_segs

class CamVidDataset(Dataset):

    def __init__(self, path_images, path_segs, image_transform, mask_transform, label_colors_list, classes_to_train):
        print(f"TRAINING ON CLASSES: {classes_to_train}")

        self.path_images = path_images
        self.path_segs = path_segs
        self.label_colors_list = label_colors_list
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.classes_to_train = classes_to_train
        # convert str names to class values on masks
        self.class_values = set_class_values(self.classes_to_train)
        
    def __len__(self):
        return len(self.path_images)
        
    def __getitem__(self, index):
        image = np.array(Image.open(self.path_images[index]).convert('RGB'))
        mask = np.array(Image.open(self.path_segs[index]).convert('RGB'))
          
        ##### THIS (READING WITH OPENCV) WORKS TOO #####
        # image = cv2.imread(self.path_images[index], cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # mask = cv2.imread(self.path_segs[index], cv2.IMREAD_COLOR)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        #############################################               
        
        image = self.image_transform(image=image)['image']
        mask = self.mask_transform(image=mask)['image']
        
        # get the colored mask labels
        mask = get_label_mask(mask, self.class_values)
       
        image = np.transpose(image, (2, 0, 1))
        
        image = torch.tensor(image, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.long) 

        return image, mask

class Transforms():
    def train_image_transforms(self):
        train_image_transform = albumentations.Compose([
            albumentations.Resize(224, 224, always_apply=True),
            albumentations.Normalize(
                    mean=[0.45734706, 0.43338275, 0.40058118],
                    std=[0.23965294, 0.23532275, 0.2398498],
                    always_apply=True)
        ])
        print(train_image_transform)
        return train_image_transform

    def valid_image_transforms(self):
        valid_image_transform = albumentations.Compose([
            albumentations.Resize(224, 224, always_apply=True),
            albumentations.Normalize(
                    mean=[0.45734706, 0.43338275, 0.40058118],
                    std=[0.23965294, 0.23532275, 0.2398498],
                    always_apply=True)
        ])
        return valid_image_transform

    def train_target_transroms(self):
        train_mask_transform = albumentations.Compose([
            albumentations.Resize(224, 224, always_apply=True),
        ])
        return train_mask_transform

    def valid_target_transforms(self):
        valid_mask_transform = albumentations.Compose([
            albumentations.Resize(224, 224, always_apply=True),
        ])
        return valid_mask_transform

transforms = Transforms()
        
def get_dataset(train_images, train_segs, 
                label_colors_list, valid_images, 
                valid_segs, classes_to_train, 
                user_train_image_transform=transforms.train_image_transforms(), 
                user_train_mask_transform=transforms.train_target_transroms(), 
                user_valid_image_transform=transforms.valid_image_transforms(), 
                user_valid_mask_transform=transforms.valid_target_transforms()):
    train_dataset = CamVidDataset(train_images, train_segs, user_train_image_transform, 
                                user_train_mask_transform,
                                label_colors_list, 
                                classes_to_train)
    valid_dataset = CamVidDataset(valid_images, valid_segs, user_valid_image_transform,
                                user_valid_mask_transform,
                                label_colors_list, 
                                classes_to_train)

    return train_dataset, valid_dataset

def get_data_loaders(train_dataset, valid_dataset, batch_size):
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size)

    return train_data_loader, valid_data_loader