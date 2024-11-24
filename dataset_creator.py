
"""
Module for Generating Preprocessed dataset files



"""


import numpy as np
import os

import torch
from torch.utils.data import Dataset
from PIL import Image

from typing import Tuple


import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter


def load_images_from_folder(folder, image_size_x, image_size_y):


   #Define paths to normal and pneumonia
   normal =os.path.join(path, "chest_xray", folder, "NORMAL")
   pneumonia = os.path.join(path, "chest_xray", folder, "PNEUMONIA")


#For both normal and pneumonia, get images and label them
   x, y = [], []
   for img in os.listdir(normal):
       img_path = os.path.join(normal, img)
       try:
         img = Image.open(img_path)
         img = Image.open(img_path).convert("L") #Convert to grayscale
         img = img.resize((image_size_x, image_size_y))
         img_array = np.array(img) / 255.0 #Scale the image
         x.append(img_array)
         y.append(0)
       except Exception as e:
         print(f"Error loading image {img_path}: {e}")


   for img in os.listdir(pneumonia):
     img_path = os.path.join(pneumonia,img)
     try:
       img = Image.open(img_path)
       img = Image.open(img_path).convert("L") #Convert to grayscale
       img = img.resize((image_size_x, image_size_y))
       img_array = np.array(img) / 255.0 #Scale the image
       x.append(img_array)
       y.append(1)
     except Exception as e:
       print(f"Error loading image {img_path}: {e}")


   x = np.array(x)
   y = np.array(y)
   return x, y





class AugmentedBalancedDataset(Dataset):
    def __init__(self, images, labels, target_class=None, samples_per_class=None, transform=None, end_width = 256, end_height = 256):
        """
        Args:
            images: List of images or image paths
            labels: List of labels
            target_class: The class to augment (typically minority class)
            samples_per_class: Target number of samples per class
        """
        self.images = torch.Tensor(images)
        self.labels = torch.LongTensor(labels)
        self.target_class = target_class
        
        # Add channel dimension if not present
        if len(self.images.shape) == 3:  # (n, height, width)
            self.images = self.images.unsqueeze(1)  # (n, 1, height, width)
        
        # Count samples per class
        class_counts = Counter(labels)
        majority_count = max(class_counts.values())
        self.samples_per_class = samples_per_class or majority_count
        
        # Calculate augmentations needed
        self.augmentations_needed = (self.samples_per_class - class_counts[target_class] 
                                   if target_class is not None else 0)
        
        # Get indices of target class samples
        self.target_indices = [i for i, label in enumerate(labels) 
                             if label == target_class]
        
        # Define augmentation pipeline
        self.transform = A.Compose([
        # 1. Random rotation by 30 degrees
        A.Rotate(limit=30, p=0.5),
        
        # 2. Random zoom (scale) by 20%
        A.RandomScale(scale_limit=0.2, p=0.5),
        
        # 3. Random horizontal shift by 10%
        A.ShiftScaleRotate(
            shift_limit=0.2,      # Shift by up to 20%
            scale_limit=0,        # No scaling
            rotate_limit=0,       # No rotation
            p=0.5,               # 50% probability
            border_mode=4        # BORDER_REFLECT_101 in OpenCV
        ),
        # 4. Random vertical shift by 10%
       # A.Shift(limit=(0.1, 0.1), p=0.5),
        
        # 5. Random horizontal flip
        A.HorizontalFlip(p=0.5),
    
    
    A.Resize(height=end_height, width=end_width, always_apply=True),
    ToTensorV2()
])
        
        # Basic transform for non-augmented images
        self.basic_transform = A.Compose([
            #A.Normalize(mean=[0], std=[255]),
            A.Resize(height=end_height, width=end_width, always_apply=True),
            ToTensorV2()
        ])
    
    def __len__(self):
        return len(self.images) + self.augmentations_needed
    
    def __getitem__(self, idx):
        if idx < len(self.images):
            # Handle original images
            image = self.images[idx].squeeze().numpy()  # Remove channel dim for albumentation
            label = self.labels[idx]
            
            # Apply appropriate transform
            if self.target_class is not None and label == self.target_class:
                transformed = self.transform(image=image)
            else:
                transformed = self.basic_transform(image=image)
            
            return transformed["image"], label
        
        else:
            # Handle augmented images
            original_idx = self.target_indices[idx % len(self.target_indices)]
            image = self.images[original_idx].squeeze().numpy()
            label = self.labels[original_idx]
            
            transformed = self.transform(image=image)
            return transformed["image"], label




class CustomDataset(Dataset):
   """
   Custom Dataset for loading image data and labels.
   Encodes the y values into a one hot encoding of classes
  
   """
  
   def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
       self.X = torch.FloatTensor(X)



       self.y = torch.LongTensor(y)


       """
       self.y = torch.zeros((y.shape[0], 2))
       self.y[np.arange(len(y)), y] = 1
       assert self.y.shape[1] == 2
       """
      
       torch.LongTensor(y)










       # TODO: Define specific desired tranformations needed
       self.transform = transform


       # Add channel dimension if not present --> NOTE: This function unsqueezes the tensor to be a different shape.
       if len(self.X.shape) == 3:  # (n, height, width)
           self.X = self.X.unsqueeze(1)  # (n, 1, height, width)


   def __len__(self) -> int:
       return len(self.X)


   def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
       image = self.X[idx]
       label = self.y[idx]


       if self.transform:
           image = self.transform(image)
        



       return {'id': idx, 'image': image, 'label': label}


if __name__ == "__main__":
   

    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")



    image = Image.open(os.path.join(path, "chest_xray/train/NORMAL/IM-0115-0001.jpeg"))


    print("Path to dataset files:", path)


    end_height = 128
    scale_factor = end_height / image.size[1]
    end_width = int(image.size[0] * scale_factor)


    x_train, y_train = load_images_from_folder("train", end_width, end_height)
    x_test_full, y_test_full = load_images_from_folder("test", end_width, end_height)
    #x_val, y_val = load_images_from_folder("val", end_width, end_height)



    print(path)
    #train_dataset = CustomDataset(x_train, y_train, target_class= 0, end_width = end_width, end_height = end_height)

    train_dataset = CustomDataset(x_train, y_train)
    

    from sklearn.model_selection import train_test_split
    
    x_val, x_test, y_val, y_test = train_test_split(x_test_full, y_test_full, train_size = 0.6)

    val_dataset = CustomDataset(x_val, y_val)
    test_dataset = CustomDataset(x_test, y_test) 
    #test_dataset2 = CustomDataset(x_test, y_test)

    print(len(train_dataset))
    print(len(test_dataset))
    print(len(val_dataset))
#

    torch.save(train_dataset, "./data/trainset.pt")
    torch.save(val_dataset, "./data/valset.pt")
    torch.save(test_dataset, "./data/testset.pt")
                            
                        
                                
