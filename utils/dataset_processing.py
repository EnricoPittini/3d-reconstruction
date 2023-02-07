import os
import shutil
import numpy as np
import cv2
import torch
from zipfile import ZipFile


def unzip_dataset(zip_file_path):    
    # loading the temp.zip and creating a zip object
    with ZipFile(zip_file_path, 'r') as zObject:    
        # Extracting all the members of the zip 
        # into a specific location.
        zObject.extractall(path='datasets/nyu_data')


def process_trainDataset_folder(source_dataset_folder_path, step_frames=10):
    os.makedirs(os.path.join(os.getcwd(),source_dataset_folder_path,f'step{step_frames}'), exist_ok=True)

    subfolders_paths = [subfolder_path for subfolder_path in os.listdir(source_dataset_folder_path) 
                        if os.path.isdir(os.path.join(os.getcwd(),source_dataset_folder_path,subfolder_path)) and 'step' not in subfolder_path]

    #i = 0

    for subfolder_path in subfolders_paths:
        #print(subfolder_path)
        #print(subfolder_path)
        subfolder_images_number = len(os.listdir(os.path.join(os.getcwd(),source_dataset_folder_path,subfolder_path)))//2
        #print(subfolder_images_number)
        for image_index in range(1, subfolder_images_number+1, step_frames):
            img_file_from_path = os.path.join(os.getcwd(),source_dataset_folder_path,subfolder_path,f'{image_index}.jpg')
            img_file_to_path = os.path.join(os.getcwd(),source_dataset_folder_path,f'step{step_frames}',f'{subfolder_path}_{image_index}.jpg')
            shutil.copyfile(img_file_from_path,img_file_to_path)

            target_file_from_path = os.path.join(os.getcwd(),source_dataset_folder_path,subfolder_path,f'{image_index}.png')
            target_file_to_path = os.path.join(os.getcwd(),source_dataset_folder_path,f'step{step_frames}',f'{subfolder_path}_{image_index}.png')
            shutil.copyfile(target_file_from_path,target_file_to_path)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.images_paths = sorted([os.path.join(data_path,image_path) for image_path in os.listdir(data_path)])

    def __getitem__(self, idx):
        img_path = self.images_paths[2*idx]
        img = cv2.imread(img_path)
        target_path = self.images_paths[2*idx+1]
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

        img = np.transpose(img, (2,0,1))
        img = (img-img.min())/(img.max()-img.min())

        return img.astype(np.double), target.astype(np.double)

    def __len__(self):
        return len(self.images_paths)//2