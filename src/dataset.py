import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as t
import os
import numpy as np
import cv2
from typing import Any, Callable
from .utils.transform import *


def load_image(image_path: str, flag, size = (256,256), norm: Callable = None):
    flag_str = flag
    if isinstance(flag, str):
        flag = cv2.IMREAD_GRAYSCALE if 'gray' in flag or flag == 'gray' else cv2.IMREAD_COLOR
    image = cv2.imread(image_path, flag)
    if (flag_str != 'hsv' or 'hsv' not in flag_str) and flag != cv2.IMREAD_GRAYSCALE:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif flag != cv2.IMREAD_GRAYSCALE:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.resize(image, size, interpolation= cv2.INTER_CUBIC)
    if norm is not None:
        try:
            image = norm(image)
        except:
            pass
    return image

class Normalize:
    def __init__(self, type: str = 'range01', mean = [.5, .5, .5], std = [.5, .5, .5], eps = 1e-9) -> None:
        assert type in ['range01','meanstd', 'identity']
        assert not(type == 'meanstd' and mean is None and std is None)
        self.type = type
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.eps = eps

    def __call__(self,image) -> Any:

        image = np.array(image).astype(float)
        
        if self.type == 'range01':
            imin = min(image.min(), 0)  ## We only want to get value below 0 to 0.
            imax = image.max()

            return (image - imin)/(imax - imin + self.eps)
        elif self.type == 'meanstd':

            curr_mean = np.broadcast_to(image.mean((0,1)), image.shape)
            curr_std = np.broadcast_to(image.std((0,1)), image.shape)

            image_normal = (image - curr_mean) / ( curr_std + self.eps)  #Image have mean 0 and std 1

            image = image_normal * np.broadcast_to(self.std, image.shape) + np.broadcast_to(self.mean, image.shape)

            return image
        else:
            return image


class StandardDataset(Dataset):
    def __init__(self, 
                 root_dir= '',  
                 transform= t.Compose([RandomErasing(), Cutout(), RandomHorizontalFlip(), ToTensor(), RandomResizedCrop(), ColorJitter()]),
                 get_real = True,
                 preload= True, 
                 norm = Normalize(), 
                 im_size = (256,256), 
                 dep_im_size = (32,32),
                 get_hsv = False, **kwarg):
        self.root_dir = root_dir
        self.transform = transform
        self.norm = norm
        self.im_size = im_size
        self.dep_im_size = dep_im_size
        self.hsv = get_hsv
        self.preprocess(get_real, preload)



    
    def __len__(self):
        return self.len

    def __getitem__(self, idx) -> Any:
        return self.get1_sample(idx)


    def get1_sample(self, idx):
        if self.lines is not None:
            # print(len(self.lines[idx]))
            dir, filename, label = self.lines[idx][0], self.lines[idx][1], int(self.lines[idx][2])
        else:
            with open(f'{self.root_dir}Imfile.txt', 'r+') as f:
                lines = f.readlines()
            dir, filename, label = lines[idx].strip().split(" ")
            label = int(label)


        rgb_im = load_image(dir+'/color'+filename, flag= 'color', size= self.im_size,norm= self.norm)
        
        dep_im = load_image(dir+'/depth'+filename, flag= 'gray', size= self.dep_im_size, norm= self.norm)
        if self.hsv:
            hsv_im = load_image(dir+'/color'+filename, flag= 'hsv', size= self.im_size,norm= self.norm)
            sample = (rgb_im, hsv_im, dep_im)
        else:
            sample = (rgb_im, dep_im)
        
        if self.transform is not None:
            sample = self.transform(sample)

        if self.hsv:
            rgb_im, hsv_im, dep_im = sample
            cat_img = torch.cat([rgb_im, hsv_im], dim = 0)
        else:
            rgb_im, dep_im = sample
            cat_img = rgb_im

        return cat_img.float(), dep_im.float().squeeze(0), torch.tensor(label).float()
    
    
    def preprocess(self, get_real, preload):
        self.filename = self.root_dir
        self.len = 0
        if get_real is not None:
            mode = "real" if get_real else "fake"
            with open(f'{self.filename}{mode}Imfile.txt', 'w+') as f:
                dir = self.root_dir + r'/color'
                for filename in os.listdir(dir):
                    if mode in filename:
                        f.writelines(self.root_dir + f' /{filename} {1 if get_real else 0}'+'\n')
                        self.len += 1

            if preload:
                with open(f'{self.filename}{mode}Imfile.txt', 'r+') as f:
                    lines = f.readlines()
                    self.lines = [path.strip().split(" ") for path in lines]
                os.remove(f'{self.filename}{mode}Imfile.txt')
            else:
                self.lines = None
        else:
            with open(f'{self.filename}Imfile.txt', 'w+') as f:
                dir = self.root_dir + r'/color'
                for filename in os.listdir(dir):
                    f.writelines(self.root_dir + f' /{filename} {1 if "real" in filename else 0}'+'\n')
                    self.len += 1
            if preload:
                with open(f'{self.filename}Imfile.txt', 'r+') as f:
                    lines = f.readlines()
                    self.lines = [path.strip().split(" ") for path in lines]
                os.remove(f'{self.filename}Imfile.txt')
            else:
                self.lines = None

    def get_label(self):
        if self.lines is None:
                with open(f'{self.filename}Imfile.txt', 'r+') as f:
                    lines = f.readlines()
                    lines = [path.strip().split(" ") for path in lines]
        else:
                lines = self.lines
                
        label = list(zip(*lines))[2]
        
        return label
    
    def get_weight(self):
        cls_label = self.get_label()

        labels, count = np.unique(cls_label, return_counts=True)
        labels = labels.tolist()
        count = 1- count.astype(float) / count.sum()
        count = count / count.sum()
        
        cls_weight = list(map(lambda n: count[labels.index(n)], cls_label))

        return cls_weight



class CombinedDataset(StandardDataset):
    def __init__(self, root_dir:list[str]= [''],
                 transform= t.Compose([RandomErasing(), Cutout(), RandomHorizontalFlip(), ToTensor(), RandomResizedCrop(), ColorJitter()]),
                 get_real = True,
                 preload= True, 
                 norm = Normalize(), 
                 im_size = (256,256), 
                 dep_im_size = (32,32),
                 get_hsv = False, **kwarg):
        super(CombinedDataset, self).__init__(root_dir, transform, get_real, preload, norm, im_size, dep_im_size, get_hsv, **kwarg)

    def preprocess(self, get_real, preload):
        self.filename = ''
        self.len = 0
        for dir in self.root_dir:
            self.filename += dir.replace('/','_')
            
        if get_real is not None:
            mode = "real" if get_real else "fake"
            with open(f'{self.filename}{mode}Imfile.txt', 'w+') as f:
                for dir in self.root_dir:
                    tmp_dir = dir + r'/color'
                    for file in os.listdir(tmp_dir):
                        if mode in file:
                            f.writelines(dir + f' /{file} {1 if get_real else 0}'+'\n')
                            self.len += 1
            if preload:
                with open(f'{self.filename}{mode}Imfile.txt', 'r+') as f:
                    lines = f.readlines()
                    self.lines = [path.strip().split(" ") for path in lines]
                os.remove(f'{self.filename}{mode}Imfile.txt')
            else:
                self.lines = None
                
        else:
            with open(f'{self.filename}Imfile.txt', 'w+') as f:
                for dir in self.root_dir:
                    tmp_dir = dir + r'/color'
                    for filename in os.listdir(tmp_dir):
                        f.writelines(dir + f' /{filename} {1 if "real" in filename else 0}'+'\n')    
                        self.len += 1
            if preload:
                with open(f'{self.filename}Imfile.txt', 'r+') as f:
                    lines = f.readlines()
                    self.lines = [path.strip().split(" ") for path in lines]
                os.remove(f'{self.filename}Imfile.txt')
            else:
                self.lines = None



# class StandardDataset(Dataset):
#     def __init__(self, 
#                  root_dir= '',  
#                  transform= t.Compose([RandomErasing(), RandomHorizontalFlip(), Cutout(), ToTensor()]),
#                  get_patch = True,
#                  num_patch = 1,
#                  preload= True, 
#                  norm = Normalize(), 
#                  im_size = (256,256), 
#                  dep_im_size = (32,32), **kwarg):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.norm = norm
#         self.im_size = im_size
#         self.dep_im_size = dep_im_size
#         self.preprocess()
#         if preload:
#             with open(f'{self.filename}Imfile.txt', 'r+') as f:
#                 lines = f.readlines()
#                 self.lines = [path.strip().split(" ") for path in lines]
#             os.remove(f'{self.filename}Imfile.txt')
#         else:
#             self.lines = None

#         self.get_patch = get_patch
#         self.num_patch = num_patch
#         self.cropper = RandomResizedCrop()

    
#     def __len__(self):
#         if self.lines is not None:
#             return len(self.lines)

#         with open(f'{self.filename}Imfile.txt', 'r+') as f:
#             len_ds = len(f.readlines())
#         return len_ds-1

#     def __getitem__(self, idx) -> Any:
#         if self.num_patch == 1:
#             return self.get1_sample(idx)
#         else:
#             return self.get2random_patch(idx)

#     def get1_sample(self, idx):
#         if self.lines is not None:
#             dir, filename = self.lines[idx][0], self.lines[idx][1]
#         else:
#             with open(f'{self.root_dir}Imfile.txt', 'r+') as f:
#                 lines = f.readlines()
#             dir, filename = lines[idx].strip().split(" ")


#         rgb_im = load_image(dir+'/color'+filename, flag= 'color', size= self.im_size,norm= self.norm)
#         dep_im = load_image(dir+'/depth'+filename, flag= 'gray', size= self.dep_im_size, norm= self.norm)

#         label = 0 if 'fake' in dir+filename else 1

#         sample = (rgb_im, dep_im)
        
#         if self.transform is not None:
#             sample = self.transform(sample)

#         if self.get_patch:
#             rgb_im, dep_im = self.cropper(sample)
#         else:
#             rgb_im, dep_im = sample

#         return rgb_im, dep_im, torch.tensor(label)
    
#     def get2random_patch(self, idx):
#         if self.lines is not None:
#             dir, filename = self.lines[idx][0], self.lines[idx][1]
#         else:
#             with open(f'{self.root_dir}Imfile.txt', 'r+') as f:
#                 lines = f.readlines()
#             dir, filename = lines[idx].strip().split(" ")


#         rgb_im = load_image(dir+'/color'+filename, flag= 'color', size= self.im_size,norm= self.norm)
#         dep_im = load_image(dir+'/depth'+filename, flag= 'gray', size= self.dep_im_size, norm= self.norm)

#         label = 0 if 'fake' in dir+filename else 1

#         sample = (rgb_im, dep_im)
        
#         if self.transform is not None:
#             sample1 = self.transform(sample)
#         else:
#             sample1 = sample
#         rgb_im1, map_x1 = self.cropper(sample1)

#         if self.transform is not None:
#             sample2 = self.transform(sample)
#         else:
#             sample2 = sample
#         rgb_im2, map_x2 = self.cropper(sample2)

#         return rgb_im1, rgb_im2, map_x1, map_x2, torch.tensor(label)

    
#     def preprocess(self):
#         self.filename = self.root_dir
#         with open(f'{self.filename}Imfile.txt', 'w+') as f:
#             dir = self.root_dir + r'/color'
#             for filename in os.listdir(dir):
#                 f.writelines(self.root_dir + f' /{filename}'+'\n')



# class CombinedDataset(StandardDataset):
#     def __init__(self, root_dir:list[str]= [''],
#                  transform= t.Compose([RandomErasing(), RandomHorizontalFlip(), Cutout(), ToTensor()]),
#                  get_patch = True,
#                  num_patch = 1,
#                  preload= True, 
#                  norm = Normalize(), 
#                  im_size = (256,256), 
#                  dep_im_size = (32,32), **kwarg):
#         super(CombinedDataset, self).__init__(root_dir, transform, get_patch,num_patch, preload, norm, im_size, dep_im_size, **kwarg)

#     def preprocess(self):
#         self.filename = ''
#         for dir in self.root_dir:
#             self.filename += dir.replace('/','_')
#         with open(f'{self.filename}Imfile.txt', 'w+') as f:
#             for dir in self.root_dir:
#                 tmp_dir = dir + r'/color'
#                 for file in os.listdir(tmp_dir):
#                     f.writelines(dir + f' /{file}'+'\n')