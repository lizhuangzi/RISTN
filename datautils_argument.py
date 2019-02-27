from os import listdir
from os.path import join
import random
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop,Lambda, ToTensor, ToPILImage, CenterCrop, Scale,RandomHorizontalFlip,RandomVerticalFlip,RandomRotation,Resize,ColorJitter,Normalize
import skimage.io
import torch
import numpy as np

import torchvision.transforms.functional as F

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG','bmp'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        RandomVerticalFlip(),
        RandomHorizontalFlip(),
        ColorJitter(0.2,0.2,0.1,0.1),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Scale(crop_size // upscale_factor, interpolation=Image.BICUBIC),

    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Scale(400),
        CenterCrop(400),
        ToTensor()
    ])








import os

class ValDatasetFromFolder2():
    def __init__(self, dataset_dir, upscale_factor):
        self.currentIndex = 0
        self.relation = 5
        self.sequeueslists = []
        self.dataset_dir = dataset_dir
        for x in listdir(dataset_dir):
            self.sequeueslists.append(join(dataset_dir, x))

    def generatebatch(self,index):
        squeen = self.sequeueslists[index]
        image_filenames = []
        for x in listdir(squeen):
            sx = join(squeen, x)
            if is_image_file(sx):
                image_filenames.append(sx)

        length = len(image_filenames)
        totalnumber = 0
        leng = length - length% self.relation
        tt0,tt1,tt2 = None,None,None
        for i in range(0,leng-3,3):

            t0, t1, t2 = None, None,None
            for j in range( self.relation):
                path = os.path.join(self.dataset_dir,os.listdir(self.dataset_dir)[index],'frame_%.04d.png' % (totalnumber + 2))
                hr_image1 = Image.open(path)
                w, h = hr_image1.size
                crop_size1 = calculate_valid_crop_size(w, 4)
                crop_size2 = calculate_valid_crop_size(h, 4)
                lr_scale = Resize((crop_size2 // 4, crop_size1 // 4), interpolation=Image.BICUBIC)
                hr_scale = Resize((crop_size2, crop_size1), interpolation=Image.BICUBIC)

                hr_image1 = CenterCrop((crop_size2, crop_size1))(hr_image1)
                lr_image1 = lr_scale(hr_image1)

                # Y channel
                lr_image = torch.unsqueeze(ToTensor()(lr_image1), dim=0)
                hr_image = torch.unsqueeze(ToTensor()(hr_image1), dim=0)
                bic_hr = torch.unsqueeze(ToTensor()(hr_scale(lr_image1)), dim=0)


                # lr_image = torch.unsqueeze(lr_image, 0)
                # bic_hr = torch.unsqueeze(bic_hr, 0)
                # hr_image = torch.unsqueeze(hr_image, 0)
                totalnumber+=1

                if j == 0:
                    t0 = lr_image
                    t1 = bic_hr
                    t2 = hr_image
                else:
                    t0 = torch.cat((t0, lr_image), 0)
                    t1 = torch.cat((t1, bic_hr), 0)
                    t2 = torch.cat((t2, hr_image), 0)

            t0 = torch.unsqueeze(t0,dim=0)
            t1= torch.unsqueeze(t1,dim=0)
            t2 =torch.unsqueeze(t2,dim=0)

            totalnumber = totalnumber - 2
            if i == 0:
                tt0 = t0
                tt1 = t1
                tt2 = t2
            else:
                tt0 = torch.cat((tt0, t0), 0)
                tt1 = torch.cat((tt1, t1), 0)
                tt2 = torch.cat((tt2, t2), 0)

        return tt0,tt1,tt2




class ValDatasetFromFolder3():
    def __init__(self, dataset_dir, upscale_factor):
        self.currentIndex = 0
        self.relation = 5
        self.sequeueslists = []
        self.dataset_dir = dataset_dir
        self.framelists = os.listdir(dataset_dir)

    def generatebatch(self,index=0):

        length = len(self.framelists)
        leng = length - length% self.relation
        tt0,tt1,tt2 = None,None,None
        for i in range(0,leng- self.relation):

            t0, t1, t2 = None, None,None
            for j in range( self.relation):
                str1 = '%d.png' % (i+j)
                lrpath = os.path.join(self.dataset_dir,str1)
                lr_image1 = Image.open(lrpath)
                w, h = lr_image1.size


                # Y channel
                lr_image = torch.unsqueeze(ToTensor()(lr_image1), dim=0)


                if j == 0:
                    t0 = lr_image

                else:
                    t0 = torch.cat((t0, lr_image), 0)

            t0 = torch.unsqueeze(t0,dim=0)



            if i == 0:
                tt0 = t0

            else:
                tt0 = torch.cat((tt0, t0), 0)


        return tt0


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_imgg = skimage.io.imread(self.image_filenames[index])
        hr_image = Image.open(self.image_filenames[index])
        if len(hr_imgg.shape) !=3 :
            hr_image = hr_image.convert('RGB')
        w, h = hr_image.size
        crop_size1 = calculate_valid_crop_size(w, 4)
        crop_size2 = calculate_valid_crop_size(h, 4)

        lr_scale = Resize((crop_size2 // 4, crop_size1 // 4), interpolation=Image.BICUBIC)
        hr_scale = Resize((crop_size2, crop_size1), interpolation=Image.BICUBIC)

        hr_image = CenterCrop((crop_size2, crop_size1))(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)

        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)
