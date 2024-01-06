# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#    来自vitslim  后面看看能不能都搞成基于的timm
import os
import json
import pickle
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
#https://blog.csdn.net/weixin_46713695/article/details/125032851

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def getpicname():
    #fileroot = "~/dataset/puzzle2k"
    #filename = "my.xlsx"
    #filepath = fileroot +"/"+ filename
    filepath = './0424cleaned.csv'#1000000  11111111111111
    print(filepath)
    #data_info = pd.read_excel(filepath, sheet_name="Sheet1", header = 0)
    data_info = pd.read_csv(filepath)
    print(".columns",data_info.columns)
    print(data_info.index)
    # print(data_info.at[2,'图片url'])
    # print(type(data_info.at[2,'图片url']))
    # name = data_info.at[2,'图片url'].split('/')[-1]
    # print(name)

    m = data_info.values[:,1]
    c = m.astype(str)
    print("more")
    #name_sp = np.char.split(c,sep = "/").tolist()
    name_sp = np.char.split(c,sep = "/")
    #print(name_sp)
    for i in range(len(name_sp)):
        name_sp[i] = name_sp[i][-1]
    #print(len(name_sp))
    m64 = data_info.values[:,2]
    #print(len(m64))
    m100 = data_info.values[:,3]
    #print(len(m100))
    return name_sp,m64,m100



class ReadDataFromCSV(Dataset):
    def __init__(self, train,csv_path, transform,infer=False):
        # Transforms
        self.transform = transform
        self.file_path = "/homeB/liangxiaoyu/dataset/puzzle2k/img/"
        
        # read csv
        #self.data_info = pd.read_csv(csv_path, header=None)
        #self.image_arr,self.label_arr64,self.label_arr100 = getpicname()
        image_arr,label_arr64,label_arr100 = getpicname()
        #self.label_arr = np.asarray(  )
        lenth = 10000#042555555555555 2222222222222
        p = 0.8
        lenp = int(lenth * p)
        self.infer = infer

        if train:
            self.image_arr,self.label_arr64,self.label_arr100 = image_arr[0:lenp],label_arr64[0:lenp],label_arr100[0:lenp]
        else:
            self.image_arr,self.label_arr64,self.label_arr100 = image_arr[lenp:-1],label_arr64[lenp:-1],label_arr100[lenp:-1]
        
        if self.infer:
            self.image_arr,self.label_arr64,self.label_arr100 = image_arr[:],label_arr64[:],label_arr100[:]


        self.data_len = len(self.image_arr)
    def __getitem__(self, index):
        # get image
        single_img_name = self.image_arr[index]
        #single_img_img = Image.open('../' + single_img_name)
        
        #single_img_img = Image.open(self.file_path + single_img_name )

        img_fn = os.path.join(self.file_path, single_img_name)
        single_img_img = Image.open(img_fn).convert('RGB')
        #single_img_img.show()

        
        single_img_tensor = self.transform(single_img_img)

        # get label
        single_image_label64 = self.label_arr64[index]
        single_image_label100 = self.label_arr100[index]
        
        if self.infer:
            return(single_img_name,single_img_tensor, single_image_label64, single_image_label100)

        return (single_img_tensor, single_image_label64, single_image_label100)
    def __len__(self):
        return self.data_len




def build_dataset(is_train,args, infer=False):
    transform = build_transform(is_train, args)

    if args.b_data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.b_data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.b_data_set == 'IMNET':
        root = os.path.join(args.b_data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.b_data_set == 'puzzle':
        dataset = ReadDataFromCSV(train=is_train,csv_path = '',transform=transform,infer=infer)
        #dataset.data_len = 5000

        print(is_train,infer,dataset.data_len)
        nb_classes = 10
    else :
        return 0,0

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = 0
    is_train = 0
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.b_input_size,
            is_training=True,
            color_jitter=args.b_color_jitter,
            auto_augment=args.b_aa,
            interpolation=args.b_train_interpolation,
            re_prob=args.b_reprob,
            re_mode=args.b_remode,
            re_count=args.b_recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.b_input_size, padding=4)
        return transform

    t = []
    t.append(transforms.Resize(224))


    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)



# ---------------------------------------------------------------------------- #
import torchvision.datasets as dset
import torchvision.transforms as transforms

def init_dataset(data_path,datasetname):
    # Init dataset
    if not os.path.isdir(data_path):
        print("模型不存在")
        os.makedirs(data_path)

    if datasetname == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif datasetname == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    else:
        assert False, "Unknow dataset : {}".format(datasetname)

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

    if datasetname == 'cifar10':
        train_data = dset.CIFAR10(data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(data_path, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif datasetname == 'cifar100':
        train_data = dset.CIFAR100(data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(data_path, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif datasetname == 'svhn':
        train_data = dset.SVHN(data_path, split='train', transform=train_transform, download=True)
        test_data = dset.SVHN(data_path, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif datasetname == 'stl10':
        train_data = dset.STL10(data_path, split='train', transform=train_transform, download=True)
        test_data = dset.STL10(data_path, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif datasetname == 'imagenet':
        assert False, 'Do not finish imagenet code'
    else:
        assert False, 'Do not support dataset : {}'.format(datasetname)
    return train_data,test_data,num_classes