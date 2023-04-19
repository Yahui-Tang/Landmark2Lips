import cv2 as cv
import os
import threadpool
from torchvision import transforms
import torch
import torchvision
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from math import exp


def get_dir_filename_list(dir, type='.png'):
    files = []
    for _, _, filenames in os.walk(dir):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() == type.lower():
                files.append(filename)
    files.sort()
    return files


def loading_img_as_numpy(path, flag=None, normalize=False):
    img_data = cv.imread(path, flag)
    img_data = cv.cvtColor(img_data, cv.COLOR_BGR2RGB)
    if normalize:
        img_data = cv.normalize(img_data.astype('float32'), None, 0.0, 1.0, cv.NORM_MINMAX)
    else:
        if flag == -1:
            img_data = img_data.astype('float32') / 65535.
        else:
            img_data = img_data.astype('float32') / 255.
    return img_data


def loading_img_as_numpy_list(dir, type='.png', flag=None, normalize=True):
    img_data_list = []
    img_name_list = get_dir_filename_list(dir, type=type)
    for img_name in img_name_list:
        img_path = dir + '/' + img_name
        img_data = loading_img_as_numpy(img_path, flag=flag, normalize=normalize)
        img_data_list.append(img_data)

    return img_data_list


def loading_img_as_batch(dir, type='.png', flag=None, normalize=True):
    trans = transforms.ToTensor()
    img_data_batch = None
    img_name_list = get_dir_filename_list(dir, type=type)
    for img_name in img_name_list:
        img_path = dir + '/' + img_name
        img_data = loading_img_as_numpy(img_path, flag=flag, normalize=normalize)
        if img_data_batch is None:
            img_data_batch = trans(img_data).unsqueeze(0)
        else:
            img_data_batch = torch.cat([img_data_batch, trans(img_data).unsqueeze(0)], dim=0)

    return img_data_batch


def loading_img_as_tensor_list(dir, type='.png', flag=None, normalize=True, unet_cut=0):
    trans = transforms.ToTensor()
    img_data_list = []
    img_name_list = get_dir_filename_list(dir, type=type)
    for img_name in img_name_list:
        img_path = dir + '/' + img_name
        img_data = loading_img_as_numpy(img_path, flag=flag, normalize=normalize)
        if unet_cut != 0:
            h, w, c = img_data.shape
            h_offset = h % unet_cut
            w_offset = w % unet_cut
            img_data = img_data[h_offset:, w_offset:, :]
        img_data_list.append(trans(img_data).unsqueeze(0))

    return img_data_list


def loading_img_as_tensor_list_from_list(path_list, flag=None, normalize=False, unet_cut=0):
    trans = transforms.ToTensor()
    img_data_list = []
    for img_path in path_list:
        img_data = loading_img_as_numpy(img_path, flag=flag, normalize=normalize)
        if unet_cut != 0:
            h, w, c = img_data.shape
            h_offset = h % unet_cut
            w_offset = w % unet_cut
            img_data = img_data[h_offset:, w_offset:, :]
        img_data_list.append(trans(img_data).unsqueeze(0))

    return img_data_list


def loading_img_as_tensor(path, bit_depth='8bit', normalize=False, device='cuda'):
    img_data = cv.imread(path, -1)
    img_data = cv.cvtColor(img_data, cv.COLOR_BGR2RGB)
    trans = transforms.ToTensor()
    if normalize:
        img_data = cv.normalize(img_data.astype('float32'), None, 0.0, 1.0, cv.NORM_MINMAX)
    else:
        if bit_depth == '16bit':
            img_data = img_data.astype('float32') / 65535.
        else:
            img_data = img_data.astype('float32') / 255.
    img_data_tensor = trans(img_data).unsqueeze(0)
    if device == 'cuda':
        img_data_tensor = img_data_tensor.cuda()
    return img_data_tensor




