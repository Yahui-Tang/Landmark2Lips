import cv2
import torch
import cv2 as cv
import numpy as np
import random
import logging
from tools.Imgloader import loading_img_as_tensor_list, get_dir_filename_list, loading_img_as_numpy
from torchvision import transforms
from torch.utils.data import DataLoader

from torch.utils.data import Dataset


def landmake_map_generator(landmark_cord, img_size = 256):
    '''
    :param landmark_cord: [B,self.config.LANDMARK_POINTS,2] or [self.config.LANDMARK_POINTS,2], tensor or numpy array
    :param img_size:
    :return: landmark_img [B, 1, img_size, img_size] or [1, img_size, img_size], tensor or numpy array
    '''
    if landmark_cord.ndimension() == 3:
        landmark_img = torch.zeros(landmark_cord.shape[0], 1, img_size, img_size).cuda()
        for i in range(landmark_cord.shape[0]):
            landmark_img[i, 0, landmark_cord[i, :, 1], landmark_cord[i, :, 0]] = 1
    elif landmark_cord.ndimension() == 2:
        landmark_img = torch.zeros(1, img_size, img_size)
        landmark_img[0, landmark_cord[:, 1], landmark_cord[:, 0]] = 1

    return landmark_img

def mask_generator(landmark_cord, img_size = 256):
    print("OK")


class CeleA_HQ(Dataset):
    def __init__(self, img_dir, ldm_dir, ldm_num=68, input_size=256, training=True):
        self.ldm_num = ldm_num
        self.training = training
        self.input_size = input_size
        self.img_dir = img_dir + '/'
        self.ldm_dir = ldm_dir + '/'
        self.img_name_list = get_dir_filename_list(img_dir, type='.png')
        self.ldm_name_list = get_dir_filename_list(ldm_dir, type='.txt')

    def mask_landmark_generator(self, ldm_path):
        landmarks = np.genfromtxt(ldm_path, dtype=np.str, encoding='utf-8')
        landmarks = landmarks.reshape(self.ldm_num, 2).astype('float').astype('int')
        landmarks = np.clip(landmarks, 0, self.input_size - 1)

        mask_h = landmarks[8, 1] - landmarks[33, 1] + 1
        mask_w = landmarks[13, 0] - landmarks[3, 0] + 2
        mask = np.zeros((self.input_size, self.input_size, 1), dtype=np.float32)
        mask_x = landmarks[3, 0] - 1
        mask_y = landmarks[33, 1] + 1

        mask[mask_y:mask_y + mask_h, mask_x:mask_x + mask_w, 0] = 1

        landmark_map = np.zeros((self.input_size, self.input_size, 1), dtype=np.float32)
        landmark_map[landmarks[0:self.ldm_num, 1], landmarks[0:self.ldm_num, 0], 0] = 1
        return landmark_map, mask

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, index):
        img_name = self.img_name_list[index]
        ldm_name = self.img_name_list[index][:-4] + '.txt'
        face_img = loading_img_as_numpy(self.img_dir + img_name, flag=1, normalize=False)
        ldm_map, mask = mask_landmark_generator(self.ldm_dir + ldm_name)

        return {'face_img': torch.from_numpy(face_img.transpose(2, 0, 1)),
                'landmark': torch.from_numpy(ldm_map.transpose(2, 0, 1)),
                'mask': torch.from_numpy(mask.transpose(2, 0, 1))}

def Ldm2img_DataLoader( img_dir, ldm_dir, batch_size=1, workers=8, shuffle=True):
    _dataset = CeleA_HQ(img_dir, ldm_dir)
    dataloader = DataLoader(dataset=_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=workers, drop_last=True)
    return dataloader


def eval_celeba_dataloader(eval_img_dir, eval_ldm_dir):
    img_name_list = get_dir_filename_list(eval_img_dir, type='.png')
    img_list = []
    ldm_list = []
    mask_list = []
    for img_name in img_name_list:
        img_path = eval_img_dir + '/' + img_name
        ldm_path = eval_ldm_dir + '/' + img_name[:-4] + '.txt'
        landmark_map, mask = mask_landmark_generator(ldm_path)
        img = loading_img_as_numpy(img_path, flag=1, normalize=False)

        img_list.append(torch.from_numpy(img.transpose(2, 0, 1)))
        ldm_list.append(torch.from_numpy(landmark_map.transpose(2, 0, 1)))
        mask_list.append(torch.from_numpy(mask.transpose(2, 0, 1)))

    img_dic = {
        'face_img': img_list,
        'landmark': ldm_list,
        'mask': mask_list
    }
    return img_dic


def mask_landmark_generator(ldm_path):
    ldm_num = 68
    input_size = 256

    landmarks = np.genfromtxt(ldm_path, dtype=np.str, encoding='utf-8')
    landmarks = landmarks.reshape(ldm_num, 2).astype('float').astype('int')
    landmarks = np.clip(landmarks, 0, input_size - 1)

    mask_h = landmarks[8, 1] - landmarks[33, 1] + 1
    mask_w = landmarks[13, 0] - landmarks[3, 0] + 2
    mask = np.zeros((input_size, input_size, 1), dtype=np.float32)
    mask_x = landmarks[3, 0] - 1
    mask_y = landmarks[33, 1] + 1

    mask[mask_y:mask_y + mask_h, mask_x:mask_x + mask_w, 0] = 1

    landmark_map = np.zeros((input_size, input_size, 1), dtype=np.float32)
    landmark_map[landmarks[0:ldm_num, 1], landmarks[0:ldm_num, 0], 0] = 1
    return landmark_map, mask


if __name__ == '__main__':
    from PIL import Image
    a, b = mask_landmark_generator('E:\\celeba-1024\\landmarks\\000009.txt')
    img = loading_img_as_numpy('E:\\celeba-1024\\celeba-re_512\\000009.png', flag=1, normalize=False)
    a = np.uint8(a * 255)
    b = np.uint8(b * 255)
    cv2.imwrite('test1.bmp', a)
    cv2.imwrite('test2.bmp', b)
    print("OK")
