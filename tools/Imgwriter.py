import torch
import numpy as np
import cv2 as cv
import torchvision


def save_tensor_as_grid_png(tensor_list: list, name=''):  # 将若干个相同大小的tensor保存为网格图片
    nrow = tensor_list[0].size()[0]
    grid_img = None
    for data in tensor_list:
        if grid_img is None:
            grid_img = data.detach().cpu()
        else:
            grid_img = torch.cat([grid_img, data.detach().cpu()], 0)

        torchvision.utils.save_image(grid_img, name, nrow=nrow, padding=2,
                                     normalize=False, range=(0, 1), scale_each=False, pad_value=0)


def from_tensor_to_image(tensor, device='cuda'):
    """ converts tensor to image """
    tensor = torch.squeeze(tensor, dim=0)
    if device == 'cpu':
        image = tensor.data.numpy()
    else:
        image = tensor.cpu().data.numpy()
    # CHW to HWC
    image = image.transpose((1, 2, 0))
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image


def outOfGamutClipping(I):
    """ Clips out-of-gamut pixels. """
    I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
    I[I < 0] = 0  # any pixel is below 0, clip it to 0
    return I


def saving_tensor_as_img(path, save_img, bit_depth='8bit'):
    save_img = save_img.detach().cpu()
    save_img = from_tensor_to_image(save_img, device='cpu')
    save_img = outOfGamutClipping(save_img)
    if bit_depth == '8bit':
        save_img = save_img * 255
        cv.imwrite(path, save_img.astype(np.uint8))
    elif bit_depth == '16bit':
        save_img = save_img * 65535
        cv.imwrite(path, save_img.astype(np.uint16))