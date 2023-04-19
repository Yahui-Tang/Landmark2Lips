import os
import torch
import numpy as np
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter


class TensorboardSummary(object):
    def __init__(self, directory):
        self.writer = SummaryWriter(log_dir=os.path.join(directory))

    def visualize_normal_0_1(self, input_data):
        if isinstance(input_data, torch.Tensor):
            max = input_data.max()
            min = input_data.min()
            input_data = (input_data - min) / (max - min + 0.0001)
        elif isinstance(input_data, np.ndarray):
            max = np.max(input_data)
            min = np.min(input_data)
            input_data = np.float32((input_data - min) / np.maximum((max - min), 0.001))
        return input_data

    def Write_visualize_grid_image(self, tensor_list: list, img_name: str, global_step=None):
        nrow = tensor_list[0].size()[0]
        grid_img = None
        for data in tensor_list:
            if grid_img is None:
                grid_img = self.visualize_normal_0_1(data)
            else:
                grid_img = torch.cat([grid_img, self.visualize_normal_0_1(data)], 0)
        grid_img = make_grid(grid_img, nrow, 3, normalize=False, range=(0, 1), scale_each=False, pad_value=0)
        self.writer.add_image(img_name, grid_img, global_step)

    def Write_feaure_map_grid(self, tensor_list: list, img_name: str, global_step=None):
        # tensor_list []
        nrow = tensor_list[0].size()[0]
        grid_img = None
        for data in tensor_list:
            if grid_img is None:
                grid_img = self.visualize_normal_0_1(data.detach().cpu())
            else:
                grid_img = torch.cat([grid_img, self.visualize_normal_0_1(data.detach().cpu())], 0)
        grid_img = make_grid(grid_img, nrow, 3, normalize=False, range=(0, 1), scale_each=False, pad_value=0)
        self.writer.add_image(img_name, grid_img, global_step)

    def Write_value_log(self, scalar_name: str, value, n_iter):
        self.writer.add_scalar(scalar_name, value, global_step=n_iter)

    def Write_value_logs(self, scalar_name: str, tag: list, value: list, n_iter):
        temp_dic = {}
        for i in range(len(tag)):
            temp_dic[tag] = value[i]
        self.writer.add_scalars(scalar_name, temp_dic, global_step=n_iter)

    def Write_refesh(self):
        self.writer.close()