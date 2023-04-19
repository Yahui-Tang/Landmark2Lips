import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import scipy.ndimage
import PIL.Image
import numpy as np
from models.generator import InpaintGenerator
from tools.Imgloader import loading_img_as_tensor_list, get_dir_filename_list, loading_img_as_numpy
from tools.Imgwriter import saving_tensor_as_img, save_tensor_as_grid_png
import face_alignment

def net_init(device='cpu'):
    gen_net = InpaintGenerator()
    weight = torch.load('./ckp/gen.pth', map_location='cpu')['generator']
    gen_net.load_state_dict(weight)
    if device == 'cuda':
        return gen_net.cuda()
    else:
        return gen_net

def face_init(img_dir, device='cpu'):
    img_name_list = get_dir_filename_list(img_dir, type='.jpg')
    if device == 'cuda':
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, face_detector='sfd', device='cuda')
    else:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, face_detector='sfd', device='cpu')
    img_list = []
    ldm_list = []
    mask_list = []

    for img_name in img_name_list:
        img_path = img_dir + '/' + img_name
        img = PIL.Image.open(img_path)
        l_pos = fa.get_landmarks(np.array(img, dtype=np.uint8))[0]

        # Choose oriented crop rectangle.
        lm = l_pos
        eye_avg = (lm[40] + lm[46]) * 0.5 + 0.5
        mouth_avg = (lm[48] + lm[54]) * 0.5 + 0.5
        eye_to_eye = lm[46] - lm[40]
        eye_to_mouth = mouth_avg - eye_avg
        x = eye_to_eye - np.array([-eye_to_mouth[1], eye_to_mouth[0]])
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.array([-x[1], x[0]])
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        zoom = 1024 / (np.hypot(*x) * 2)

        # crop
        border = max(int(np.round(1024 * 0.1 / zoom)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Simulate super-resolution.
        superres = int(np.exp2(np.ceil(np.log2(zoom))))
        if superres > 1:
            img = img.resize((img.size[0] * superres, img.size[1] * superres), PIL.Image.ANTIALIAS)
            quad *= superres
            zoom /= superres

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if max(pad) > border - 4:
            pad = np.maximum(pad, int(np.round(1024 * 0.3 / zoom)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.mgrid[:h, :w, :1]
            mask = 1.0 - np.minimum(np.minimum(np.float32(x) / pad[0], np.float32(y) / pad[1]), np.minimum(np.float32(w-1-x) / pad[2], np.float32(h-1-y) / pad[3]))
            blur = 1024 * 0.02 / zoom
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.round(img), 0, 255)), 'RGB')
            quad += pad[0:2]

        img = img.transform((4096, 4096), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        img = img.resize((256, 256), PIL.Image.ANTIALIAS)
        img = np.asarray(img, dtype=np.uint8)

        l_pos_256 = fa.get_landmarks(img)[0].astype('int')

        mask_h = l_pos_256[8, 1] - l_pos_256[33, 1] + 1
        mask_w = l_pos_256[13, 0] - l_pos_256[3, 0] + 2
        mask = np.zeros((256, 256, 1), dtype=np.float32)
        mask_x = l_pos_256[3, 0] - 1
        mask_y = l_pos_256[33, 1] + 1

        mask[mask_y:mask_y + mask_h, mask_x:mask_x + mask_w, 0] = 1

        landmark_map = np.zeros((256, 256, 1), dtype=np.float32)
        landmark_map[l_pos_256[0:68, 1], l_pos_256[0:68, 0], 0] = 1

        img_list.append(torch.from_numpy(img.transpose(2, 0, 1).astype('float32')/255.))
        ldm_list.append(torch.from_numpy(landmark_map.transpose(2, 0, 1)))
        mask_list.append(torch.from_numpy(mask.transpose(2, 0, 1)))


    return {'face_img': img_list, 'landmark': ldm_list, 'mask': mask_list}


def main():
    eval_img_dir = 'C:/Users/73891/Desktop/sheng'
    face_dic = face_init(eval_img_dir, device='cuda')
    torch.cuda.empty_cache()
    img_list = face_dic['face_img']
    ldm_list = face_dic['landmark']
    mask_list = face_dic['mask']
    count = len(img_list)

    with torch.no_grad():
        net = net_init(device='cuda')
        grid_list=[]
        for count in range(count):
            face_img = img_list[count].unsqueeze(0).cuda()
            ldm = ldm_list[count].unsqueeze(0).cuda()
            mask = mask_list[count].unsqueeze(0).cuda()

            images_masked = (face_img * (1 - mask).float()) + mask
            inputs = torch.cat((images_masked, ldm), dim=1)
            scaled_masks_quarter = F.interpolate(mask, size=[int(mask.shape[2] / 4), int(mask.shape[3] / 4)],
                                         mode='bilinear', align_corners=True)
            scaled_masks_half = F.interpolate(mask, size=[int(mask.shape[2] / 2), int(mask.shape[3] / 2)],
                                         mode='bilinear', align_corners=True)

            outputs = net(inputs,mask,scaled_masks_half,scaled_masks_quarter)
            outputs_merged = (outputs * mask) + (face_img * (1 - mask))

            grid_list.append(torch.cat((face_img, torch.cat((ldm, ldm, ldm), dim=1), images_masked, outputs_merged), dim=0))
            print(count)


        save_tensor_as_grid_png(grid_list, 'output.png')


if __name__ == '__main__':
    # from PIL import Image
    # face_dic = face_init('C:/Users/73891/Desktop/sheng', device='cuda')
    # saving_tensor_as_img('1.png', face_dic['face_img'][0]/255)
    # print("OK")
    main()
