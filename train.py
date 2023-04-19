import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import scipy.ndimage
import PIL.Image
import numpy as np
from models.generator import InpaintGenerator
from models.discriminator import Discriminator
from tools.Imgloader import loading_img_as_tensor_list, get_dir_filename_list, loading_img_as_numpy
from tools.Imgwriter import saving_tensor_as_img, save_tensor_as_grid_png
from loss import AdversarialLoss, PerceptualLoss, StyleLoss, TVLoss
import torch.optim as optim
from dataloader import Ldm2img_DataLoader, eval_celeba_dataloader


def eval_and_save_checkpoint(epochs, gen_net, eval_data, ckp_save_dir='', img_save_dir=''):
    face_img_list = eval_data['face_img']
    landmark_list = eval_data['landmark']
    mask_list = eval_data['mask']
    count = len(face_img_list)

    with torch.no_grad():
        gen_net.eval()
        pre_loss = 0
        grid_list = []
        for i in range(count):
            face_img = face_img_list[i].unsqueeze(0).cuda()
            landmark = landmark_list[i].unsqueeze(0).cuda()
            mask = mask_list[i].unsqueeze(0).cuda()

            images_masked = (face_img * (1 - mask).float()) + mask
            inputs = torch.cat((images_masked, landmark), dim=1)
            scaled_masks_quarter = F.interpolate(mask, size=[int(mask.shape[2] / 4), int(mask.shape[3] / 4)],
                                                 mode='bilinear', align_corners=True)
            scaled_masks_half = F.interpolate(mask, size=[int(mask.shape[2] / 2), int(mask.shape[3] / 2)],
                                              mode='bilinear', align_corners=True)

            outputs = gen_net(inputs, mask, scaled_masks_half, scaled_masks_quarter)
            outputs_merged = (outputs * mask) + (face_img * (1 - mask))

            grid_list.append(torch.cat((face_img, torch.cat((landmark, landmark, landmark), dim=1),
                                        images_masked, outputs_merged), dim=0))

            pre_loss = pre_loss + torch.mean(torch.pow(outputs_merged - face_img, 2))
        gen_net.train()
        pre_loss = pre_loss / count

        if pre_loss < 1.0e-10:
            pre_psnr = 100
        else:
            pre_psnr = (20 * torch.log10(1 / torch.sqrt(pre_loss))).item()

        checkpoint = {
            'weight': gen_net.state_dict(),
            'iteration': epochs
        }

        checkpoint_path = ckp_save_dir + '/gen' + '_' + str(epochs) + '_' + str(pre_psnr)[0:7] + '_' + '.pth'
        torch.save(checkpoint, checkpoint_path)
        save_tensor_as_grid_png(grid_list[:6], img_save_dir + '/gen' + '_' + str(pre_psnr)[0:7] + '_' + str(epochs) + '_' + '.png')



def main():
    # param
    lr = 0.0001
    batch_size = 12
    epochs = 10000
    l1_loss_weight = 1
    style_loss_weight = 250
    content_loss_weight = 0.1
    adv_loss_weight = 0.01
    tv_loss_weight = 0.1
    eval_interval = 1

    training_img_dir = '/home/tyh/dataset/CelebA-HQ/celeba_256_training'
    training_ldm_dir = '/home/tyh/dataset/CelebA-HQ/landmarks_training'
    eval_img_dir = '/home/tyh/dataset/CelebA-HQ/celeba_256_validation'
    eval_ldm_dir = '/home/tyh/dataset/CelebA-HQ/landmarks_validation'

    ckp_saving_dir = './ckp'
    img_saving_dir = './output'

    # net
    generator = InpaintGenerator().cuda()
    discriminator = Discriminator(in_channels=4, use_sigmoid=True).cuda()

    # loss function
    l1_loss = nn.L1Loss().cuda()
    perceptual_loss = PerceptualLoss().cuda()
    style_loss = StyleLoss().cuda()
    adversarial_loss = AdversarialLoss().cuda()
    tv_loss = TVLoss().cuda()

    # optimizer
    gen_optimizer = optim.Adam(params=generator.parameters(), lr=lr, betas= (0, 0.9))
    dis_optimizer = optim.Adam(params=discriminator.parameters(), lr=lr*0.1, betas=(0, 0.9))

    # dataset
    loader = Ldm2img_DataLoader(img_dir=training_img_dir, ldm_dir=training_ldm_dir, batch_size=batch_size)
    eval_data = eval_celeba_dataloader(eval_img_dir=eval_img_dir, eval_ldm_dir=eval_ldm_dir)


    # train
    generator.train()
    discriminator.train()
    for i in range(epochs):
        for iteration, face_dic in enumerate(loader):

            face_image = face_dic['face_img'].cuda()
            landmark_map = face_dic['landmark'].cuda()
            mask = face_dic['mask'].cuda()

            # zero optimizers
            gen_optimizer.zero_grad()
            dis_optimizer.zero_grad()

            # process outputs
            images_masked = (face_image * (1 - mask).float()) + mask
            inputs = torch.cat((images_masked, landmark_map), dim=1)
            scaled_masks_quarter = F.interpolate(mask, size=[int(mask.shape[2] / 4), int(mask.shape[3] / 4)],
                                                 mode='bilinear', align_corners=True)
            scaled_masks_half = F.interpolate(mask, size=[int(mask.shape[2] / 2), int(mask.shape[3] / 2)],
                                              mode='bilinear', align_corners=True)

            pre_face_image = generator(inputs, mask, scaled_masks_half, scaled_masks_quarter)

            # discriminator loss
            dis_input_real = face_image
            dis_input_fake = pre_face_image.detach()
            dis_real, _ = discriminator(torch.cat((dis_input_real, landmark_map), dim=1))
            dis_fake, _ = discriminator(torch.cat((dis_input_fake, landmark_map), dim=1))
            dis_real_loss = adversarial_loss(dis_real, True, True)
            dis_fake_loss = adversarial_loss(dis_fake, False, True)
            dis_loss = (dis_real_loss + dis_fake_loss) / 2

            # adversarial loss
            gen_input_fake = pre_face_image
            gen_fake, _ = discriminator(torch.cat((gen_input_fake, landmark_map), dim=1))  # in: [rgb(3)]
            gen_adv_loss = adversarial_loss(gen_fake, True, False) * adv_loss_weight

            # generator l1 loss
            gen_l1_loss = l1_loss(pre_face_image, face_image) * l1_loss_weight / torch.mean(mask)

            # generator perceptual loss
            gen_content_loss = perceptual_loss(pre_face_image, face_image) * content_loss_weight

            # generator style loss
            gen_style_loss = style_loss(pre_face_image * mask, face_image * mask) * style_loss_weight

            # generator tv loss
            gen_tv_loss = tv_loss(pre_face_image * mask + face_image * (1 - mask)) * tv_loss_weight


            gen_loss = gen_adv_loss + gen_l1_loss + gen_content_loss + gen_style_loss + gen_tv_loss

            # loss backward
            dis_loss.backward()
            gen_loss.backward()
            
            dis_optimizer.step()
            gen_optimizer.step()

            gen_loss_item = gen_loss.detach().cpu().item()
            dis_loss_item = dis_loss.detach().cpu().item()
            print('Epoch: (' + str(i) + ') ' + str(iteration) + ': gen loss: %.6f' % gen_loss_item + '  dis loss: %.6f' % dis_loss_item)

        if i > 0 and i % eval_interval == 0:
            eval_and_save_checkpoint(i, generator, eval_data, ckp_saving_dir, img_saving_dir)



if __name__ == '__main__':
    main()
