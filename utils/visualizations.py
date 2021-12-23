import torch.nn.functional as NF
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import torch
import math

def save_images(images, pred_normals, gt_normals, mask, save_dir, dataset, i):
    '''
    Save visualization of color images, surface normals, gt_normal, and error mapping.
    '''
    dataset_dir = os.path.join(save_dir, dataset)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    plt.imsave(os.path.join(dataset_dir,'{}_gt.png'.format(i) ),gt_normals[0].cpu().permute(1,2,0).numpy() * 0.5 + 0.5)
    plt.imsave(os.path.join(dataset_dir,'{}_pred.png'.format(i) ),pred_normals[0].cpu().permute(1,2,0).numpy() * 0.5 + 0.5)
    plt.imsave(os.path.join(dataset_dir,'{}_rgb.png'.format(i) ),images[0].cpu().permute(1,2,0).numpy())
    # plt.imsave(os.path.join(dataset_dir,'{}_depth.png'.format(i) ),raw_depth[0].cpu())
    plt.imsave(os.path.join(dataset_dir,'{}_mask.png'.format(i) ),mask[0].cpu())
    angle_diff = torch.abs(torch.acos((pred_normals * gt_normals).sum(1)) / math.pi * 180)
    angle_diff[mask == 0] = 0
    differece_layer = torch.zeros_like(angle_diff)
    differece_layer[angle_diff < 30] = 4
    differece_layer[angle_diff < 22.5] = 2
    differece_layer[angle_diff < 11.25] = 1
    differece_layer[angle_diff >= 30] = 5
    differece_layer[angle_diff  == 0] = 0 
    plt.imsave(os.path.join(dataset_dir,'{}_angle_diff.png'.format(i) ),angle_diff[0].cpu(), cmap = 'gray', vmin = 0, vmax = 50)
    plt.imsave(os.path.join(dataset_dir,'{}_inliers.png'.format(i) ),differece_layer[0].cpu(), vmin = 0, vmax = 5)    

def to_depth_image(depth, mask=None, ranges=None):
    '''
    Given HxW depth image, returns nice visualization of the depth image
    '''
    d = depth.detach().cpu().unsqueeze(-1).float()
    if ((d > 0).sum() == 0):
        return d.repeat(1, 1, 3).numpy().astype('uint8').transpose(2, 0, 1)

    if not (ranges):
        min_d = d[d > 0].min().numpy()
        max_d = d.max().numpy()
    else:
        d[d>0].clamp_(ranges[0].cpu(), ranges[1].cpu())
        min_d = ranges[0].cpu().numpy()
        max_d = ranges[1].cpu().numpy()
    if mask is None:
        m = (d == 0).numpy()
    else:
        m = mask.cpu().unsqueeze(-1).numpy()
    d = d.numpy()

    depth_image = (((max_d - d) / (max_d - min_d)) * 255).astype('uint8')
    depth_colored_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_PARULA)
    depth_colored_image[(m).repeat(3, 2)] = 0

    return depth_colored_image[:, :, ::-1].transpose(2, 0, 1)

def to_normal_image(normal):
    '''
    Given 3xHxW normal image, returns nice visualization of the normal image
    '''
    normal_image = normal.float().clamp(-1, 1).detach().cpu().numpy()
    nn = (normal_image + 1) / 2
    nn[normal_image == 0] = 0
    return nn
