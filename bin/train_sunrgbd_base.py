import sys
import os
sys.path.append('.')
sys.path.append('..')

import glob
import os
from os import path
import numpy as np
import random
import time
from tqdm import tqdm

import matplotlib.pyplot as plt

import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.utils import data
from torchvision import models
from torchvision.transforms import ToTensor, Normalize, Resize
import torch.optim as optim
import h5py
import math

from dataloader.nyuv2_loader import NYUv2
from dataloader.sunrgbd_loader import SUNRGBD
import torch.nn.functional as NF
import time
import csv
import argparse

parser = argparse.ArgumentParser(description='Training hrnet')
# The locationi of training set
parser.add_argument('--dataRoot', default=None, help='path to images' )
parser.add_argument('--resume', type=str, default= None, help='Normal hrnet weight' )
parser.add_argument('--checkpoints', type=str, help='dir to save weight' )
parser.add_argument('--results_dir', type=str, help='dir to save results' )


# The basic training setting
parser.add_argument('--nepoch', type=int, default=80, help='the number of epochs for training' )
parser.add_argument('--batch_size', type=int, default=4, help='input batch size' )
parser.add_argument('--ori_hrnet', action='store_true', help='Use original hrnet resolution and upsample' )


# The loss parameters
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate' )
parser.add_argument('--n_loss', type=str, default='L1', help='Loss of normal' )
parser.add_argument('--decay_num', type=int, default=6000, help='iteration for each decay of lr' )
parser.add_argument('--decay_scale', type=float, default=0.9, help='scale for each decay of lr' )

# The gpu setting
parser.add_argument('--device', type=str, default='cuda', help='the devices used for training network' )

# The network setting 
# parser.add_argument('--network', type = str, default = 'high', help = 'Use baseline network or hrnet')

# Whether enforce rotation
parser.add_argument('--rotation', action='store_true', help = 'Whether enforce a rotation matrix' )

args = parser.parse_args()
print(args)

def get_rotation_matrix(v1, v2):
    N = v1.shape[0]
    v = torch.cross(v1,v2, dim = -1) # N x 3
    s = v.float().norm(dim = -1) # N  
    c = (v1 * v2).sum(-1) # N 
    v_matrix = torch.zeros(N, 3, 3) # N x 3 x 3
    v_matrix[:, 0, 1] = -v[:,2]
    v_matrix[:, 0, 2] = v[:,1]
    v_matrix[:, 1, 0] = v[:,2]
    v_matrix[:, 1, 2] = -v[:,0]
    v_matrix[:, 2, 0] = -v[:,1]
    v_matrix[:, 2, 1] = v[:,0]
    R = torch.eye(3).repeat(N, 1, 1) + v_matrix + v_matrix @ v_matrix * (1 - c).view(N, 1, 1) / (s**2).view(N, 1, 1)
    return R

def ran_sec(predicted_normals, gt_normals, valid_mask, iteration = 100, threshold = 20):
    # normals: 3 x H x W
    # random select X pixels
    valid_flat = valid_mask.view(-1).nonzero()
    predict_flat = predicted_normals.view(3, -1)[:,valid_flat[:,0]] # 3 x N
    gt_flat =  gt_normals.view(3, -1)[:,valid_flat[:,0]] # 3 x N
    length = len(valid_flat)
    pixels = torch.randint(length, (iteration,)) # M
    selected_predicted_normals = predict_flat[:,pixels.long()].permute(1,0)
    selected_gt_normals = gt_flat[:,pixels.long()].permute(1,0) # M x 3
    R = get_rotation_matrix(selected_predicted_normals,selected_gt_normals) # M x 3 x 3
    rotated_predicted = (R.unsqueeze(1) @ predict_flat.permute(1,0).unsqueeze(-1).unsqueeze(0))[...,0] # M x N x 3
    angle_diff = torch.acos((rotated_predicted * gt_flat.permute(1,0)).sum(-1).clamp(-1 + 1e-8, 1 - 1e-8)) * 180 / math.pi# M x N
    num_inlier = (angle_diff < threshold).sum(-1) # M
    final_R = R[num_inlier.argmax()]
    return final_R.to(dev)

def calculate_angle(valid_predicted_noraml,valid_gt_noraml, raw_depth):
    # input normals: 1 x 3 x H x W 
    valid_predicted_noraml = valid_predicted_noraml.view(3,-1)
    valid_gt_noraml = valid_gt_noraml.view(3,-1)
    raw_depth = raw_depth.view(-1)
    valid_index = raw_depth.nonzero() # N x 1
    valid_predicted_noraml = valid_predicted_noraml[:,valid_index[:,0]]
    valid_gt_noraml = valid_gt_noraml[:,valid_index[:,0]]    
        
    sum_of_dot = (valid_predicted_noraml * valid_gt_noraml).sum(0) # n
#     print(sum_of_dot.shape)
    sum_of_dot[sum_of_dot>1] = 1
    sum_of_dot[sum_of_dot<-1] = -1
    angle = torch.abs(torch.acos(sum_of_dot) / math.pi * 180)
#     print(angle)
    mean_angle = angle.mean()
    median_angle = angle.median()
    thres_1 = (angle<11.25).float().mean()
    thres_2 = (angle<22.5).float().mean()
    thres_3 = (angle<30).float().mean()
    
    return mean_angle,median_angle,thres_1,thres_2,thres_3

def sharp_normal_mask(normals, angle_thre = 40):
    # For each pixel, if one of angle diff between it and its 
    # neighboring pixel is larger than threshold, it would be masked
    # print(normals.shape)
    B, _, H, W = normals.shape
    generated_normal_right = normals[:,:,:,1:]
    generated_normal_left = normals[:,:,:,:-1]
    angle_diff_lr = torch.acos((generated_normal_right * generated_normal_left).sum(1)).unsqueeze(1) / math.pi * 180

    generated_normal_bot = normals[:,:,1:,:]
    generated_normal_up = normals[:,:,:-1,:]
    angle_diff_ud = torch.acos((generated_normal_bot * generated_normal_up).sum(1)).unsqueeze(1) / math.pi * 180
    
    mask_lr = angle_diff_lr > angle_thre # mask
    mask_ud = angle_diff_ud > angle_thre # mask
    
    
    up_down_mask = torch.zeros(B, 1, H, W).to(dev)
    up_down_mask[:,:,1:] = mask_ud 
    up_down_mask[:,:,:-1] = up_down_mask[:,:,:-1] + mask_ud 

    left_right_mask = torch.zeros(B, 1, H, W).to(dev)
    left_right_mask[:,:,:, 1:] = mask_lr
    left_right_mask[:,:,:, :-1] = left_right_mask[:,:,:, :-1] + mask_lr 

    mask = up_down_mask.long() | left_right_mask.long()
    mask[mask > 1] = 1
    return mask

def val_loss(index, n_model,val_loader, dataset = 'BMVS'):
    epoch_it = tqdm(total = len(val_loader))
    n_losses = []
    mean_angle_diff = []
    median_angle_diff = []
    thres_1 = [] # 11.25
    thres_2 = [] # 22.5
    thres_3 = [] # 30
    # b_losses = []
    n_model.eval()
    # b_model.eval()
    for i, data in enumerate(val_loader):
        images, raw_depth, normals = data['image'].to(dev), data['depth'].to(dev), data['normals'].to(dev)

        with torch.no_grad():
            predicted_normal = n_model(images)
            # predicted_boundary = b_model(images)
#         print(predicted_normal.shape)
        valid_mask = raw_depth > 0
        if do_rotation:
            R = ran_sec(predicted_normal.cpu(), normals.cpu(), valid_mask)
            predicted_normal = R.view(1,1,3,3) @ predicted_normal[0].permute(1,2,0).unsqueeze(-1)
            predicted_normal = predicted_normal[...,0].permute(2,0,1).unsqueeze(0)            
        valid_predicted_normal = predicted_normal * valid_mask
        valid_gt_normal = normals * valid_mask
        # b_loss = b_loss_func(predicted_boundary[:,0] * valid_mask,boundary * valid_mask)
#         n_loss = (torch.abs(1 - (predicted_normal * normals).sum(dim = 1))).mean()
        angle_result = calculate_angle(valid_predicted_normal, valid_gt_normal, valid_mask)
        mean_angle_diff.append(angle_result[0])
        median_angle_diff.append(angle_result[1])
        thres_1.append(angle_result[2])
        thres_2.append(angle_result[3])
        thres_3.append(angle_result[4])
        n_loss = n_loss_func(valid_gt_normal,valid_predicted_normal)
        # n_loss = (torch.abs(1 - (valid_gt_normal * valid_predicted_normal).sum(dim = 1))).mean()
        n_losses.append(n_loss)
        # b_losses.append(b_loss)
        epoch_it.set_description("n_loss: %.3f"%(n_loss))
        # epoch_it.set_description("n_loss: {}, b_loss: {}".format(n_loss, b_loss))
        epoch_it.update(1)
    epoch_it.close()    
    mean_n_loss = torch.Tensor(n_losses).mean()
    # mean_b_loss = torch.Tensor(b_losses).mean()
    print('validate loss: n_loss: {}, b_loss: {}'.format(mean_n_loss,0))
    print('Mean angle: {}\n Median angle: {}\n 11.25: {}%\n 22.5: {}%\n 30: {}%'\
          .format(torch.tensor(mean_angle_diff).mean(),\
                  torch.tensor(median_angle_diff).mean(),\
                  torch.tensor(thres_1).mean(),\
                  torch.tensor(thres_2).mean(),\
                  torch.tensor(thres_3).mean())\
         )
    # print('validate loss: n_loss: {}, b_loss: {}'.format(mean_n_loss,mean_b_loss))
    with open(val_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'num':index , 'dataset':dataset, 'mean_ang': float(torch.tensor(mean_angle_diff).mean()),
        'median_ang':float(torch.tensor(median_angle_diff).mean()),
        'thres_1':float(torch.tensor(thres_1).mean()),
        'thres_2':float(torch.tensor(thres_2).mean()), 
        'thres_3':float(torch.tensor(thres_3).mean()),
        'n_loss':float(mean_n_loss)})
    return mean_n_loss,0
    
class dot_prod_loss(Module):
    """
    Binary Cross Entropy with ignore regions, not balanced.
    """

    def __init__(self):
        super(dot_prod_loss, self).__init__()

    def forward(self, input, gt, valid_mask):
        diff = torch.abs(1 - (input * gt).sum(dim = 1))
        loss = diff[valid_mask[0]].mean()
        return loss

fieldnames = ['num', 'dataset', 'mean_ang', 'median_ang', 'thres_1', 'thres_2', 'thres_3', 'n_loss' , 'b_loss']

output_directory = path.join('results', args.results_dir)
save_dir = path.join('weights',args.checkpoints)
epoch = args.nepoch
batch_size = args.batch_size
lr = args.lr
dev = args.device
if args.n_loss == 'L1':
    n_loss_func = nn.L1Loss()
else:
    n_loss_func = dot_prod_loss()


if not os.path.exists(output_directory):
    os.makedirs(output_directory)
train_csv = os.path.join(output_directory, 'train_record.csv')
val_csv = os.path.join(output_directory, 'val_record.csv')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# b_model = BoundaryDetector().to(dev)
do_rotation = False
if args.rotation:
    do_rotation = True
from models.normal_old_estimator import NormalEstimator

if not args.resume:
    # print('need weight')
    # exit()
    n_model = NormalEstimator().to(dev)
else:
    print('resume weight')
    n_model = NormalEstimator(pretrained_weight = args.resume).to(dev)




train_loader, val_loader = SUNRGBD(batch_size = batch_size)
_, nyu_val_loader = NYUv2()

n_optim = optim.SGD(n_model.parameters(),lr = lr, momentum = 0.9)
# b_optim = optim.SGD(b_model.parameters(),lr = 1e-2, momentum = 0.9)
nscheduler = optim.lr_scheduler.StepLR(n_optim, args.decay_num, args.decay_scale)
# bscheduler = optim.lr_scheduler.StepLR(b_optim, 2000, 0.90)
# b_loss_func = nn.BCELoss()




    
best_n = 1
best_b = 1
for j in range(epoch):   
    if j > 0: 
        print('validate sunrgbd')
        val_loss(j, n_model,val_loader, dataset = 'sunrgbd')        
        print('validate nyu')
        val_loss(j, n_model,nyu_val_loader, dataset = 'nyu')
    if j == 0:
        val_n_loss  = 1
    if val_n_loss < best_n:
        best_n = val_n_loss
        if (j != 0):
            print('get best: {}'.format(j))
            torch.save(n_model.state_dict(), path.join(save_dir, 'best_normal.pt'))
    if j == args.nepoch - 1:
        torch.save(n_model.state_dict(), path.join(save_dir, 'final_normal_{}.pt'.format(j)))
    n_losses = []
    mean_angle_diff = []
    median_angle_diff = []
    thres_1 = [] # 11.25
    thres_2 = [] # 22.5
    thres_3 = [] # 30    
    # b_losses = []
    epoch_it = tqdm(total = len(train_loader))
    for i, data in enumerate(train_loader):
        images, raw_depth, normals = data['image'].to(dev).float(), data['depth'].to(dev).float(), data['normals'].to(dev).float()
        predicted_normal = n_model(images)
        valid_mask = (raw_depth > 0) 
        if (valid_mask>0).sum() == 0:
            continue
        if do_rotation:
            R = ran_sec(predicted_normal.detach().cpu(), normals.cpu(), valid_mask)
            predicted_normal = R.view(1,1,1,3,3) @ predicted_normal.permute(0,2,3,1).unsqueeze(-1) # N x H x W x 3 x 1
            predicted_normal = predicted_normal[...,0].permute(0,3,1,2) # N x 3 x H x W 
            # rendered = (final_R.to(dev).view(1, 1, 3, 3) @ predicted_normals_3.permute(1,2,0).unsqueeze(-1))[...,0]
        valid_predicted_normal = predicted_normal * valid_mask
        valid_gt_noraml = normals * valid_mask
        n_loss = n_loss_func(valid_predicted_normal, valid_gt_noraml)
        angle_result = calculate_angle(valid_predicted_normal.detach().contiguous(), valid_gt_noraml.contiguous(), valid_mask.contiguous())
        mean_angle_diff.append(angle_result[0])
        median_angle_diff.append(angle_result[1])
        thres_1.append(angle_result[2])
        thres_2.append(angle_result[3])
        thres_3.append(angle_result[4])        
        # b_loss.backward()
        n_loss.backward()
        # b_optim.step()
        n_optim.step()
        # b_optim.zero_grad()
        n_optim.zero_grad()
        n_losses.append(n_loss.detach())
        # b_losses.append(b_loss.detach())
        lr = n_optim.param_groups[0]['lr']
        epoch_it.set_description("n_loss: %.3f, lr: %.5f"%(n_loss, lr))
    #     if (((epoch % 100) == 0) and (epoch != 0)):
        nscheduler.step()
        # bscheduler.step()
        epoch_it.update(1)
        n_losses.append(n_loss.detach())
        # b_losses.append(b_loss.detach())
    mean_n_loss = torch.Tensor(n_losses).mean()
    # mean_b_loss = torch.Tensor(b_losses).mean()        
    print('Training loss: n_loss: {}, b_loss: {}'.format(mean_n_loss,0))
    print('Mean angle: {}\n Median angle: {}\n 11.25: {}%\n 22.5: {}%\n 30: {}%'\
          .format(torch.tensor(mean_angle_diff).mean(),\
                  torch.tensor(median_angle_diff).mean(),\
                  torch.tensor(thres_1).mean(),\
                  torch.tensor(thres_2).mean(),\
                  torch.tensor(thres_3).mean())\
         )
    # print('validate loss: n_loss: {}, b_loss: {}'.format(mean_n_loss,mean_b_loss))
    with open(train_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'num':j, 'mean_ang': float(torch.tensor(mean_angle_diff).mean()),
        'median_ang':float(torch.tensor(median_angle_diff).mean()),
        'thres_1':float(torch.tensor(thres_1).mean()),
        'thres_2':float(torch.tensor(thres_2).mean()), 
        'thres_3':float(torch.tensor(thres_3).mean()),
        'n_loss':float(mean_n_loss)})
    # print('{}: training loss: n_loss: {}, b_loss: {}'.format(j, mean_n_loss,mean_b_loss))
    epoch_it.close()    
    
