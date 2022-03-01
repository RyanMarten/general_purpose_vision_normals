import sys
sys.path.append('..')
sys.path.append('../..')

import torch
import torch.nn.functional as NF
import numpy as np
from tqdm import tqdm
from dataloader.Blended_MVS_loader import MyDataloader
from torchvision.utils import save_image

dev = torch.device('cuda:1')


# dev = torch.device('cpu')


def compute_camera_directions(K, shape):
    """
    Given width and height, creates a mesh grid, and returns homogeneous
    coordinates of image
    Arguments:
        K: tensor of shape 1x3x3
        shape: shape of the image [H, W]
    Returns:
        torch.Tensor -- H x W x 3, oriented in x, y, z order
    """
    # compute camera point from image / intrinsics
    H, W = shape
    dev = K.device
    img_xs = torch.linspace(0.5, W - 0.5, W).to(dev)
    img_ys = torch.linspace(0.5, H - 0.5, H).to(dev)
    img_y, img_x = torch.meshgrid(img_ys, img_xs)
    img_z = torch.ones_like(img_x).to(dev)
    img_pts = torch.stack((img_x, img_y, img_z), dim=-1)
    cam_pts = (K.inverse().view(1, 1, 3, 3) @ img_pts.unsqueeze(-1)).view(H, W, 3)
    return cam_pts


def compute_normal(depth, K):
    """
    Compute surface normal using finite difference method
    Args:
        depth: depth map (H, W)
        K: intrinsics of the image

    Returns:
        normals: calculated surface normal
    """
    H, W = depth.shape[-2:]
    cam_pts = compute_camera_directions(K, (H, W))

    # compute world point using depth
    world_pts = cam_pts * depth.unsqueeze(-1)

    # compute pdx / pdy using finite difference
    up = torch.zeros(H, W, 3)
    down = torch.zeros(H, W, 3)
    left = torch.zeros(H, W, 3)
    right = torch.zeros(H, W, 3)

    up[1:] = (world_pts[1:] - world_pts[:-1])
    left[:, 1:] = world_pts[:, 1:] - world_pts[:, :-1]
    down[:-1] = (world_pts[:-1] - world_pts[1:])
    right[:, :-1] = world_pts[:, :-1] - world_pts[:, 1:]

    # compute normal using cross product
    raw_normals_1 = torch.cross(up, left, dim=-1)
    raw_normals_2 = torch.cross(left, down, dim=-1)
    raw_normals_3 = torch.cross(down, right, dim=-1)
    raw_normals_4 = torch.cross(right, up, dim=-1)
    normals_1 = NF.normalize(raw_normals_1, p=2, dim=-1)
    normals_2 = NF.normalize(raw_normals_2, p=2, dim=-1)
    normals_3 = NF.normalize(raw_normals_3, p=2, dim=-1)
    normals_4 = NF.normalize(raw_normals_4, p=2, dim=-1)
    normals = normals_1 + normals_2 + normals_3 + normals_4
    normals = NF.normalize(normals, p=2, dim=-1)
    #     normals[...,1] = - normals[...,1]
    return normals


train_loader, val_loader = MyDataloader('train'), MyDataloader('val')

for i in tqdm(range(len(train_loader))):
    images, d_depth, intrinsics, image_dir = train_loader[i]
    gt_n = compute_normal(d_depth.to(dev), intrinsics.to(dev))
    store_path = image_dir.split('/')
    store_path[-2] = 'normals'
    a = store_path[-1]
    k = a.split('.')
    k[-1] = 'png'
    a = '.'.join(k)
    store_path[-1] = a
    store_path = '/'.join(store_path)
    gt_n = gt_n.permute(2, 0, 1) * 0.5 + 0.5
    # path = image_dir.split('/')
    save_image(gt_n, store_path)

for i in tqdm(range(len(val_loader))):
    images, d_depth, intrinsics, image_dir = val_loader[i]
    gt_n = compute_normal(d_depth.to(dev), intrinsics.to(dev))
    store_path = image_dir.split('/')
    store_path[-2] = 'normals'
    store_path = '/'.join(store_path)
    gt_n = gt_n.permute(2, 0, 1) * 0.5 + 0.5
    # path = image_dir.split('/')
    save_image(gt_n, store_path)
