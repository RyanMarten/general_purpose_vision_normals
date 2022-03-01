import sys
sys.path.append('.')
sys.path.append('..')

import torch
import torch.nn.functional as NF
from torchvision.utils import save_image

import numpy as np
from dataloader.general_nyu_loader import Train_loader


def camera_directions(K, shape, offset=0.5):
    """
    Given width and height, creates a mesh grid, and returns homogeneous
    coordinates of image
    Arguments:
        K: tensor of shape 1x3x3
    Returns:
        torch.Tensor -- 1x3xHxW, oriented in x, y, z order
    """
    dev = K.device
    H, W = shape
    O = offset
    x_coords = torch.linspace(O, W - 1 + O, W, device=dev)
    y_coords = torch.linspace(O, H - 1 + O, H, device=dev)

    # HxW grids
    ys, xs = torch.meshgrid([y_coords, x_coords])
    zs = torch.ones_like(xs)

    # HxWx3
    img_coords = torch.stack((xs, ys, zs), dim=-1)

    # HxWx3x1 => 3xHxW
    cam_d = K.inverse().view(1, 1, 3, 3) @ img_coords.unsqueeze(-1)
    return cam_d[..., 0].permute(2, 0, 1)


def compute_normal_nyu(d_depth, path, i, intrinsics, depth_threshold=0.05):
    """
    Compute surface normal using NYUv2 method:
        1. project pixels in images into point clouds
        2. for each pixel, get multiple surrounding points within depth threshold, calculate the
           surface normal by solving equation AX + BY + CZ + D  = 0
    Because basic calculation (i.e., svd solver) in PyTorch is slower than Numpy, this function would
    first turn tensor into numpy for calculate, and turn numpy back for saving.
    Args:
        d_depth: raw depth of NYUv2
        path: saving path of the generated surface normal
        i: index of images, used for reporting
        intrinsics: intrinsics of the camera
        depth_threshold: the threshold used to filter out outliers inside one window
    """
    H, W = d_depth.shape[-2:]
    shape = d_depth.shape[-2:]
    c_coor = camera_directions(intrinsics, shape).numpy()
    point_cloud = c_coor.transpose(1, 2, 0) * d_depth.reshape(H, W, 1)  # get the point cloud of image points

    # concatenate the image points with ones, which would be used for solving AX + BY + CZ + D = 0
    point_cloud_add = np.concatenate((point_cloud, np.ones((H, W, 1))), axis=-1).reshape(-1, 4)

    w, h = np.meshgrid(np.arange(0, W), np.arange(0, H))
    h, w = h.reshape(-1), w.reshape(-1)
    N = H * W
    Z = d_depth.reshape(-1)

    # get the relative index of surrounding points
    block_widths = torch.tensor([-9, -6, -3, -1, 0, 1, 3, 6, 9])
    nh, nw = torch.meshgrid(block_widths, block_widths)
    nh, nw = nh.numpy().astype(np.int64), nw.numpy().astype(np.int64)
    nh, nw = nh.reshape(-1), nw.reshape(-1)

    generated_normal = np.zeros((H * W, 3))  # tensor for generated normals
    for k in range(N):
        if Z[k] == 0:
            continue
        h2 = h[k] + nh  # surrounding points' locations' height
        w2 = w[k] + nw  # surrounding points' locations' width

        # exclude invalid surrounding points
        valid_index = ((h2 >= 0) & (h2 < H) & (w2 >= 0) & (w2 < W)).nonzero()
        h2 = h2[valid_index[0]]
        w2 = w2[valid_index[0]]
        index2 = w2 + h2 * W
        valid_index = (np.abs(Z[index2] - Z[k]) < Z[k] * depth_threshold).nonzero()
        h2 = h2[valid_index[0]]
        w2 = w2[valid_index[0]]
        index2 = w2 + h2 * W

        if len(h2) < 3:
            continue

        # solve for surface normal
        A = point_cloud_add[index2]  # n x 4
        u_solution, s_solution, v_solution = np.linalg.svd(A)  # v: 4 x 4
        solution = v_solution[-1]  # solution: 4 x 1
        normal = solution[:-1] * solution[-1] / abs(solution[-1])  # get correct direction
        generated_normal[k] = normal

    # normalize and turn numpy back into tensor
    generated_normal = NF.normalize(torch.Tensor(generated_normal).reshape(H, W, 3), p=2, dim=-1)

    # save normals
    normals = generated_normal * 0.5 + 0.5
    normals = normals.permute(2, 0, 1)
    save_image(normals, path)
    print('finish {}'.format(i))

    return


def main():
    intrinsics = torch.Tensor([519, 0, 320,
                               0, 519, 240,
                               0, 0, 1]).reshape(3, 3)
    arg = sys.argv
    i = int((arg[1]))

    if i >= 795 + 654:
        print('finish! Upper index')
        return

    train, val = Train_loader(batch_size=1)

    train_data_path = '/mnt/data/sparse2dense/data/train_nyu/train/ori_normals/'
    val_data_path = '/mnt/data/sparse2dense/data/train_nyu/val/ori_normals/'

    dataset = train.dataset
    path = train_data_path

    if i >= len(train):
        i = i - len(train)
        print('i change to {}'.format(i))
        dataset = val.dataset
        path = val_data_path

    raw_depth, d_depth = dataset[i][3], dataset[i][1]
    path = (path + '%05d.png' % i)
    compute_normal_nyu(raw_depth.numpy(), path, i, intrinsics, d_depth.numpy())


if __name__ == '__main__':
    main()

