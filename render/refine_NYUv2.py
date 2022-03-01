import sys

sys.path.append('.')
sys.path.append('..')

import torch
import torch.nn.functional as NF
from torchvision.utils import save_image

import numpy as np
import h5py
from dataloader.general_nyu_loader_copy import Train_loader


def smooth(images, normals, depth, instance_segment, i, path, sigma=None, block_widths=20):
    """
    Use bilateral filter with instance segment, rgb color, coordinate distance, and normal to acquire better normals
    Args:
        images: RGB image
        normals: original surface normals
        depth: raw depth
        instance_segment: instance segment
        i: index
        path: saved path
        sigma: params of the weight

    Returns:

    """
    if sigma is None:
        sigma = [20, 0.1, 0.25, 0.005]

    instance_segment_flat = instance_segment.reshape(-1)
    images_flat = images.transpose((1, 2, 0)).reshape((-1, 3))
    normals_flat = normals.transpose((1, 2, 0)).reshape((-1, 3))
    depth_flat = depth.reshape(-1)

    H, W = depth.shape[-2:]
    w, h = np.meshgrid(np.arange(0, W), np.arange(0, H))
    h, w = h.reshape(-1), w.reshape(-1)
    N = H * W
    Z = depth.reshape(-1)
    new_normals = np.zeros((H * W, 3))

    # get the relative index of surrounding points (different from rendering!)
    block_widths = np.arange(-block_widths, block_widths)
    nw, nh = np.meshgrid(block_widths, block_widths)
    nh, nw = nh.reshape(-1), nw.reshape(-1)

    for k in range(N):
        if Z[k] == 0:
            continue
        h2 = h[k] + nh
        w2 = w[k] + nw

        # exclude invalid surrounding points
        valid_index = ((h2 >= 0) & (h2 < H) & (w2 >= 0) & (w2 < W)).nonzero()
        h2 = h2[valid_index[0]]
        w2 = w2[valid_index[0]]
        index2 = w2 + h2 * W
        valid_index = (depth_flat[index2] > 0).nonzero()
        h2 = h2[valid_index[0]]
        w2 = w2[valid_index[0]]
        index2 = w2 + h2 * W

        # Ensure it is not in boundary because boundary doesn't have segment info
        if instance_segment_flat[k] != 0:
            combine_segment = instance_segment_flat[index2]
            valid_index = (combine_segment == instance_segment_flat[k]).nonzero()
            h2 = h2[valid_index[0]]
            w2 = w2[valid_index[0]]
            index2 = w2 + h2 * W

        # calculate new normals using bilateral filter
        rgb_weight = ((images_flat[index2] - images_flat[k]) ** 2).sum(-1) / sigma[1] ** 2
        coor_weight = ((h2 - h[k]) ** 2 + (w2 - w[k]) ** 2) / sigma[0] ** 2
        normal_weight = ((normals_flat[index2] - normals_flat[k]) ** 2).sum(-1) / sigma[2] ** 2

        weight = (- coor_weight - rgb_weight - normal_weight)
        weight = np.exp(weight)
        weight = weight.reshape(len(weight), 1)
        new_normals[k] = (weight * normals_flat[index2]).sum(0)

    # normalizing and turn back to tensor
    new_normals = new_normals.reshape(H, W, 3)
    new_normals = NF.normalize(torch.Tensor(new_normals).permute(2, 0, 1), p=2, dim=0)

    # save normals
    new_normals = new_normals * 0.5 + 0.5
    save_image(new_normals, path)
    print('finish {}'.format(i))

    return


def main():
    intrinsics = torch.Tensor([519, 0, 320,
                               0, 519, 240,
                               0, 0, 1]).reshape(3, 3)
    path = '/mnt/data/sparse2dense/data/nyudepth_hdf5/nyu_depth_v2_labeled.mat'
    f = h5py.File(path, mode='r')
    total_instance = np.array(f['instances'])
    total_semantics = np.array(f['labels'])

    train, val = Train_loader(batch_size=1)

    val_match = torch.load('../../quan/sparse2dense/match.pt')
    train_match = torch.load('../../quan/sparse2dense/train_match.pt')
    train_data_path = '/mnt/data/sparse2dense/data/train_nyu/train/smoothed_normal/'
    val_data_path = '/mnt/data/sparse2dense/data/train_nyu/val/smoothed_normal/'

    arg = (sys.argv)
    i = int((arg[1]))

    dataset = train.dataset
    path = train_data_path
    match = train_match

    if i >= len(train) + len(val):
        print('finish! Upper index')
        return

    if i >= len(train):
        i = i - len(train)
        print('i change to {}'.format(i))
        dataset = val.dataset
        path = val_data_path
        match = val_match

    images, _, _, raw_depth, ori_normals = dataset[i]
    images, raw_depth, ori_normals = \
        images.numpy(), \
        raw_depth.numpy(), ori_normals.numpy()

    # get the instance segmentation from the file by combining semantic and instance
    semantic = total_semantics[match[i]].transpose(1, 0)
    instance = total_instance[match[i]].transpose(1, 0)
    for_source_instance = instance / instance.max() * 255.
    for_source_instance = for_source_instance.astype(np.uint8)
    for_source_semantic = semantic / semantic.max() * 255.
    for_source_semantic = for_source_semantic.astype(np.uint8)
    instance_segment = (for_source_semantic * for_source_instance)  # the instance segmentation

    path = (path + '%05d.png' % i)
    smooth(images, ori_normals, raw_depth, instance_segment, i, intrinsics, path)


if __name__ == '__main__':
    main()
