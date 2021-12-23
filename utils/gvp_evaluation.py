import torch
import math
from torch.nn.modules.module import Module

def get_rotation_matrix(v1, v2):
    '''
    Get the rotation matrix given two unit vector

    Arguments:
        v1: first vector
        v2: second vector

    Return:
        R: The rotation matrix from v1 to v2 (v2 = R @ v1)
    '''
    N = v1.shape[0]
    v = torch.cross(v1,v2, dim = -1) # N x 3
    s = v.float().norm(dim = -1) # N  
    c = (v1 * v2).sum(-1) # N 
    v_matrix = torch.zeros(N, 3, 3).to(v1.device) # N x 3 x 3
    v_matrix[:, 0, 1] = -v[:,2]
    v_matrix[:, 0, 2] = v[:,1]
    v_matrix[:, 1, 0] = v[:,2]
    v_matrix[:, 1, 2] = -v[:,0]
    v_matrix[:, 2, 0] = -v[:,1]
    v_matrix[:, 2, 1] = v[:,0]
    R = torch.eye(3).repeat(N, 1, 1).to(v1.device) + v_matrix + v_matrix @ v_matrix * (1 - c).view(N, 1, 1) / (s**2).view(N, 1, 1)
    return R

def ran_sec(predicted_normals, gt_normals, valid_mask, dev, iteration = 100, threshold = 10):
    '''
    Using RANSEC to calculate the rotation matrix that maximize the inliers between predicted normals and ground truth normals

    Arguments:
        valid_mask: where ground truth normals are valid
        dev: gpu devices
        iteration: RANSEC iteration number
        threshold: angle difference threshold for counting inliers

    Return:
        final_R: Selected rotation matrix
    '''
    # normals: 3 x H x W
    # random select X pixels
    valid_flat = valid_mask.view(-1).nonzero()
    predict_flat = predicted_normals.view(3, -1)[:,valid_flat[:,0]] # 3 x N
    gt_flat =  gt_normals.view(3, -1)[:,valid_flat[:,0]] # 3 x N
    length = len(valid_flat)
    inliers = []
    save_R = []
    for i in range(iteration):
        pixels = torch.randint(length, (1,)).to(dev) # M
        selected_predicted_normals = predict_flat[:,pixels.long()].permute(1,0)
        selected_gt_normals = gt_flat[:,pixels.long()].permute(1,0) # M x 3
        R = get_rotation_matrix(selected_predicted_normals,selected_gt_normals) # M x 3 x 3
        rotated_predicted = (R.unsqueeze(1) @ predict_flat.permute(1,0).unsqueeze(-1).unsqueeze(0))[...,0] # M x N x 3
        angle_diff = torch.acos((rotated_predicted * gt_flat.permute(1,0)).sum(-1).clamp(-1 + 1e-8, 1 - 1e-8)) * 180 / math.pi# M x N
        num_inlier = (angle_diff < threshold).sum(-1) # M
        save_R.append(R)
        inliers.append(num_inlier)
    # print(f'length of save_R: {len(save_R)}, length of inliers: {len(inliers)}')
    # print(f'shape of R: {save_R[0].shape}')
    # print(f'inliers max num: {torch.tensor(inliers).max()}')
    final_R = save_R[torch.tensor(inliers).argmax()]
    return final_R.to(dev)

def calculate_angle(predicted_noraml, gt_noraml, valid_mask):
    '''
    Calculate the metrics of surface noraml evaluation

    Return:
        Evaluation metrics
    '''
    predicted_noraml = predicted_noraml.view(3,-1)
    gt_noraml = gt_noraml.view(3,-1)
    valid_mask = valid_mask.view(-1)
    valid_index = valid_mask.nonzero() # N x 1
    valid_predicted_noraml = predicted_noraml[:,valid_index[:,0]]
    valid_gt_noraml = gt_noraml[:,valid_index[:,0]]    
        
    sum_of_dot = (valid_predicted_noraml * valid_gt_noraml).sum(0) # n
    sum_of_dot[sum_of_dot>1] = 1
    sum_of_dot[sum_of_dot<-1] = -1
    angle = torch.abs(torch.acos(sum_of_dot) / math.pi * 180)
    mean_angle = angle.mean()
    median_angle = angle.median()
    thres_1 = (angle<11.25).float().mean()
    thres_2 = (angle<22.5).float().mean()
    thres_3 = (angle<30).float().mean()
    
    return mean_angle,median_angle,thres_1,thres_2,thres_3

def sharp_normal_mask(normals, angle_thre = 40):
    '''
    Calculate the invalid mask where pixels'  surface normals have sharp changes. Typically used for BlendedMVS and DTU.

    Arguments:
        normals: ground truth normals
        angle_thre: the threshold to be considered as sharp change
    
    Return:
        invalid_mask: Invalid mask
    '''
    # For each pixel, if one of angle diff between it and its 
    # neighboring pixel is larger than threshold, it would be masked
    B, _, H, W = normals.shape
    dev = normals.device
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

    invalid_mask = up_down_mask.long() | left_right_mask.long()
    invalid_mask[invalid_mask > 1] = 1
    return invalid_mask

class dot_prod_loss(Module):
    """
    Binary Cross Entropy with ignore regions, not balanced.
    """

    def __init__(self):
        super(dot_prod_loss, self).__init__()

    def forward(self, input, gt):
        loss = (torch.abs(1 - (input * gt).sum(dim = 1))).mean()

        return loss