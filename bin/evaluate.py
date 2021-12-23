import sys
import os
sys.path.append('.')
sys.path.append('..')

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from utils.gvp_evaluation import ran_sec, calculate_angle, sharp_normal_mask
from utils.visualizations import save_images
import datasets as Datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as NF


def val_loss(n_model, dataset_list, args):
    '''
    Validate the loss of the networks on each dataset, and save the results as a file
    params: 
        n_model: network to be evaluated (could be a warpped function if network need special operation)
        dataset_list: the list of evaluation datasets
    '''
    for dataset in dataset_list:
        print(f'Evaluating {dataset}...')
        val_dataset = getattr(Datasets, dataset)(root = args.data_dir, mode = 'val')
        val_dataset.concatenate_dataset()
        val_loader = DataLoader(val_dataset, batch_size = 1, num_workers = 8)
        epoch_it = tqdm(total = len(val_loader))
        n_losses = []
        mean_angle_diff = []
        median_angle_diff = []
        thres_1 = [] # 11.25
        thres_2 = [] # 22.5
        thres_3 = [] # 30
        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)        
        for i, data in enumerate(val_loader):
            images, normals = data['image'], data['normals']
            images, normals = images.to(dev), normals.to(dev)  
            valid_mask = ((normals **2).sum(dim = 1) > 0.8) # Remove invalid pixels of normals

            if dataset == 'BlendedMVS' or dataset == 'DTU':
                angle_sharp_mask =  sharp_normal_mask(normals)
                valid_mask = valid_mask & (angle_sharp_mask[0] == 0)                

            normals = NF.normalize(normals, p = 2, dim = 1) # normalize it after getting the mask
            with torch.no_grad():
                predicted_normal = n_model(images)

            if args.rotation: # Get maximum rotation
                R = ran_sec(predicted_normal, normals, valid_mask, dev)
                predicted_normal = R.view(1,1,3,3) @ predicted_normal[0].permute(1,2,0).unsqueeze(-1)
                predicted_normal = predicted_normal[...,0].permute(2,0,1).unsqueeze(0) 

            if i % 100 == 0:
                save_images(images, predicted_normal, normals, valid_mask, args.results_dir, dataset, i)

            # Get valid pixels
            valid_predicted_normal = predicted_normal * valid_mask
            valid_gt_normal = normals * valid_mask
            angle_result = calculate_angle(valid_predicted_normal, valid_gt_normal, valid_mask)

            mean_angle_diff.append(angle_result[0])
            median_angle_diff.append(angle_result[1])
            thres_1.append(angle_result[2])
            thres_2.append(angle_result[3])
            thres_3.append(angle_result[4])
            n_loss = (torch.abs(1 - (valid_gt_normal * valid_predicted_normal).sum(dim = 1))).mean()
            n_losses.append(n_loss)
            epoch_it.set_description("n_loss: %.3f"%(n_loss))
            epoch_it.update(1)
        epoch_it.close()    

        # Save results
        logs = {}
        logs['Mean angle'] = torch.tensor(mean_angle_diff).mean()
        logs['Median angle'] = torch.tensor(median_angle_diff).mean()
        logs['inliers percentage of 11.25'] = torch.tensor(thres_1).mean()
        logs['inliers percentage of 22.5'] = torch.tensor(thres_2).mean()
        logs['inliers percentage of 30'] = torch.tensor(thres_3).mean()
        with open(args.log_path, 'a') as f:
                f.write('=' * 40 + '\n')
                f.write(dataset +'\n')
                f.write('=' * 40 + '\n')
                print('='*40)
                print(dataset)
                print('='*40)
                for k, v in logs.items():       
                    f.write(f'{k}: {v}' + '\n')
                    print(f'{k}: {v}')


if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Evaluate Normal Estimation')
    parser.add_argument('--data_dir', type=str, default = '/data/yuqunwu2/general_purpose_vision', help='dir of datasets' )
    parser.add_argument('--results_dir', type=str, help='dir to save images')
    parser.add_argument('--log_path', type=str, help='dir to save evaluation results')
    parser.add_argument('--dataset', type=str, nargs = '+', default = None, help='tested datasets' )
    parser.add_argument('--device', type=str, default='cuda', help='the devices used for evaluation' )
    parser.add_argument('--rotation', action='store_true', help = 'Whether enforce a rotation matrix' )

    # Added ones
    parser.add_argument('--architecture', default = 'BN', type=str, help='{BN, GN}')
    parser.add_argument("--pretrained", default = 'scannet', type=str, help="{nyu, scannet}")
    parser.add_argument('--sampling_ratio', type=float, default=0.4)
    parser.add_argument('--importance_ratio', type=float, default=0.7)

    args = parser.parse_args()
    print(args)

    dev = args.device

    if not args.dataset:
        dataset_list = ['NYUv2', 'BlendedMVS', 'DTU', 'SUNRGBD']
    else:
        dataset_list = args.dataset

    #########################################################################################################
    # Get network, need manual modification

    # from model.normal_old_estimator import NormalEstimator
    # n_model = NormalEstimator(pretrained_weight = 'weights/ori_weight/normal_scannet.pth').to(dev)
    # n_model.eval()
    
    sys.path.append('../surface_normal_uncertainty')
    from models.NNET import NNET
    from utils import utils
    checkpoint = '../surface_normal_uncertainty/checkpoints/%s.pt' % args.pretrained
    model = NNET(args).to(dev)
    model = utils.load_checkpoint(checkpoint, model)
    model.eval()

    def n_model(images):
        transformation = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalized_image = transformation(images)
        with torch.no_grad():
            output, _, _ = model(normalized_image)
        norm_out = output[-1]
        pred_normals = norm_out[:, :3, :, :]
        pred_normals = - pred_normals
        return pred_normals
    #########################################################################################################

    # Evaluate
    val_loss(n_model, dataset_list, args)
    print('finish evaluation')