# -*- coding: utf-8 -*-
# """
# inference.py
# Created on Oct Sept 02, 2024
# """

import os
import torch
import argparse
import datetime
from model.roofnet import RoofNet
from utils import common_utils, model_utils

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True, help='Path to point cloud file (x, y, z)')
    parser.add_argument('--cfg_file', type=str, default='./model_cfg.yaml', help='Path to model config file')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID for inference')
    parser.add_argument('--output_dir', type=str, default='./inference_output', help='Directory to save results')
    parser.add_argument('--ckpt_file', type=str, default='./checkpoint_epoch_90.pth', help='Path to the checkpoint file')

    args = parser.parse_args()
    cfg = common_utils.cfg_from_yaml_file(args.cfg_file)
    return args, cfg

def main():
    args, cfg = parse_config()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Ensure output directory exists
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(output_dir, 'inference_log.txt')
    logger = common_utils.create_logger(log_file)
    logger.info('**********************Start Inference**********************')

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    common_utils.log_config_to_file(cfg, logger=logger)

    # Load point cloud file
    points = load_point_cloud(args.file_path)

    # Load model
    net = RoofNet(cfg.MODEL)
    net.use_edge = True  # Enable edge processing for inference
    net.cuda()
    net.eval()
    
    # Load checkpoint
    if not os.path.exists(args.ckpt_file):
        logger.error(f"Checkpoint file {args.ckpt_file} does not exist.")
        return
    model_utils.load_params(net, args.ckpt_file, logger=logger)

    # Run inference
    logger.info('**********************Running Inference**********************')
    with torch.no_grad():
        points = points.cuda()  # Send data to GPU
        
        # Prepare batch_dict for RoofNet
        batch_dict = {'points': points}  # Include other keys if required by PointNet2 or submodules
        
        # Run the model
        output = net(batch_dict)
        
        # Log output structure for debugging
        logger.info("Model Output Structure:")
        for key, value in output.items():
            logger.info(f"{key}: Shape - {value.shape}, Type - {type(value)}")

        # Post-process and save results in .obj format
        obj_path = os.path.join(output_dir, 'reconstructed_model.obj')
        save_as_obj(output, obj_path)
        logger.info(f'Saved inference result to {obj_path}')

    logger.info('**********************Inference Completed**********************')

def load_point_cloud(file_path):
    import numpy as np
    points = np.loadtxt(file_path)
    return torch.tensor(points, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

def save_as_obj(tensor_output, obj_file_path):
    with open(obj_file_path, 'w') as f:
        for vertex in tensor_output.get('vertices', []):  # Assuming output includes vertices
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in tensor_output.get('faces', []):  # Assuming output includes faces
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")  # 1-based indexing for .obj format

if __name__ == '__main__':
    main()