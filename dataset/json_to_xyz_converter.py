# -*- coding: utf-8 -*-
# """
# json_to_xyz_converter.py
# Created on Oct Sept 30, 2024
# """

import os
import json
import numpy as np
import argparse

def convert_json_to_xyz(dataset_dir, output_dir):
    # Traverse the dataset directory to locate all dsm.json files
    for subdir, _, files in os.walk(dataset_dir):
        for file in files:
            if file == 'dsm.json':
                json_path = os.path.join(subdir, file)
                
                # Load JSON file
                with open(json_path, 'r') as f:
                    points = json.load(f)
                
                # Ensure output sub-folder matches the structure of dataset directory
                relative_path = os.path.relpath(subdir, dataset_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                
                # Save points in .xyz format
                xyz_path = os.path.join(output_subdir, 'points.xyz')
                np.savetxt(xyz_path, points, fmt='%.6f')
                
                print(f"Converted {json_path} to {xyz_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dsm.json files to .xyz format")
    parser.add_argument('--dataset_dir', type=str, required=True, help="Path to the dataset root directory")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory for .xyz files")

    args = parser.parse_args()
    
    convert_json_to_xyz(args.dataset_dir, args.output_dir)

# Usage:
# python ./dataset/json_to_xyz_converter.py --dataset_dir ./data_point2roof/Dataset_v1 --output_dir ./data_point2roof/points
