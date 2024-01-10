# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 12:59:48 2022

@author: 12593
"""

import numpy as np
#import nibabel as nb
import cv2
import os
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
join = os.path.join
basename = os.path.basename
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--gt_path',
    type=str,
    default=''
)
parser.add_argument(
    '--seg_path',
    type=str,
    default=''
)
parser.add_argument(
    '--save_path',
    type=str,
    default=''
)

args = parser.parse_args()

gt_path = args.gt_path
seg_path = args.seg_path
save_path = args.save_path

filenames = os.listdir(seg_path)
filenames = [x for x in filenames if x.endswith('.png')]
filenames = [x for x in filenames if os.path.exists(join(seg_path, x))]
filenames.sort()

seg_metrics = OrderedDict(
    Name = list(),
    DSC = list(),
)

for name in tqdm(filenames):
    seg_metrics['Name'].append(name)

    gt_mask = cv2.imread(join(gt_path, name), cv2.IMREAD_UNCHANGED)
    seg_mask = cv2.imread(join(seg_path, name), cv2.IMREAD_UNCHANGED)
    case_spacing = [1,1,1]
    gt_data = np.uint8(gt_mask)
    seg_data = np.uint8(seg_mask)

    gt_labels = np.unique(gt_data)[1:]
    seg_labels = np.unique(seg_data)[1:]
    labels = np.union1d(gt_labels, seg_labels)

    assert len(labels) > 0, 'Ground truth mask max: {}'.format(gt_data.max())

    #DSC_arr = np.zeros(len(labels))
    DSC_arr = []
    for i in labels:
        if np.sum(gt_data==i)==0 and np.sum(seg_data==i)==0:
            DSC_i = 1
            NSD_i = 1
        elif np.sum(gt_data==i)==0 and np.sum(seg_data==i)>0:
            DSC_i = 0
            NSD_i = 0
        else:
            tool_i_gt, tool_i_seg = gt_data==i, seg_data==i
            DSC_i = compute_dice_coefficient(tool_i_gt, tool_i_seg)

        DSC_arr.append(DSC_i)

    DSC = np.mean(DSC_arr)
    seg_metrics['DSC'].append(round(DSC, 4))

dataframe = pd.DataFrame(seg_metrics)
#dataframe.to_csv(seg_path + '_DSC.csv', index=False)
dataframe.to_csv(save_path, index=False)

case_avg_DSC = dataframe.mean(axis=0, numeric_only=True)
print(20 * '>')
print(f'Average DSC for {basename(seg_path)}: {case_avg_DSC.mean()}')
print(20 * '<')