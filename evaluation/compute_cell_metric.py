"""
Created on Thu Mar 31 18:10:52 2022
adapted form https://github.com/stardist/stardist/blob/master/stardist/matching.py
Thanks the authors of Stardist for sharing the great code

"""

import argparse
import numpy as np
from numba import jit
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
import pandas as pd
from skimage import segmentation, io, measure
import tifffile as tif
import os
join = os.path.join
from tqdm import tqdm
import traceback

def _intersection_over_union(masks_true, masks_pred):
    """ intersection over union of all mask pairs
    
    Parameters
    ------------
    
    masks_true: ND-array, int 
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels
    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou

@jit(nopython=True)
def _label_overlap(x, y):
    """ fast function to get pixel overlaps between masks in x and y 
    
    Parameters
    ------------

    x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
        where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    overlap: ND-array, int
        matrix of pixel overlaps of size [x.max()+1, y.max()+1]
    
    """
    x = x.ravel()
    y = y.ravel()
    
    # preallocate a 'contact map' matrix
    overlap = np.zeros((1+x.max(),1+y.max()), dtype=np.uint)
    
    # loop over the labels in x and add to the corresponding
    # overlap entry. If label A in x and label B in y share P
    # pixels, then the resulting overlap is P
    # len(x)=len(y), the number of pixels in the whole image 
    for i in range(len(x)):
        overlap[x[i],y[i]] += 1
    return overlap

def dice(gt, seg):
    if np.count_nonzero(gt)==0 and np.count_nonzero(seg)==0:
        dice_score = 1.0
    elif np.count_nonzero(gt)==0 and np.count_nonzero(seg)>0:
        dice_score = 0.0
    else:
        union = np.count_nonzero(np.logical_and(gt, seg))
        intersection = np.count_nonzero(gt) + np.count_nonzero(seg)
        dice_score = 2*union/intersection
    return dice_score

def _true_positive(iou, th):
    """ true positive at threshold th
    
    Parameters
    ------------

    iou: float, ND-array
        array of IOU pairs
    th: float
        threshold on IOU for positive label

    Returns
    ------------

    tp: float
        number of true positives at threshold
    """
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2*n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    return tp

def eval_tp_fp_fn(masks_true, masks_pred, threshold=0.5):
    num_inst_gt = np.max(masks_true)
    num_inst_seg = np.max(masks_pred)
    if num_inst_seg>0:
        iou = _intersection_over_union(masks_true, masks_pred)[1:, 1:]
            # for k,th in enumerate(threshold):
        tp = _true_positive(iou, threshold)
        fp = num_inst_seg - tp
        fn = num_inst_gt - tp
    else:
        # print('No segmentation results!')
        tp = 0
        fp = 0
        fn = 0
        
    return tp, fp, fn

def remove_boundary_cells(mask):
    "We do not consider boundary cells during evaluation"
    W, H = mask.shape
    bd = np.ones((W, H))
    bd[2:W-2, 2:H-2] = 0
    bd_cells = np.unique(mask*bd)
    for i in bd_cells[1:]:
        mask[mask==i] = 0
    new_label,_,_ = segmentation.relabel_sequential(mask)
    return new_label

# def main():
parser = argparse.ArgumentParser('Compute F1 score for cell segmentation results', add_help=False)
# Dataset parameters
parser.add_argument('-g', '--gt_path', default='cellTs/inst-labelsVal', type=str, help='path to ground truth')
parser.add_argument('-s', '--seg_path', type=str, default='cellTs/nnunet_2d', help='path to segmentation results; file names are the same as ground truth', required=False)
parser.add_argument('-thre', '--thresholds', nargs='+', default=[0.5], type=float, help='threshold to count correct cells')
parser.add_argument('-o', '--output_path', default='./cellTs', type=str, help='path where to save metrics')
parser.add_argument('-n', '--save_name', default='unet2d', type=str, help='name of the csv file')
parser.add_argument('--count_bd_cells', default=False, action='store_true', required=False, help='remove the boundary cells when computing metrics by default')
args = parser.parse_args()

gt_path = args.gt_path
seg_path = args.seg_path

names = sorted(os.listdir(seg_path))
names = [i for i in names if i.endswith('.png')]
names = [i for i in names if os.path.isfile(join(gt_path, i.split('.png')[0]+'_label.tiff'))]
print('num of files:', len(names))

for threshold in args.thresholds:
    print('compute metrics at threshold:', threshold)
    metrics = OrderedDict()
    metrics['names'] = []
    metrics['true_num'] = []
    metrics['pred_num'] = []
    metrics['correct_num(TP)'] = []
    metrics['missed_num(FN)'] = []
    metrics['wrong_num(FP)'] = []
    metrics['precision'] = []
    metrics['recall'] = []
    metrics['dice'] = []
    metrics['F1'] = []
    failed = []
    for name in tqdm(names):
        try:
            gt = tif.imread(join(gt_path, name.split('.png')[0]+'_label.tiff'))
            seg = io.imread(join(seg_path, name))
            seg = measure.label(seg==1)
            dice_score = dice(gt>0, seg>0)
            # Score the cases
            # do not consider cells on the boundaries during evaluation
            if np.prod(gt.shape)<25000000:
                if not args.count_bd_cells:
                    gt = remove_boundary_cells(gt.astype(np.int32)) 
                    seg = remove_boundary_cells(seg.astype(np.int32))    
                gt, _, _ = segmentation.relabel_sequential(gt)
                seg, _, _ = segmentation.relabel_sequential(seg)
                cell_true_num = np.max(gt)
                cell_pred_num = np.max(seg)
                tp, fp, fn = eval_tp_fp_fn(gt, seg, threshold=threshold)
            else: # for large images (>5000x5000), the F1 score is computed by a patch-based way
                H, W = gt.shape
                roi_size = 2000
            
                if H % roi_size != 0:
                    n_H = H // roi_size + 1
                    new_H = roi_size * n_H
                else:
                    n_H = H // roi_size
                    new_H = H
            
                if W % roi_size != 0:
                    n_W = W // roi_size + 1
                    new_W = roi_size * n_W    
                else:
                    n_W = W // roi_size
                    new_W = W    
            
                gt_pad = np.zeros((new_H, new_W), dtype=gt.dtype)
                seg_pad = np.zeros((new_H, new_W), dtype=gt.dtype)
                gt_pad[:H, :W] = gt
                seg_pad[:H, :W] = seg
                  
                tp = 0
                fp = 0
                fn = 0
                cell_true_num = 0
                cell_pred_num = 0
                for i in range(n_H):
                    for j in range(n_W):
                        if not args.count_bd_cells:
                            gt_roi  = remove_boundary_cells(gt_pad[roi_size*i:roi_size*(i+1), roi_size*j:roi_size*(j+1)])
                            seg_roi = remove_boundary_cells(seg_pad[roi_size*i:roi_size*(i+1), roi_size*j:roi_size*(j+1)])
                        gt_roi, _, _ = segmentation.relabel_sequential(gt_roi)
                        seg_roi, _, _ = segmentation.relabel_sequential(seg_roi)
                        cell_true_num += np.max(gt_roi)
                        cell_pred_num += np.max(seg_roi)
                        tp_i, fp_i, fn_i = eval_tp_fp_fn(gt_roi, seg_roi, threshold=threshold)
                        tp += tp_i
                        fp += fp_i
                        fn += fn_i            
            if tp == 0:
                precision = 0
                recall = 0
                f1 = 0
            else:
                precision = tp / cell_pred_num
                recall = tp / cell_true_num
                f1 = 2 * (precision * recall)/ (precision + recall)
                
            metrics['names'].append(name)
            metrics['true_num'].append(cell_true_num)
            metrics['pred_num'].append(cell_pred_num)
            metrics['correct_num(TP)'].append(tp)
            metrics['missed_num(FN)'].append(fn)    
            metrics['wrong_num(FP)'].append(fp)
            metrics['precision'].append(np.round(precision,4))
            metrics['recall'].append(np.round(recall, 4))
            metrics['dice'].append(np.round(dice_score, 4))
            metrics['F1'].append(np.round(f1, 4))
        except Exception:
            print('!'*20)
            print(name, 'evaluation error!')
            traceback.print_exc()
            failed.append(name)
        
    seg_metric_df = pd.DataFrame(metrics)
    if args.save_name.endswith('.csv'):
        args.save_name = args.save_name.split('.csv')[0]
    save_name = args.save_name + '-' + str(threshold) + '.csv'
    seg_metric_df.to_csv(join(args.output_path, save_name), index=False)
    print(save_name, 'threshold:', threshold, 'mean F1 Score:', np.mean(metrics['F1']), 'median F1 Score:', np.median(metrics['F1']))
    print(save_name, 'failed cases:', failed)
