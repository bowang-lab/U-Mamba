#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 20:27:52 2024

This script is used to generate the visualization video for the segmentation results.

author (@GitHub ID): Zhihe Wang (@h1shen), Feifei Li (@ff98li), Jun Ma (@JunMa11)
"""

import cv2
import os
join = os.path.join
listdir = os.listdir
makedirs = os.makedirs
remove = os.remove
isfile = os.path.isfile
isdir = os.path.isdir
basename = os.path.basename
from tqdm import trange

def slide_image(
		img1,
		img2,
		now_step,
		slide_step,
		target_size,
		line_thick = 5
	):
	slide_lenth = target_size[0]
	slide_unit = slide_lenth / slide_step

	start = int(slide_unit*now_step + 0.5)
	slide_img = img1.copy()
	slide_img[:, start:] = img2[:, start:]
	cv2.rectangle(slide_img, (start+line_thick, 0), (start, target_size[1]), (255, 255, 255), -1)

	return slide_img

def generate_video(
		fg_img_path,
		bg_img_path,
		slide_step,
		save_name = None,
		line_thick = 5,
		save_video_dir = './',
	):
	if save_name is None:
		save_vid_path = join(save_video_dir, basename(fg_img_path).replace('.png', '.mp4'))
	else:
		save_vid_path = join(save_video_dir, save_name)
	if isfile(save_vid_path):
		print(f"Video {save_vid_path} already exists. Skipping...")
		return
	
	bg_img = cv2.imread(bg_img_path, cv2.IMREAD_UNCHANGED)
	fg_img = cv2.imread(fg_img_path, cv2.IMREAD_UNCHANGED)
	
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	output_fps = 60
	output_width = fg_img.shape[1]
	output_height = fg_img.shape[0]
	target_size = (output_width, output_height)
	vizwriter = cv2.VideoWriter(
	    save_vid_path,
	    fourcc,
	    output_fps,
	    (output_width, output_height),
	    True
	)
	total_frames = slide_step

	for item in trange(0, slide_step):
		frame = slide_image(fg_img, bg_img, item, slide_step, target_size, line_thick)
		vizwriter.write(cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR))
	if vizwriter:
		vizwriter.release()


if __name__ == '__main__':
	bg_img_path = 'original_img.png'
	fg_img_path = 'seg_mask_overlay.png'
	slide_step = 200 # the number of frames in the video
	save_name = 'visual_seg.mp4'
	generate_video(fg_img_path, bg_img_path, slide_step, save_name)