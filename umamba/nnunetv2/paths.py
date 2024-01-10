#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
join = os.path.join
"""
Please make sure your data is organized as follows:

data/
├── nnUNet_raw/
│   ├── Dataset701_AbdomenCT/
│   │   ├── imagesTr
│   │   │   ├── FLARE22_Tr_0001_0000.nii.gz
│   │   │   ├── FLARE22_Tr_0002_0000.nii.gz
│   │   │   ├── ...
│   │   ├── labelsTr
│   │   │   ├── FLARE22_Tr_0001.nii.gz
│   │   │   ├── FLARE22_Tr_0002.nii.gz
│   │   │   ├── ...
│   │   ├── dataset.json
│   ├── Dataset702_AbdomenMR/
│   │   ├── imagesTr
│   │   │   ├── amos_0507_0000.nii.gz
│   │   │   ├── amos_0508_0000.nii.gz
│   │   │   ├── ...
│   │   ├── labelsTr
│   │   │   ├── amos_0507.nii.gz
│   │   │   ├── amos_0508.nii.gz
│   │   │   ├── ...
│   │   ├── dataset.json
│   ├── ...
"""
base = join(os.sep.join(__file__.split(os.sep)[:-3]), 'data') 
# or you can set your own path, e.g., base = '/home/user_name/Documents/U-Mamba/data'
nnUNet_raw = join(base, 'nnUNet_raw') # os.environ.get('nnUNet_raw')
nnUNet_preprocessed = join(base, 'nnUNet_preprocessed') # os.environ.get('nnUNet_preprocessed')
nnUNet_results = join(base, 'nnUNet_results') # os.environ.get('nnUNet_results')

if nnUNet_raw is None:
    print("nnUNet_raw is not defined and nnU-Net can only be used on data for which preprocessed files "
          "are already present on your system. nnU-Net cannot be used for experiment planning and preprocessing like "
          "this. If this is not intended, please read documentation/setting_up_paths.md for information on how to set "
          "this up properly.")

if nnUNet_preprocessed is None:
    print("nnUNet_preprocessed is not defined and nnU-Net can not be used for preprocessing "
          "or training. If this is not intended, please read documentation/setting_up_paths.md for information on how "
          "to set this up.")

if nnUNet_results is None:
    print("nnUNet_results is not defined and nnU-Net cannot be used for training or "
          "inference. If this is not intended behavior, please read documentation/setting_up_paths.md for information "
          "on how to set this up.")
