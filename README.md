# U-Mamba
Official repository for U-Mamba: enhancing...


## Installation 

Requirements: `Ubuntu 20.04`, `CUDA 11.7`


```bash
conda create -n umamba python=3.10 -y
conda activate umamba 
pip install torch==2.0.1 torchvision==0.15.2
pip install causal-conv1d==1.1.1
pip install mamba-ssm

cd umamba
pip install -e .
```
