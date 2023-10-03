import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
# from losses import get_optimizer
import numpy as np
import cv2
from pathlib import Path

import datasets
import time
# for radon
from physics.ct import CT, CT_LA
import matplotlib.pyplot as plt
import pydicom as dicom

import scipy.io as io

###############################################
# Configurations
###############################################
solver = 'MCG'
config_name = 'AAPM_256_ncsnpp_continuous'
sde = 'VESDE'
num_scales = 1000
ckpt_num = 185
N = num_scales

root = './samples/CT'

# Parameters for the inverse problem
angle_full = 180
sparsity = 6
num_proj = angle_full // sparsity  # 180 / 6 = 30
det_spacing = 1.0
size = 256

det_count = int((size)) #* (2*torch.ones(1)).sqrt()).ceil()) # ceil(size * \sqrt{2})










idx = 623
#filename = Path(root) / (str(idx).zfill(4) + '.npy')
# Specify save directory for saving generated samples
save_root = Path(f'./results/SV-CT/m{angle_full/sparsity}/{idx}/{solver}')


# Read data
sinogram = cv2.imread('./result_sinogram.png',0)
sinogram=np.array(sinogram)
print(sinogram.shape)
sino = torch.from_numpy(sinogram).view(1,1,256,180).to('cuda')




print("Loading all data")





# full
angles = np.linspace(0, np.pi, angle_full, endpoint=False)
print(1)
radon = CT(img_width=size, radon_view=23, circle=True, device='cuda')
radon_all = CT(img_width=size, radon_view=angle_full, circle=False, device='cuda')


# FBP
fbp = radon_all.A_dagger(sino)
plt.imsave(str(save_root / 'input' / f'FBP_GAN.png'), fbp.squeeze().cpu().detach().numpy(), cmap='gray')