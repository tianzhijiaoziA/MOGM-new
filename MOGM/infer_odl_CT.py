import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
# from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import numpy as np
import controllable_generation

from utils import restore_checkpoint, clear, \
    lambda_schedule_const, lambda_schedule_linear
from pathlib import Path
from models import utils as mutils
from models import ncsnpp
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector)
import datasets
import time
# for radon
from physics.ct import CT, CT_LA
import matplotlib.pyplot as plt
import pydicom as dicom
import cv2
import scipy.io as io
from mask import generate_mask, rec_image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr,structural_similarity as compare_ssim,mean_squared_error as compare_mse
from build_gemotry import rec
rec_al = rec()
def np2torch(x,x_device):
  x = torch.from_numpy(x)
  x = x.view(1,1,256,256)
  x = x.to(x_device.device)
  return x
def np2torch_radon(x,x_device):
  x = torch.from_numpy(x)
  x = x.view(1,1,720,640)
  x = x.to(x_device.device)
  return x 


view=23
def np2torch_radon_view(x,x_device):
  x = torch.from_numpy(x)
  x = x.view(1,1,view,640)
  x = x.to(x_device.device)
  return x 
###############################################
# Configurations
###############################################
solver = 'song_mean'
config_name = 'AAPM_256_ncsnpp_continuous'
sde = 'VESDE'
num_scales = 2000
ckpt_num = 185
N = num_scales

root = '/opt/data/private/zhangzhx2/3D-SDE/AAPM/L067_test_ye'

# Parameters for the inverse problem
angle_full = 180
sparsity = int(180/view)
num_proj = angle_full // sparsity  # 180 / 6 = 30
det_spacing = 1.0
size = 256

det_count = int((size)) #* (2*torch.ones(1)).sqrt()).ceil()) # ceil(size * \sqrt{2})

schedule = 'linear'

start_lamb = 1.0
end_lamb = 0.6
num_posterior_sample = 1

if schedule == 'const':
    lamb_schedule = lambda_schedule_const(lamb=start_lamb)
elif schedule == 'linear':
    lamb_schedule = lambda_schedule_linear(start_lamb=start_lamb, end_lamb=end_lamb)
else:
    NotImplementedError(f"Given schedule {schedule} not implemented yet!")

freq = 1

if sde.lower() == 'vesde':
    from configs.ve import AAPM_256_ncsnpp_continuous as configs
    ckpt_filename = f"./workdir/AAPM512_3d_4slicer/checkpoints/checkpoint_{ckpt_num}.pth"  #AAPM512_3d_4slicer
    config = configs.get_config()
    config.model.num_scales = N
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sde.N = N
    sampling_eps = 1e-5

batch_size = 1
config.training.batch_size = batch_size
predictor = ReverseDiffusionPredictor
corrector = LangevinCorrector
probability_flow = False
###修改了
snr = 0.16
n_steps = 1

batch_size = 1
config.training.batch_size = batch_size
config.eval.batch_size = batch_size
random_seed = 0

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

# optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
# state = dict(step=0, optimizer=optimizer,
#              model=score_model, ema=ema)
state = dict(step=0, model=score_model, ema=ema)

state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=False)

ema.copy_to(score_model.parameters())

idx = 399
filename = Path(root) / (str(idx).zfill(4) + '.npy')
# Specify save directory for saving generated samples
save_root = Path(f'./results/SV-CT/m{angle_full/sparsity}/{idx}/{solver}')
save_root.mkdir(parents=True, exist_ok=True)

irl_types = ['input', 'recon', 'label']
for t in irl_types:
    save_root_f = save_root / t
    save_root_f.mkdir(parents=True, exist_ok=True)

# Read data
#filename="./samples/L067_test_ye/0100.npy"
img = torch.from_numpy(np.load(filename).astype(np.float32)).view(1,1,256,256)
data1=torch.from_numpy(io.loadmat('./L506399_down.mat')['sub1']).view(1,1,256,256)
data2=torch.from_numpy(io.loadmat('./L506399_down.mat')['sub2']).view(1,1,256,256)
data3=torch.from_numpy(io.loadmat('./L506399_down.mat')['sub3']).view(1,1,256,256)
data4=torch.from_numpy(io.loadmat('./L506399_down.mat')['sub4']).view(1,1,256,256)
img2 = []

print("Loading all data")


img2 = torch.cat((data1,data2,data3,data4), dim=0)
img2 = img2.to(config.device)
# #img = data1

img = img.to(config.device)
plt.imsave(str(save_root / 'label' / f'label.png'), clear(img[0,:,:,:].unsqueeze(0)), cmap='gray')



# full
angles = np.linspace(0, np.pi, angle_full, endpoint=False)
print(1)
radon = CT(img_width=size, radon_view=num_proj, circle=True, device=config.device)
radon_all = CT(img_width=size, radon_view=angle_full, circle=True, device=config.device)

mask = torch.zeros([batch_size, 1, 720, 640]).to(config.device)
print(1)
mask[..., ::sparsity] = 1
#现在是有限角度
#mask[..., 1:num_proj+1] = 1

# Dimension Reducing (DR)
sinogram = np2torch_radon_view(rec_al.fp_view(clear(img)),img)
print("sinogram",sinogram.shape)
plt.imsave(str(save_root / 'input' / f'sino-SV.png'), clear(sinogram), cmap='gray')
# Dimension Preserving (DP)

sinogram_full = np2torch_radon(rec_al.fp(clear(img)),img) * mask
plt.imsave(str(save_root / 'input' / f'sino.png'), clear(np2torch_radon(rec_al.fp(clear(img)),img)), cmap='gray')
# FBP
fbp = np2torch(rec_al.fbp_view(clear(sinogram)),img)
plt.imsave(str(save_root / 'input' / f'FBP.png'), clear(fbp[0,:,:,:].unsqueeze(0)), cmap='gray')

# #再读取另一层
# dataset_400 = dicom.read_file('./L506_FD_0400.IMA')
# #dataset = dicom.read_file('./0001.IMA')
# img1_400 = dataset_400.pixel_array.astype(np.float32)/2500
# imgdel_400 = cv2.resize(img1_400, (320,320))
# imgdel_400 = np.expand_dims(imgdel_400,0)
# print("imgdel",np.max(imgdel_400))
# img_400 = torch.from_numpy(imgdel_400).view(1, 1, h, w)
# img_400 = img_400.to(config.device)
# sinogram_full_400 = radon_all.A(img_400) * mask
# print(sinogram.shape,sinogram_full_400.shape)
# fy =torch.cat((sinogram_full,sinogram_full_400),dim=1)
print("begin",solver)
if solver == 'MCG':
    pc_MCG = controllable_generation.get_pc_radon_MCG(sde,
                                                      predictor, corrector,
                                                      inverse_scaler,
                                                      snr=snr,
                                                      n_steps=n_steps,
                                                      probability_flow=probability_flow,
                                                      continuous=config.training.continuous,
                                                      denoise=True,
                                                      radon=radon,
                                                      radon_all=radon_all,
                                                      weight=0.5,
                                                      save_progress=False,
                                                      save_root=save_root,
                                                      lamb_schedule=lamb_schedule,
                                                      mask=mask)
 
    print("img or data",img.shape)
    
   
    x = pc_MCG(score_model, scaler(img2), measurement=sinogram.repeat(4,1,1,1))
elif solver == 'song_mean':
    pc_song = controllable_generation.get_pc_radon_song(sde,
                                                        predictor, corrector,
                                                        inverse_scaler,
                                                        snr=snr,
                                                        n_steps=n_steps,
                                                        probability_flow=probability_flow,
                                                        continuous=config.training.continuous,
                                                        save_progress=True,
                                                        save_root=save_root,
                                                        denoise=True,
                                                        radon=radon_all,
                                                        lamb=0.7)
    x,x_512 = pc_song(score_model, scaler(img.repeat(4,1,1,1)), mask, sinogram,0.1,True)
    plt.imsave(str(save_root / 'recon' / f'{str(idx).zfill(4)}.png'), clear(x[0,:,:,:].unsqueeze(0)), cmap='gray')
elif solver == 'song_improv':
    pc_song = controllable_generation.get_pc_radon_song(sde,
                                                        predictor, corrector,
                                                        inverse_scaler,
                                                        snr=snr,
                                                        n_steps=n_steps,
                                                        probability_flow=probability_flow,
                                                        continuous=config.training.continuous,
                                                        save_progress=True,
                                                        save_root=save_root,
                                                        denoise=True,
                                                        radon=radon_all,
                                                        lamb=0.7)
    x,x_512 = pc_song(score_model, scaler(img.repeat(4,1,1,1)), mask, sinogram,0.1,True)
    plt.imsave(str(save_root / 'recon' / f'{str(idx).zfill(4)}.png'), clear(x[0,:,:,:].unsqueeze(0)), cmap='gray')

elif solver == 'song_wo':
    pc_song = controllable_generation.get_pc_radon_song(sde,
                                                        predictor, corrector,
                                                        inverse_scaler,
                                                        snr=snr,
                                                        n_steps=n_steps,
                                                        probability_flow=probability_flow,
                                                        continuous=config.training.continuous,
                                                        save_progress=True,
                                                        save_root=save_root,
                                                        denoise=True,
                                                        radon=radon_all,
                                                        lamb=0.7)
    x,x_512 = pc_song(score_model, scaler(img.repeat(4,1,1,1)), mask, sinogram,0.1,False)
    plt.imsave(str(save_root / 'recon' / f'{str(idx).zfill(4)}.png'), clear(x[0,:,:,:].unsqueeze(0)), cmap='gray')   

elif solver == 'CGLS':
    pc_song = controllable_generation.get_pc_radon_CGLS(sde,
                                                        predictor, corrector,
                                                        inverse_scaler,
                                                        snr=snr,
                                                        n_steps=n_steps,
                                                        probability_flow=probability_flow,
                                                        continuous=config.training.continuous,
                                                        save_progress=True,
                                                        save_root=save_root,
                                                        denoise=True,
                                                        radon=radon_all,
                                                        lamb=0.7)
    x,x_512 = pc_song(score_model, scaler(img.repeat(4,1,1,1)), mask, sinogram,0.1,True)
    plt.imsave(str(save_root / 'recon' / f'{str(idx).zfill(4)}.png'), clear(x[0,:,:,:].unsqueeze(0)), cmap='gray')

# Recon
x_er = (x[0,:,:,:].unsqueeze(0)+x[1,:,:,:].unsqueeze(0)+x[2,:,:,:].unsqueeze(0)+x[3,:,:,:].unsqueeze(0))/4
psnr = compare_psnr(255*clear(x_er),255*(clear(img)),data_range=256)
ssim = compare_ssim(255*clear(x_er),255*(clear(img)),data_range=256)
psnr1 = compare_psnr(255*clear(x[0,:,:,:].unsqueeze(0)),255*(clear(img)),data_range=256)
ssim1 = compare_ssim(255*clear(x[0,:,:,:].unsqueeze(0)),255*(clear(img)),data_range=256)
print("PSNR and SSIM++",psnr,ssim)
print("PSNR and SSIM",psnr1,ssim1)
io.savemat(str(save_root / 'recon' / f'recon.mat'),{'input': clear(fbp),'label': clear(img),'reconsub1': clear(x[0,:,:,:]),'reconsub2': clear(x[1,:,:,:]),'reconsub3': clear(x[2,:,:,:]),'reconsub4': clear(x[3,:,:,:])})  
#plt.imsave(str(save_root / 'recon' / f'{str(idx).zfill(4)}_clip.png'), np.mean(clear(x[0,:,:,:].unsqueeze(0)), 0.1, 1.0), cmap='gray')
  