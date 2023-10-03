import os
import zipimport
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from pathlib import Path
from models import utils as mutils
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector,
                      AnnealedLangevinDynamics,
                      get_pc_fouriercs_RI)
from models import ncsnpp
import time
from utils import fft2, ifft2, get_mask, get_data_scaler, get_data_inverse_scaler, restore_checkpoint
import torch
import torch.nn as nn
import numpy as np   
from models.ema import ExponentialMovingAverage
import matplotlib.pyplot as plt
import matplotlib
import importlib
import argparse
import skimage
from matplotlib import cm
from scipy import io
from torch.utils.data import DataLoader, Dataset
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
#from skimage.measure import compare_psnr,compare_ssim 

#CUDA_VISIBLE_DEVICES=2 python inference_single-coil.py --data '001' --mask_type 'gaussian1d' --acc_factor 4 --center_fraction 0.08 --N 2000

def main():
    ###############################################
    # 1. Configurations
    ###############################################
    # Specify save directory for saving generated samples
    
    root='/home/wangyy/score_mri/ceshidata/'

 
    save_path='/home/wangyy/score_mri/tongjiwo/'

    file_name = os.listdir(root)
    save_root = Path(f'./results/paperdata')
    save_root.mkdir(parents=True, exist_ok=True)

    irl_types = ['input', 'recon', 'recon_chaju', 'label']
    for t in irl_types:
        save_root_f = save_root / t
        save_root_f.mkdir(parents=True, exist_ok=True)
    # args
    args = create_argparser().parse_args()
    N = args.N
    m = args.m
    #fname = args.data
    

    print('initaializing...')
    configs = importlib.import_module(f"configs.ve.fastmri_knee_320_ncsnpp_continuous")
    config = configs.get_config()
    img_size = config.data.image_size
    batch_size = 1

           
         
    #ckpt_filename = f"/data/wyy/score-MRI-main/workdir/fastmri_multicoil_knee_320_batch4ontall/checkpoints/checkpoint_24.pth"  
    #ckpt_filename = f"/data/wyy/score-MRI-main/workdir/fastmri_coil_knee_320_0922/checkpoints/checkpoint_R12.pth"
    #ckpt_filename = f"/data/wangyy/workdir/fastmri_knee_1101quanwcnn/checkpoints/checkpoint_12.pth"#r47
    #ckpt_filename = f"./weights/0902/checkpoint_63.pth"
    ckpt_filename = f"./weights/checkpoint_R12.pth" ##fastmri_knee_1101quanwcnn/checkpoints/checkpoint_12.pth为我们的   
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=N)

    config.training.batch_size = batch_size  
    predictor = ReverseDiffusionPredictor 
    corrector = LangevinCorrector
    probability_flow = False
    snr = 0.16

    # sigmas = mutils.get_sigmas(config)
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # create model and load checkpoint
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(),
                                   decay=config.model.ema_rate)
    state = dict(step=0, model=score_model, ema=ema)
    state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=False)
    ema.copy_to(score_model.parameters())
    #load data
    
    #创建保存np
    datanp = np.zeros([120,4])
    path=os.path.join(root,file_name[0])
    n=0
    for name in file_name:
        
        path=os.path.join(root,name)
        #path = f'/home/wangyy/score_mri/samples/single-coil/001.npy'  
        #path = f'./ceshidata/file1002546_esc19.npy'
        dataCT=np.load(path)
        inputer = dataCT.astype(np.float32)
        inputer = (inputer-np.min(inputer))/(np.max(inputer)-np.min(inputer))
        img = torch.from_numpy(inputer)
    
        img = img.view(1, 1, 320, 320)  

        img = img.to(config.device)
    

    
         
        # #filename = f'/data/wyy/score-MRI-main/npy/ifft_K/file1000363_k15.npy'
        # #filename = f'/data/wyy/score-MRI-main/npy/ceshidata/file1002546_esc19.npy'  
           
        # # Read data 
        # inputer = np.load(filename).astype(np.float32)
        # print(np.max(inputer),np.min(inputer))
        # #inputer = np.load(filename).astype(np.complex64) 
        # #print(inputer[:5,:5])
        # inputer = (inputer-np.min(inputer))/(np.max(inputer)-np.min(inputer))
        
        # #matplotlib.image.imsave('/data/wyy/score-MRI-main/'+'20input_papershiyan.png', inputer.astype(np.float32), cmap = cm.coolwarm)
        # print('inputer', inputer.shape)

        # img = torch.from_numpy(inputer)
        # #print('img', type(img))
        # #print(img.shape) torch.Size([320, 320])
        # img = img.view(1, 1, 320, 320)  
        # #print(img.shape) torch.tensor([1,1,320,320])
        # #输入  

        # img = img.to(config.device)
        mask = get_mask(img, 320, 1,
                        type='gaussian1d',
                        acc_factor=4,  
                        center_fraction=0.08) 
        
        # mask = np.load("./18acc4w_mask.npy").astype(np.complex64)
        # mask = torch.from_numpy(mask) 
        # mask = mask.view(1, 1, 320, 320).to(config.device) 
        # mask = get_mask(img, img_size, batch_size,
        #                 type=args.mask_type,
        #                 acc_factor=args.acc_factor,  
        #                 center_fraction=args.center_fraction)  
        
        #np.save(save_root / 'mask_4_0.08_uf.npy', mask.squeeze().cpu().detach().numpy())
        ###############################################
        # 2. Inference
        ###############################################
        
        pc_fouriercs = get_pc_fouriercs_RI(sde,
                                        predictor, corrector,
                                        inverse_scaler,
                                        snr=snr,
                                        n_steps=m,
                                        probability_flow=probability_flow,
                                        continuous=config.training.continuous,
                                        denoise=True)
    
        # fft
        kspace = fft2(img)
    
        # undersampling
        under_kspace = kspace * mask
        
        under_img = ifft2(under_kspace)
        print('under_img',under_img.shape)
        z = under_img.mean(dim=-3)
        z = z.squeeze().cpu().detach().numpy()
        #io.savemat(str(save_root/str(fname)) + 'input.mat',{'input': z,'label': inputer})
        
        
        print(f'Beginning inference')
        tic = time.time()
        x = pc_fouriercs(score_model, scaler(under_img), mask, Fy=under_kspace, inputer=inputer)
        x=x.mean(dim=-3)
        xer = x
        toc = time.time() - tic

        ###############################################
        #PSNR，SSIM
        
        xer = xer.squeeze().cpu().detach().numpy()
    
        #print(xer.shape, inputer.shape) (320,).torch.size([1,1,320,320])
        psnr1 = peak_signal_noise_ratio(255*np.real(xer),255*np.real(inputer),data_range=255)
        ssim1 = structural_similarity(255*np.real(xer),255*np.real(inputer),data_range=255)
        
        io.savemat(save_path+'Gua4_fullwo'+name+'.mat',{'recon': xer,'label': inputer})
        print(int(name[4:11]+name[15:16]))
        datanp[n,0]=int(name[4:11]+name[15:16])
        datanp[n,1]=psnr1
        datanp[n,2]=ssim1
        n=n+1
        #print("PSNR_under:%.4f"%(psnr1),"SSIM_under:%.4f"%(ssim1))
        ###############################################
        #print(f'Time took for recon: {toc} secs.')
    
    np.save(save_path + 'our.npy', datanp)
    f1=open(save_path + "our.csv", 'w')
    np.savetxt(f1, datanp, delimiter=',', fmt='%.04f')
    # ###############################################
    # # 3. Saving recon
    # ###############################################
    # input = under_img.squeeze().cpu().detach().numpy()
    # label = img.squeeze().cpu().detach().numpy()
    # mask_sv = mask.squeeze().cpu().detach().numpy()

    # #np.save(str(save_root / 'input' / str(fname)) + '.npy', input)
    # #np.save(str(save_root / 'input' / (str(fname) + '_mask')) + '.npy', mask_sv)
    # #np.save(str(save_root / 'label' / str(fname)) + '.npy', label)
    # #plt.imsave(str(save_root / 'input' / str(fname)) + '.png', np.abs(input), cmap='gray')
    # #plt.imsave(str(save_root / 'input' / (str(fname) + '_mask')) + '.png', np.abs(mask_sv), cmap='gray')
    # plt.imsave(str(save_root / 'label' / str(fname)) + 'ceshi.png', np.abs(label), cmap='gray')

    # recon = x.squeeze().cpu().detach().numpy() 
    # #np.save(str(save_root / 'recon' / str(fname)) + '.npy', recon)
    # plt.imsave(str(save_root / 'recon' / str(fname)) + 'ceshi.png', np.abs(recon), cmap='gray')  
    # #recon = (recon-np.min(recon))/(np.max(recon)-np.min(recon)) 
    # chaju = (recon)-(label) 
    # plt.imsave(str(save_root / 'recon_chaju' / str(fname)) + 'ceshi.png', np.abs(chaju), cmap='viridis') #viridis
    # #plt.imsave(str(save_root / 'recon_chaju' / str(fname)) + '.png', np.abs(chaju), cmap='gray')
    
    # io.savemat(str(save_root/str(fname)) + '.mat',{'recon': recon,'label': inputer})

    # #io.savemat(str(save_root/str(fname)) + 'chajuzuo.mat',{'maskmri': chaju}) 

def create_argparser(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='which data to use for reconstruction', required=True)
    parser.add_argument('--mask_type', type=str, help='which mask to use for retrospective undersampling.'
                                                      '(NOTE) only used for retrospective model!', default='gaussian1d',
                        choices=['gaussian1d', 'uniform1d', 'gaussian2d'])
    parser.add_argument('--acc_factor', type=int, help='Acceleration factor for Fourier undersampling.'
                                                       '(NOTE) only used for retrospective model!', default=4)
    parser.add_argument('--center_fraction', type=float, help='Fraction of ACS region to keep.'
                                                       '(NOTE) only used for retrospective model!', default=0.08)
    parser.add_argument('--save_dir', default='./results')
    parser.add_argument('--N', type=int, help='Number of iterations for score-POCS sampling', default=2000)
    parser.add_argument('--m', type=int, help='Number of corrector step per single predictor step.'
                                              'It is advised not to change this default value.', default=1)
    return parser


if __name__ == "__main__":
    main()