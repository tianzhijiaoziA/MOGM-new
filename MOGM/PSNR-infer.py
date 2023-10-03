from pathlib import Path
from models import utils as mutils
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector)
import sampling
from models import ncsnpp
import time
from utils import get_data_scaler, get_data_inverse_scaler, restore_checkpoint
import torch
import torch.nn as nn
import numpy as np   
from models.ema import ExponentialMovingAverage
import matplotlib.pyplot as plt
import importlib
import argparse
import skimage
from skimage.measure import compare_psnr,compare_ssim
import pydicom as dicom  
import odl
#CUDA_VISIBLE_DEVICES=1 python PSNR-infer.py --N 1000
#CUDA_VISIBLE_DEVICES=1 python PSNR-infer.py --data '001' --mask_type 'gaussian1d' --acc_factor 4 --center_fraction 0.08 --N 1000
def main():
    ###############################################
    # 1. Configurations
    ###############################################
    
    # args
    args = create_argparser().parse_args()
    N = args.N
    m = args.m
    fname = args.data
    #dicom格式转换
    dataset = dicom.read_file('./samples/aapm/L506_FD_1_1.CT.0002.0526.2015.12.22.20.19.52.894480.358619619.IMA')
    
    #filename = f'./samples/single-coil/{fname}.npy'
    img1 = dataset.pixel_array.astype(np.float32)
    #print(type(img1)) class np.arrary
    img = img1
    RescaleSlope = dataset.RescaleSlope
    RescaleIntercept = dataset.RescaleIntercept     
    CT_img = img * RescaleSlope + RescaleIntercept
    #得到gt
    image_gt = (CT_img-np.min(CT_img))/(np.max(CT_img)-np.min(CT_img))
    inputer = torch.tensor(image_gt)
    #print('获得的输入的真实图像',inputer)
    # low CT
    angle_partition = odl.uniform_partition(0, 2 * np.pi, 1000)
    detector_partition = odl.uniform_partition(-360, 360, 1000)
    geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition,
                                    src_radius=500, det_radius=500)
    reco_space = odl.uniform_discr(min_pt=[-128, -128], max_pt=[128, 128], shape=[512, 512], dtype='float32')
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry)
    ATA = ray_trafo.adjoint(ray_trafo(ray_trafo.domain.one()))
    pseudoinverse = odl.tomo.fbp_op(ray_trafo)
    #加入possion
    photons_per_pixel =  5e4
    mu_water = 0.02
    phantom = reco_space.element(img)
    phantom = phantom/1000.0
    proj_data = ray_trafo(phantom)
    proj_data = np.exp(-proj_data * mu_water)
    proj_data = odl.phantom.poisson_noise(proj_data * photons_per_pixel)
    proj_data = np.maximum(proj_data, 1) / photons_per_pixel
    proj_data = np.log(proj_data) * (-1 / mu_water)
    image_input = pseudoinverse(proj_data)
    #得到原始的np图像
    image_input000 = image_input
    image_input = np.copy(image_input)
    maxdegrade = np.max(image_input)
    
    #print(type(image_input)) <class 'odl.discr.discr_space.DiscretizedSpaceElement'>
    image_input1 = image_input
    print('initaializing...')
    #config
    configs = importlib.import_module(f"configs.ve.aapm_512_ncsnpp_continuous")
    config = configs.get_config()
    img_size = config.data.image_size
    batch_size = 1

    # Read data
    #image_input1 = torch.from_numpy(image_input1.astype(np.float32))
    #image_input1 = image_input1.view(1, 1, 512, 512)
    #输入
    
    #########
    #print(image_input1.shape, inputer.shape) (512,512),(512,512)
    image_input1 = np.expand_dims(image_input1,2)
    #data_array_10 = data_array.repeat([1,1,10],axis=2)
    
    image_input1 = np.tile(image_input1,(1,1,10))
    image_input1=image_input1.transpose((2,0,1))
    image_input1 = torch.tensor(image_input1.astype(np.complex64))
    print(image_input1.shape,type(image_input1))
    # #huoqu yuanshide shuru
    
    #print('image_input1', image_input1.shape)
    #muti -channel
    image_input1 = image_input1.view(1, 10, 512, 512)
    #image_input1 = torch.concat([image_input1, image_input1, image_input1, image_input1, image_input1, image_input1, image_input1, image_input1, image_input1, image_input1], axis=1)
    #image_input1 = torch.concat([image_input1, image_input1], axis=0)
    #print('image_input1', type(image_input1))
    #image_input2=image_input1
    #image_input2=image_input2.mean(dim=-3)
    #image_input3=image_input2.squeeze().cpu().detach().numpy()

    image_input1 = image_input1.to(config.device)

    #mask = get_mask(img, img_size, batch_size,
                    #type=args.mask_type,
                    #acc_factor=args.acc_factor,
                    #center_fraction=args.center_fraction)

    ckpt_filename = f"./workdir/aapm_512_ncsnpp_continuous_0819/checkpoints/checkpoint_Q16.pth"
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=N)

    config.training.batch_size = batch_size
    # Probability flow ODE sampling with black-box ODE solvers
    predictor = ReverseDiffusionPredictor
    corrector = LangevinCorrector
    probability_flow = False
    ########
    snr = 0.4
    eps=1e-3
    # sigmas = mutils.get_sigmas(config)
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # create model and load checkpoint
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(),
                                   decay=config.model.ema_rate)
    state = dict(step=0, model=score_model, ema=ema)
    state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=True)
    ema.copy_to(score_model.parameters())

    # Specify save directory for saving generated samples
    save_root = Path(f'./results/aapm_ldct_one512')
    save_root.mkdir(parents=True, exist_ok=True)

    irl_types = ['input', 'recon', 'recon_progress', 'label','out']
    for t in irl_types:
        save_root_f = save_root / t
        save_root_f.mkdir(parents=True, exist_ok=True)

    ###############################################
    # 2. Inference
    ###############################################
    ##############################################
    M = 1
    sampling_shape = (config.eval.batch_size,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    #sampling_fn = samplingaapm.get_sampling_fn(config, sde, sampling_shape, scaler(image_input1), 1e-5)
    pc_ct = sampling.get_pc_sampler_CT(sde, sampling_shape, predictor, corrector, inverse_scaler, snr=snr,
                   n_steps=M, probability_flow=probability_flow, continuous=config.training.continuous,
                   denoise=True, eps=eps, device='cuda')
    # fft
    #kspace = fft2(img)
    # undersampling
    #under_kspace = kspace * mask
    #under_img = ifft2(under_kspace)
    ###CT特有
    lab = np.copy(image_input000)
    z = np.copy(lab)
    norm_diff = ray_trafo.adjoint((ray_trafo(scaler(image_input000)) - proj_data))
    print(f'Beginning inference')
    tic = time.time()

    x = pc_ct(score_model, scaler(image_input1), lab, maxdegrade, z, norm_diff, ATA, image_gt)
    x000=x.mean(dim=-3)
    x=np.array(x.cpu().detach(),dtype = np.float32)
    x_m = np.squeeze(x)
    x_m = np.mean(x_m,axis=0)
    
    #print('let me see output',x,x.shape)
    #x=torch.tensor(x)

    max_psnr = 0
    max_ssim = 0
    ##############################################
    for step in range(1):
          
        ## ********** SQS ********* ##
        hyper = 150
        #输入图像x-迭代生成x1 *maxdegrade = np.max(image_input)
        sum_diff =  lab - x_m*maxdegrade
        lab_new = z - (norm_diff + 2*hyper*sum_diff)/(ATA + 2*hyper)
        z = lab_new + 0.5 * (lab_new - lab)
        lab = lab_new
        x_rec = lab.asarray()
        x_rec = x_rec/maxdegrade
                

              
            
          
        psnr = compare_psnr(255*abs(x_rec/np.max(x_rec)),255*abs(image_gt),data_range=255)
        ssim = compare_ssim(abs(x_rec/np.max(x_rec)),abs(image_gt),data_range=1)
        ###得到rec的值
        out = x_rec
        
        #print("current {} step".format(step),'PSNR :', psnr,'SSIM :', ssim)
        x_mid = np.zeros([1,10,512,512],dtype=np.float32)
        x_rec = np.clip(x_rec,0,1)
        x_rec = np.expand_dims(x_rec,2)
        x_mid_1 = np.tile(x_rec,[1,1,10])
        x_mid_1 = np.transpose(x_mid_1,[2,0,1])
        x_mid[0,:,:,:] = x_mid_1
        x = torch.tensor(x_mid,dtype=torch.float32).cuda()
        #x0 = torch.tensor(x_mid,dtype=torch.float32)
    print("PSNR:%.4f"%(psnr),"SSIM:%.4f"%(ssim))

    
    ##############################################
    toc = time.time() - tic

    ###############################################
    #PSNR，SSIM
    '''
    max_psnr = 0
    max_ssim = 0
    x=np.array(x).astype(float)
    xer = x.squeeze().cpu().detach().numpy()
    
    psnr1 = compare_psnr(255*abs(xer),255*abs(inputer),data_range=255)
    ssim1 = compare_ssim(abs(xer),abs(inputer),data_range=1)
    print("PSNR_under:%.4f"%(psnr1),"SSIM_under:%.4f"%(ssim1))
    ###############################################
    '''
    print(f'Time took for recon: {toc} secs.')
    
    ###############################################
    # 3. Saving recon

    ###############################################
    #input = image_input3.squeeze().cpu().detach().numpy()
    input = image_input
    
    label = inputer.squeeze().cpu().detach().numpy() 
    #mask_sv = mask.squeeze().cpu().detach().numpy()

    np.save(str(save_root / 'input' / fname) + '.npy', input) 
    #np.save(str(save_root / 'input' / (fname + '_mask')) + '.npy', mask_sv)
    np.save(str(save_root / 'label' / fname) + '.npy', label)
    plt.imsave(str(save_root / 'input' / fname) + '.png', np.abs(input), cmap='gray')
    plt.imsave(str(save_root / 'label' / fname) + '.png', np.abs(label), cmap='gray')
    recon = x000.squeeze().cpu().detach().numpy()
    np.save(str(save_root / 'recon' / fname) + '.npy', recon)
    plt.imsave(str(save_root / 'recon' / fname) + '.png', np.abs(recon), cmap='gray')
    
    np.save(str(save_root / 'out' / fname) + '.npy', out)
    plt.imsave(str(save_root / 'out' / fname) + '.png', np.abs(out), cmap='gray')
    #print('recon,label PSNR',recon,label)
    #maxdegrade = np.max(input)
    #recon = recon/maxdegrade
    #psnr1 = compare_psnr(255*abs(recon/np.max(recon)),255*abs(label),data_range=255)
    #ssim1 = compare_ssim(abs(recon/np.max(recon)),abs(label),data_range=1)
    #print("PSNR_under:%.4f"%(psnr1),"SSIM_under:%.4f"%(ssim1))

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