import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
from scipy import io

def generate_mask(img):#256*256
    shape = img.shape                                           #(1,1,256,256)
    listimgH = []
    Zshape = [shape[0], shape[1],shape[2], shape[3]]            #(1,1,256,256)
    imgZ = img[:,:, :Zshape[2], :Zshape[3]].cuda()



    mask1 = torch.zeros((Zshape[0],Zshape[1], Zshape[2], Zshape[3]), dtype=torch.float).cuda()               #1,1,256,256
    mask2 = torch.zeros((Zshape[0],Zshape[1], Zshape[2], Zshape[3]), dtype=torch.float).cuda()
    mask3 = torch.zeros((Zshape[0], Zshape[1], Zshape[2], Zshape[3]), dtype=torch.float).cuda()
    mask4 = torch.zeros((Zshape[0], Zshape[1], Zshape[2], Zshape[3]), dtype=torch.float).cuda()

    for i in range(0,256,2):
        for j in range(0,256,2):

                mask1[:, :, i*2, j*2] = imgZ[:, :, i*2, j*2]
                mask1[:, :, i*2+1, j*2+2] = imgZ[:, :, i*2+1, j*2+2]
                mask1[:, :, i*2+2, j*2] = imgZ[:, :, i*2+2, j*2]
                mask1[:, :, i*2+3, j*2+2] = imgZ[:, :, i*2+3, j*2+2]

                mask2[:, :, i*2, j*2+1] = imgZ[:, :, i*2, j*2+1]
                mask2[:, :, i*2+1, j*2+3] = imgZ[:, :, i*2+1, j*2+3]
                mask2[:, :, i*2+2, j*2+1] = imgZ[:, :, i*2+2, j*2+1]
                mask2[:, :, i*2+3, j*2+3] = imgZ[:, :, i*2+3, j*2+3]

                mask3[:, :, i*2+1, j*2] = imgZ[:, :, i*2+1, j*2]
                mask3[:, :, i*2, j*2+2] = imgZ[:, :, i*2 , j*2+2]
                mask3[:, :, i*2+3, j*2] = imgZ[:, :, i*2+3, j*2]
                mask3[:, :, i*2+2, j*2+2] = imgZ[:, :, i*2+2, j*2+2]

                mask4[:, :, i*2+1, j*2+1] = imgZ[:, :, i*2+1, j*2+1]
                mask4[:, :, i*2, j*2+3] = imgZ[:, :, i*2 , j*2+3]
                mask4[:, :, i*2+3, j*2+1] = imgZ[:, :, i*2+3, j*2+1]
                mask4[:, :, i*2+2, j*2+3] = imgZ[:, :, i*2+2, j*2+3]


    listimgH.append(mask1)
    listimgH.append(mask2)
    listimgH.append(mask3)
    listimgH.append(mask4)
    return listimgH


def rec_image(sub1,sub2,sub3,sub4):
    img=torch.zeros([1,1,512,512])
    for i in range(0,256,2):
        for j in range(0,256,2):
            img[:, :, i*2, j*2] = sub1[:, :, i, j]
            img[:, :, i*2+1, j*2+2] = sub1[:, :, i, j+1]
            img[:, :, i*2+2, j*2] = sub1[:, :, i+1, j]
            img[:, :, i*2+3, j*2+2] = sub1[:, :, i+1, j+1]
            
            img[:, :, i*2, j*2+1] = sub2[:, :, i, j]
            img[:, :, i*2+1, j*2+3] = sub2[:, :, i, j+1]
            img[:, :, i*2+2, j*2+1] = sub2[:, :, i+1, j]
            img[:, :, i*2+3, j*2+3] = sub2[:, :, i+1, j+1]
            
            img[:, :, i*2+1, j*2] = sub3[:, :, i, j]
            img[:, :, i*2, j*2+2] = sub3[:, :, i, j+1]
            img[:, :, i*2+3, j*2] = sub3[:, :, i+1, j]
            img[:, :, i*2+2, j*2+2] = sub3[:, :, i+1, j+1]
            
            img[:, :, i*2+1, j*2+1] = sub4[:, :, i, j]
            img[:, :, i*2, j*2+3] = sub4[:, :, i, j+1]
            img[:, :, i*2+3, j*2+1] = sub4[:, :, i+1, j]
            img[:, :, i*2+2, j*2+3] = sub4[:, :, i+1, j+1]

    return img
#
#start
data1=torch.from_numpy(io.loadmat('./recon_L067_399_down.mat')['reconsub1']).view(1,1,256,256)
data2=torch.from_numpy(io.loadmat('./recon_L067_399_down.mat')['reconsub2']).view(1,1,256,256)
data3=torch.from_numpy(io.loadmat('./recon_L067_399_down.mat')['reconsub3']).view(1,1,256,256)
data4=torch.from_numpy(io.loadmat('./recon_L067_399_down.mat')['reconsub4']).view(1,1,256,256)



output = rec_image(data1,data2,data3,data4)

io.savemat('del_L6070001_down.mat',{'output': output.squeeze().cpu().detach().numpy()})    