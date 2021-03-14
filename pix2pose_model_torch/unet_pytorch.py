import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from torchsummary import summary

def transformer_loss(y_pred,y_recont_gt,y_prob_pred,y_prob_gt,sym):
    ##
    ## y_pred: generated image I3d bs*128*128*3 
    ## y_recont_gt: Igt bs*128*128*3 
    ## y_prob: Ie bs*128*128*1
    ## y_prob_gt: Mask of intact object bs*128*128*1
    visible = y_prob_gt.float()
    visible = torch.squeeze(visible,dim=3)
    y_pred = torch.transpose(y_pred, 1, -1) # Put the channel at last dim
    y_recont_gt = torch.transpose(y_recont_gt, 1, -1)
    y_prob_pred = torch.transpose(y_prob_pred, 1, -1)
    y_prob_gt = torch.transpose(y_prob_gt, 1, -1)
    #generate transformed values using sym
    if(len(sym)>1):
        #if(True):
        for sym_id,transform in enumerate(sym): #3x3 matrix
            tf_mat=torch.tensor(transform,dtype=y_recont_gt.dtype)
            y_gt_transformed = torch.transpose(torch.matmul(tf_mat,torch.transpose(torch.reshape(y_recont_gt,[-1,3]),1,0)),1,0)
            y_gt_transformed = torch.reshape(y_gt_transformed,[-1,128,128,3])
            loss_xyz_temp = torch.sum(torch.abs(y_gt_transformed-y_pred),dim=3)/3
            loss_sum=torch.sum(loss_xyz_temp,dim=[1,2])
            if(sym_id>0):
                loss_sums = torch.cat([loss_sums,torch.unsqueeze(loss_sum,dim=0)],dim=0)
                loss_xyzs = torch.cat([loss_xyzs,torch.unsqueeze(loss_xyz_temp,dim=0)],dim=0)
            else:
                loss_sums = torch.unsqueeze(loss_sum,dim=0) 
                loss_xyzs = torch.unsqueeze(loss_xyz_temp,dim=0)
        min_values, _ = torch.min(loss_sums,dim=0,keepdim=True)
        min_values = min_values.repeat([3,1])
        loss_switch = torch.tensor(torch.equal(loss_sums,min_values),dtype=y_pred.dtype)
        loss_xyz = torch.unsqueeze(torch.unsqueeze(loss_switch,dim=-1),dim=-1)*loss_xyzs
        loss_xyz = torch.sum(loss_xyz,dim=0)
    else:
        loss_xyz = torch.sum(torch.abs(y_recont_gt-y_pred),dim=3)/3
    y_prob_pred = torch.squeeze(y_prob_pred,-1)
    loss_xyz[loss_xyz >= 1] = 1.0
    prob_loss = (y_prob_pred-loss_xyz)**2
    loss_invisible = (1-visible)*loss_xyz
    loss_visible = visible*loss_xyz
    loss = loss_visible*3 + loss_invisible+ 0.5*prob_loss 
    loss = torch.mean(loss,dim=[1,2])
    return loss

# #sanity check for transformer_loss
# y_pred = torch.rand([5,3,128,128])
# y_recont_gt = torch.rand([5,3,128,128])
# y_prob_pred = torch.rand([5,1,128,128])
# y_prob_gt = torch.rand([5,1,128,128])
# y_prob_gt[y_prob_gt > 0.5] = 1
# y_prob_gt[y_prob_gt <= 0.5] = 0
# rt = torch.eye(3)
# sym = [rt,rt,rt]
# # sym = [rt]
# loss = transformer_loss(y_pred, y_recont_gt, y_prob_pred, y_prob_gt, sym)
# print(loss.shape)


class Unet(nn.Module):
    # Input should be batch_szie*3*128*128
    def __init__(self, in_channels=3):
        super(Unet, self).__init__()
        #C3-128
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 5, 2,padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.3,True),
        )
        #C128-256
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, 5, 2,padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.3,True)
        )
        #C256-256
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, 5, 2,padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.3,True)
        )
        #C256-512
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 5, 2,padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.3,True)
        )
        #bottle neck
        self.linear1 = nn.Linear(8*8*512, 256)
        self.linear2 = nn.Linear(256, 8*8*256)
        
        #TC256-128
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, 2, 2, output_padding=1), ##not sure should I use output_padding
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.3, True)
        )
        #C256-256
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 5, 1,padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.3,True)
        )
        #TC256-128
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, 2, 2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.3, True)
        )
        #C256-256
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 5, 1,padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.3,True)
        )
        #TC256-64
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 5, 2, 2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3, True)
        )
        #C128-128
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 128, 5, 1,padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.3,True)
        )
        #TC128-4
        self.deconv4 = nn.ConvTranspose2d(128, 4, 5, 2, 2, output_padding=1)


    def forward(self, x):
        #x should be batch_size*3*128*128
        e1 = self.conv1(x)      #128*64*64

        e2 = self.conv2(e1)     #256*32*32
        e3 = self.conv3(e2)     #256*16*16
        e4 = self.conv4(e3)     #512*8*8
        e4 = torch.flatten(e4, start_dim=1) #batch_size*(512*8*8)
        v = self.linear1(e4) #batch_size*256
        d0 = self.linear2(v) #batch_size*(8*8*256)
        d0 = torch.reshape(d0, [-1,256,8,8])
        d0_ = self.deconv1(d0)
        d1 = torch.cat([self.deconv1(d0), e3[:,:128,:,:]], dim = 1)   #256*16*16
        d1 = self.conv5(d1)     #256*16*16
        d1_ = self.deconv2(d1)
        d2 = torch.cat([self.deconv2(d1),e2[:,:128,:,:]], dim = 1)     #256*32*32
        d2 = self.conv6(d2)     #256*32*32
        d3 = torch.cat([self.deconv3(d2), e1[:,:64,:,:]], dim = 1)   #128*64*64
        d3 = self.conv7(d3)     #128*64*64
        d4 = self.deconv4(d3)   #4*128*128
        I_3d = torch.tanh(d4[:,:3,:,:])
        I_e = torch.sigmoid(d4[:,3:4,:,:])
        return I_3d, I_e

##test unet sanity
# encoder = Unet()
# x = torch.rand([10,3,128,128])
# I_3d, I_e = encoder(x)
# print(I_3d.shape)
# print(I_e.shape)


