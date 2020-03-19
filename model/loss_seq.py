import torch
from torch import nn
from torch.nn import Parameter
from numpy import pi
import math
import torch.nn.functional as F
import numpy as np

epsilon = 1e-7 #regularization value in Keras

class SphereMSE(nn.Module):
    def __init__(self, h, w):
        super(SphereMSE, self).__init__()
        self.h, self.w = h, w
        weight = torch.zeros(1, 1, h, w)
        theta_range = torch.linspace(0, pi, steps=h + 1)
        dtheta = pi / h
        dphi = 2 * pi / w
        for theta_idx in range(h):
            weight[:, :, theta_idx, :] = dphi * (math.sin(theta_range[theta_idx]) + math.sin(theta_range[theta_idx+1]))/2 * dtheta
        self.weight = Parameter(weight, requires_grad=False)

    def forward(self, out, target):
        return torch.sum((out - target) ** 2 * self.weight) / out.size(0)

def cross_entropy(input,target):
    input = input.view(input.size(0),input.size(1), -1)
    target = target.view(target.size(0),target.size(1),-1)
    loss = (-target*torch.log(torch.clamp(input,min=epsilon,max=1))).sum(-1)
    return loss.mean()

def binary_cross_entropy(input,target):
    input = input.view(input.size(0), input.size(1), -1)
    target = target.view(target.size(0), target.size(1),-1)
    loss = target*torch.log(torch.clamp(input,min=epsilon,max=1)) + (1-target)*torch.log(1-torch.clamp(input,min=epsilon,max=1))
    loss = -torch.mean(loss)
    return loss

class SphereCE(nn.Module):
    def __init__(self, h, w):
        super(SphereCE, self).__init__()
        self.h, self.w = h, w
        weight = torch.zeros(1, 1, h, w)
        theta_range = torch.linspace(0, pi, steps=h + 1)
        dtheta = pi / h
        dphi = 2 * pi / w
        for theta_idx in range(h):
            weight[:, :, theta_idx, :] = dphi * (math.sin(theta_range[theta_idx]) + math.sin(theta_range[theta_idx+1]))/2 * dtheta
        self.weight = Parameter(weight, requires_grad=False)

    def forward(self, out, target):
        batch, seq, h, w = out.size()
        # out = out.view(batch,seq,-1)
        # out = F.softmax(out,dim=-1).view(batch,seq,h,w)
        # target = target.view(batch,seq,-1)
        # target = F.softmax(target,dim=-1).view(batch,seq,h,w)
        loss = (-target*torch.log(torch.clamp(out,min=epsilon,max=1))) * self.weight
        return torch.sum(loss) / out.size(0)


def NSS(input,fixation):    
    input = input.view(input.size(0), input.size(1), -1)
    # input = torch.div(input,input.max(-1,keepdim=True)[0].expand_as(input)+epsilon)
    fixation = fixation.view(fixation.size(0),fixation.size(1),-1)
    input = torch.div(input-input.mean(-1,keepdim=True).expand_as(input),input.std(-1,keepdim=True).expand_as(input) + epsilon)
    loss = torch.div(torch.mul(input,fixation).sum(-1), fixation.sum(-1) + epsilon)

    return torch.mean(loss)

def CC(input,fixmap,weight=None): 
    if weight is not None:
        input = torch.mul(input,weight)
        fixmap = torch.mul(fixmap,weight)
    input = input.view(input.size(0), input.size(1), -1)
    # input = torch.div(input,input.max(-1,keepdim=True)[0].expand_as(input)+epsilon)
    fixmap = fixmap.view(fixmap.size(0),fixmap.size(1),-1)
    fixmap = torch.div(fixmap,fixmap.sum(-1,keepdim=True).expand_as(fixmap))
    input = torch.div(input,input.sum(-1,keepdim=True).expand_as(input))

    sum_prod = torch.mul(input,fixmap).sum(-1,keepdim=True)
    sum_x = input.sum(-1,keepdim=True)
    sum_y = fixmap.sum(-1,keepdim=True)
    sum_x_square = (input**2).sum(-1,keepdim=True)
    sum_y_square = (fixmap**2).sum(-1,keepdim=True)
    num = sum_prod - torch.mul(sum_x,sum_y)/input.size(-1)
    den = torch.sqrt((sum_x_square-sum_x**2/input.size(-1))*(sum_y_square-sum_y**2/input.size(-1)))
    loss = torch.div(num,den+epsilon)

    # cov = torch.mul(input-input.mean(-1,keepdim=True).expand_as(input),fixmap-fixmap.mean(-1,keepdim=True).expand_as(fixmap)).mean(-1)
    # loss = torch.div(cov,torch.mul(input.std(-1),fixmap.std(-1)) + epsilon)

    return torch.mean(loss)


def CC_1d(pos,neg):
    pos = torch.div(pos,pos.sum())
    neg = torch.div(neg,neg.sum())
    sum_prod = torch.mul(pos,neg).sum()
    sum_x = pos.sum()
    sum_y = neg.sum()
    sum_x_square = (pos**2).sum()
    sum_y_square = (neg**2).sum()
    num = sum_prod - torch.mul(sum_x,sum_y)/pos.size(-1)
    den = torch.sqrt((sum_x_square-sum_x**2/pos.size(-1))*(sum_y_square-sum_y**2/pos.size(-1)))
    loss = torch.div(num,den+epsilon)
    return loss

def CC_diff(pos,neg,gt_diff):
    loss = 0
    threshold = 0.1
    batch, seq, w, h = pos.size()
    count = 0
    for i in range(batch):
        for j in range(seq):
            pos_ = pos[i,j][gt_diff[i,j]>threshold]
            neg_ = neg[i,j][gt_diff[i,j]>threshold]
            if len(pos_) > 1:
                pos_ = torch.mul(pos_,gt_diff[i,j][gt_diff[i,j]>threshold])
                neg_ = torch.mul(neg_,gt_diff[i,j][gt_diff[i,j]>threshold])
                loss = loss + CC_1d(pos_,neg_)
                count += 1            

    if count>1:
        return loss/count
    else:
        return torch.zeros(1).mean().cuda()

def Corr_diff(pred_diff,gt_diff,abs_diff):
    loss = 0
    threshold = 0.4
    count = 0 
    batch, seq, w, h = pred_diff.size()
    for i in range(batch):
        for j in range(seq):
            pred_diff_ = pred_diff[i,j][abs_diff[i,j]>threshold]
            gt_diff_ = gt_diff[i,j][abs_diff[i,j]>threshold]
            if len(pred_diff_)>1:
                loss = loss + CC_1d(pred_diff_,gt_diff_)
                count += 1
    if count>1:
        return loss/count
    else:
        return torch.zeros(1).mean().cuda()


def SIM(input,fixmap): 
    input = input.view(input.size(0), input.size(1), -1)
    fixmap = fixmap.view(fixmap.size(0),fixmap.size(1),-1)
    fixmap = torch.div(fixmap,fixmap.sum(-1,keepdim=True).expand_as(fixmap))
    input = torch.div(input,input.sum(-1,keepdim=True).expand_as(input))
    loss = torch.min(input,fixmap).sum(-1)
    return torch.mean(loss)

def KLD(input,fixmap,weight=None):
    if weight is not None:
        input = torch.mul(input,weight)
        fixmap = torch.mul(fixmap,weight)
    input = input.view(input.size(0), input.size(1), -1)
    # input = torch.div(input,input.max(-1,keepdim=True)[0].expand_as(input)+epsilon)
    fixmap = fixmap.view(fixmap.size(0),fixmap.size(1),-1)
    fixmap = torch.div(fixmap,fixmap.sum(-1,keepdim=True).expand_as(fixmap))
    input = torch.div(input,input.sum(-1,keepdim=True).expand_as(input))
    loss = torch.mul(fixmap,torch.log(torch.div(fixmap,input+epsilon) + epsilon)).sum(-1)

    return torch.mean(loss)

def KLD_1d(pos,neg):
    pos = torch.div(pos,pos.sum()+epsilon)
    neg = torch.div(neg,neg.sum()+epsilon)
    loss = torch.mul(neg,torch.log(torch.div(neg,pos+epsilon) + epsilon)).sum(-1)
    return loss

def KLD_diff(pos,neg,gt_diff):
    loss = 0
    batch, seq, w, h = pos.size()
    for i in range(batch):
        for j in range(seq):
            pos_ = pos[i,j][gt_diff[i,j]>0]
            neg_ = neg[i,j][gt_diff[i,j]>0]
            pos_ = torch.mul(pos_,gt_diff[i,j][gt_diff[i,j]>0])
            neg_ = torch.mul(neg_,gt_diff[i,j][gt_diff[i,j]>0])
            loss = loss + KLD_1d(pos_,neg_)
    return loss/(batch*seq)

def distortion_corr_seq(input,weight_map):
    return torch.mul(input,weight_map.expand_as(input))

def cosine_sim(input,target):
    batch,seq,h,w = input.size()
    input = input.view(batch,seq,-1)
    target = target.view(batch,seq,-1)
    input = torch.div(input,input.max(-1,keepdim=True)[0].expand_as(input))
    target = torch.div(target,target.max(-1,keepdim=True)[0].expand_as(target))
    # input = torch.div(input-input.mean(-1,keepdim=True).expand_as(input),input.std(-1,keepdim=True).expand_as(input))
    # target = torch.div(target-target.mean(-1,keepdim=True).expand_as(target),target.std(-1,keepdim=True).expand_as(target))
    loss = F.cosine_similarity(input,target,dim=-1)
    return loss.mean()

def cosine_sim_threshold(input,target,abs_diff):
    loss = 0
    threshold = 0.3
    count = 0 
    batch,seq,h,w = input.size()
    input = input.view(batch,seq,-1)
    target = target.view(batch,seq,-1)
    abs_diff = abs_diff.view(batch,seq,-1)
    input = torch.div(input,input.max(-1,keepdim=True)[0].expand_as(input))
    target = torch.div(target,target.max(-1,keepdim=True)[0].expand_as(target))
    for i in range(batch):
        for j in range(seq):
            input_ = input[i,j][abs_diff[i,j]>threshold]
            target_ = target[i,j][abs_diff[i,j]>threshold]
            if len(input_)>1:
                loss = loss + F.cosine_similarity(input_,target_,dim=0)
                count += 1
    if count>1:
        return loss/count
    else:
        return torch.zeros(1).mean().cuda()


def mse_diff(pos,neg,gt_diff):
    loss = 0
    threshold = 0.2
    count = 0
    batch, seq, w, h = pos.size()
    for i in range(batch):
        for j in range(seq):
            pos_ = pos[i,j][gt_diff[i,j]>threshold]
            neg_ = neg[i,j][gt_diff[i,j]>threshold]
            if len(pos_) > 1:
                pos_ = torch.div(pos_,pos_.max()+epsilon)
                neg_ = torch.div(neg_,neg_.max()+epsilon)
                weight = gt_diff[i,j][gt_diff[i,j]>threshold]
                loss = loss + (0.5*weight*(pos_-neg_)**2).sum()
                count += 1
    if count>1:
        return loss/count
    else:
        return torch.zeros(1).mean().cuda()


class Spherical_sal_loss(nn.Module):
    def __init__(self,height,width):
        super(Spherical_sal_loss,self).__init__()
        weight = np.sin((np.arange(height)/height)*np.pi).reshape(-1,1)
        weight = np.repeat(weight,width,axis=1).astype('float32')
        weight = torch.from_numpy(weight).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self,input,sal,fix):
        input = torch.mul(input,self.weight.expand_as(input))
        sal = torch.mul(sal,self.weight.expand_as(sal))
        fix = torch.mul(fix,self.weight.expand_as(fix))
        return -0.1*NSS(input,fix) + KLD(input,sal) - 0.1*CC(input,sal)


def LL(input,fixmap):
    input = input.view(input.size(0), input.size(1),-1)
    input = F.softmax(input,dim=-1)
    fixmap = fixmap.view(fixmap.size(0),fixmap.size(1),-1)
    loss =  torch.mul(torch.log(input+epsilon),fixmap).sum(-1)

    return -torch.sum(loss)

def difference_loss_mse(input,target,weight=None):
    input = 2*input[:,:,0]-input[:,:,1]
    if weight is None:
        loss = 0.5*(input-target)**2 # originally without 0.5
    else:
        loss = 0.5*weight*(input-target)**2 # originally without 0.5
    return loss.sum(-1).sum(-1).mean()


def horizontal_mse(input,target):
    input = input.sum(-2)
    target = target.sum(-2)
    loss = (input-target)**2
    # loss = loss.max(-1)[0]
    return loss.mean()

def vertical_mse(input,target):
    input = input.sum(-1)
    target = target.sum(-1)
    loss = (input-target)**2
    loss = loss.max(-1)[0]
    return loss.mean()


def difference_loss_bce(input,target):
    input = (input[:,:,0]-input[:,:,1]).view(input.size(0),input.size(1),-1)
    target = target.view(target.size(0),target.size(1),-1)
    input = 1/(1+torch.exp(-input / (input.std(-1,keepdim=True).expand_as(input)+epsilon)))
    target = 1/(1+torch.exp(-target / (target.std(-1,keepdim=True).expand_as(target)+epsilon)))
    loss = target*torch.log(torch.clamp(input,min=epsilon,max=1)) + (1-target)*torch.log(1-torch.clamp(input,min=epsilon,max=1))
    loss = -torch.mean(loss)

    return loss


class Discriminator(nn.Module):
    def __init__(self,):
        super(Discriminator,self).__init__()
        # self.conv1 = nn.Conv2d(2,32,kernel_size=3,padding=1,stride=1,bias=True)
        # self.conv2 = nn.Conv2d(32,16,kernel_size=1,padding=0,stride=1,bias=True)
        #self.pred = nn.Linear(512,1,bias=True)
        self.conv1 = nn.Conv2d(2,64,kernel_size=3,padding=1,stride=1)
        self.conv2 = nn.Conv2d(64,128,kernel_size=3,padding=1,stride=1)
        self.pred = nn.Linear(128,1,bias=True)


    def forward(self,x,y):
        batch, seq, h, w = x.size()
        x = x.view(batch*seq,h,w)
        y = y.view(batch*seq,h,w)
        x = torch.cat((x.unsqueeze(1),y.unsqueeze(1)),dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,kernel_size=(16,32))
        # x = F.max_pool2d(x,kernel_size=4)
        x = torch.sigmoid(self.pred(x.view(x.size(0),-1)))
        x = x.view(batch,seq,-1)
        return x

class Discriminator_warpper(nn.Module):
    def __init__(self,discriminator_1,discriminator_2):
        super(Discriminator_warpper,self).__init__()
        self.discriminator_1 = discriminator_1
        self.discriminator_2 = discriminator_2

    def forward(self,x_diff,y_diff,x_pos,y_pos,x_neg,y_neg):
        pred_diff = self.discriminator_1(x_diff,y_diff)
        pred_pos = self.discriminator_2(x_pos,y_pos)
        pred_neg = self.discriminator_2(x_neg,y_neg)
        return pred_diff, pred_pos, pred_neg


