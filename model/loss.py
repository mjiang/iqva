import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from numpy import pi
import math

epsilon = 1e-16 #regularization value in Keras

def NSS(input,fixation):    
    input = input.view(input.size(0),-1)
    # input = torch.div(input,input.max(-1,keepdim=True)[0].expand_as(input)+epsilon)
    fixation = fixation.view(fixation.size(0),-1)
    input = torch.div(input-input.mean(-1,keepdim=True).expand_as(input),input.std(-1,keepdim=True).expand_as(input) + epsilon)
    loss = torch.div(torch.mul(input,fixation).sum(-1), fixation.sum(-1) + epsilon)

    return torch.mean(loss)

def CC(input,fixmap,weight=None): 
    if weight is not None:
        input = torch.mul(input,weight)
        fixmap = torch.mul(fixmap,weight)
    input = input.view(input.size(0),-1)

    # input = torch.div(input,input.max(-1,keepdim=True)[0].expand_as(input)+epsilon)
    # fixmap = torch.div(fixmap,fixmap.max(-1,keepdim=True)[0].expand_as(fixmap)+epsilon)

    fixmap = fixmap.view(fixmap.size(0),-1)
    fixmap = torch.div(fixmap,fixmap.sum(-1,keepdim=True).expand_as(fixmap)+epsilon)
    input = torch.div(input,input.sum(-1,keepdim=True).expand_as(input)+epsilon)

    sum_prod = torch.mul(input,fixmap).sum(-1,keepdim=True)
    sum_x = input.sum(-1,keepdim=True)
    sum_y = fixmap.sum(-1,keepdim=True)
    sum_x_square = (input**2).sum(-1,keepdim=True)
    sum_y_square = (fixmap**2).sum(-1,keepdim=True)
    num = sum_prod - torch.mul(sum_x,sum_y)/input.size(-1)
    den = torch.sqrt((sum_x_square-sum_x**2/input.size(-1))*(sum_y_square-sum_y**2/input.size(-1)))
    loss = torch.div(num,den+epsilon)

    return torch.mean(loss)

def KLD(input,fixmap,weight=None):
    input = input.view(input.size(0),-1)
    # input = torch.div(input,input.max(-1,keepdim=True)[0].expand_as(input)+epsilon)
    fixmap = fixmap.view(fixmap.size(0),-1)
    fixmap = torch.div(fixmap,fixmap.sum(-1,keepdim=True).expand_as(fixmap))
    input = torch.div(input,input.sum(-1,keepdim=True).expand_as(input))
    loss = torch.mul(fixmap,torch.log(torch.div(fixmap,input+epsilon) + epsilon))
    if weight is not None:
        weight = weight.view(weight.size(0),-1)
        loss = torch.mul(loss,weight).sum(-1)
    else:
        loss = loss.sum(-1)

    return torch.mean(loss)

def SIM(input,fixmap):
    input = input.view(input.size(0), -1)
    fixmap = fixmap.view(fixmap.size(0),-1)
    fixmap = torch.div(fixmap,fixmap.sum(-1,keepdim=True).expand_as(fixmap))
    input = torch.div(input,input.sum(-1,keepdim=True).expand_as(input))
    loss = torch.min(input,fixmap).sum(-1)
    return torch.mean(loss)


def LL(input,fixmap):
    input = input.view(input.size(0),-1)
    input = F.softmax(input,dim=-1)
    fixmap = fixmap.view(fixmap.size(0),-1)
    loss =  torch.mul(torch.log(input+epsilon),fixmap).sum(-1)

    return -torch.sum(loss)


def cross_entropy(input,target):
    input = input.view(input.size(0), -1)
    input = F.softmax(input,dim=-1)
    target = target.view(target.size(0),-1)
    loss = (-target*torch.log(torch.clamp(input,min=epsilon,max=1))).sum(-1)
    return loss.mean()

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
        normalized = out.view(out.size(0),-1).max(-1,keepdim=True)[0].unsqueeze(-1).unsqueeze(-1).expand_as(out)
        out = torch.div(out,normalized+epsilon)

        return torch.sum((out - target) ** 2 * self.weight) / out.size(0)

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
        loss = (-target*torch.log(torch.clamp(out,min=epsilon,max=1))) * self.weight
        return torch.sum(loss) / out.size(0)

def binary_cross_entropy(input,target):
    input = input.view(input.size(0), -1)
    target = target.view(target.size(0),-1)
    loss = target*torch.log(torch.clamp(input,min=epsilon,max=1)) + (1-target)*torch.log(1-torch.clamp(input,min=epsilon,max=1))
    loss = -torch.mean(loss)
    return loss

def difference_loss_bce(input,target,weight=None):
    input = (input[:,0]-input[:,1]).view(input.size(0),-1)
    target = target.view(target.size(0),-1)
    input = 1/(1+torch.exp(-input / (input.std(-1,keepdim=True).expand_as(input)+epsilon)))
    target = 1/(1+torch.exp(-target / (target.std(-1,keepdim=True).expand_as(target)+epsilon)))
    loss = target*torch.log(torch.clamp(input,min=epsilon,max=1)) + (1-target)*torch.log(1-torch.clamp(input,min=epsilon,max=1))
    if weight is not None:
        weight = weight.view(weight.size(0),-1)
        loss = torch.mul(loss,weight)
    loss = -torch.sum(loss)

    return loss

def mse(input,target):
    loss = (input-target)**2
    return loss.sum()

def non_linear_layer(input,no_batch=False):
    if no_batch:
        h, w = input.size()
        input = input.view(-1)
        input = 1/(1+torch.exp(-input / (input.std(-1,keepdim=True).expand_as(input)+epsilon)))
        return input.view(h,w)
    else:
        batch, h, w = input.size()
        input = input.view(batch, -1)
        input = 1/(1+torch.exp(-input / (input.std(-1,keepdim=True).expand_as(input)+epsilon)))
        return input.view(batch,h,w)

def difference_loss_mse(input,target,fixation):
    loss = torch.mul(fixation,(input-target)**2).sum(-1).sum(-1)
    return loss.sum()


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
        x = torch.cat((x.unsqueeze(1),y.unsqueeze(1)),dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,kernel_size=(16,32))
        # x = F.max_pool2d(x,kernel_size=4)
        x = torch.sigmoid(self.pred(x.view(x.size(0),-1)))
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