import sys
sys.path.append('./util')
sys.path.append('./model')
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import argparse
import os
import time
import gc
import cv2
import json
import tensorflow as tf
from dataloader import aggregated_loader
from multi_wrapper import SWM_agg
from loss_seq import NSS, CC, KLD
from evaluation import cal_cc_score, cal_sim_score, cal_kld_score, cal_auc_score, cal_nss_score, add_center_bias, distortion_corr,cal_sauc_score

parser = argparse.ArgumentParser(description='Predicting the correctness of reasoning based on visual behaviors')
parser.add_argument('--mode', type=str, default='train', help='Selecting running mode (default: train)')
parser.add_argument('--img_dir', type=str, default=None, help='Directory to the image data')
parser.add_argument('--sal_dir', type=str, default=None, help='Directory to the saliency data')
parser.add_argument('--que_file', type=str, default=None, help='Directory to the question information file')
parser.add_argument('--word2idx', type=str, default=None, help='Directory to the word2idx')
parser.add_argument('--split_info', type=str, default='splits_sal.json', help='Directory to the split file')
parser.add_argument('--word_size', type=int, default=620, help='Size of word embedding')
parser.add_argument('--embedding_size', type=int, default=512, help='Size of hidden state')
parser.add_argument('--clip', type=float, default=-1, help='Gradient clipping')
parser.add_argument('--batch', type=int, default=1, help='Batch size')
parser.add_argument('--epoch', type=int, default=20, help='Number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--lr_decay', type=float, default=0.5, help='Learning rate decay factor')
parser.add_argument('--lr_decay_step', type=int, default=60, help='Learning rate decay step')
parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
parser.add_argument('--que_len', type=int, default=20, help='Maximum length for the questions')
parser.add_argument('--temporal_step', type=int, default=5, help='Temporal_step for sampling the frames')
parser.add_argument('--historical_step', type=int, default=4, help='Number of historical_step')
parser.add_argument('--resume', type=bool, default=False, help='Resume or not')
parser.add_argument('--width', type=int, default=512, help='Width of input')
parser.add_argument('--height', type=int, default=256, help='Height of input')
parser.add_argument('--center_bias', type=bool, default=True, help='Using center bias or not')
args = parser.parse_args()

eval_metrics = ['cc','nss','sim','kld','sauc']

transform = transforms.Compose([
                                transforms.Resize((args.height,args.width)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_learning_rate(optimizer, epoch):
    "adatively adjust lr based on iteration"
    if epoch >= 1: #30-adam
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * (args.lr_decay ** (epoch/args.lr_decay_step))

def main():
    # IO
    tf_summary_writer = tf.summary.create_file_writer(args.checkpoint)
    train_data = aggregated_loader(args.img_dir,args.sal_dir,args.que_file,args.word2idx,'train',args.split_info,args.que_len,args.temporal_step,args.historical_step,transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=1)
    test_data = aggregated_loader(args.img_dir,args.sal_dir,args.que_file,args.word2idx,'val',args.split_info,args.que_len,args.temporal_step,args.historical_step,transform,test_mode=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch, shuffle=False, num_workers=1) # fix the batch size for evaluation as 1

    model = SWM_agg(embed_size=args.embedding_size,vocab=train_data.word2idx)
    model.load_state_dict(torch.load(os.path.join(args.checkpoint,'pretrained.pth')),strict=False)
    model = model.cuda()

    if args.resume:
        model.load_state_dict(torch.load(os.path.join(args.checkpoint,'model.pth')))
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-3, nesterov=True)

    def train(iteration):
        model.train()
        avg_loss = 0
        for i, (img, que, cur_sal, fix) in enumerate(trainloader):
            if len(img) < args.batch:
                continue
            img, que, cur_sal, fix = img.cuda(), que.cuda(), cur_sal.cuda(), fix.cuda()
            optimizer.zero_grad()
            pred = model(img,que)
            loss = -0.1*NSS(pred,fix) + KLD(pred,cur_sal) - 0.1*CC(pred,cur_sal) # AclNet setting
            loss.backward()

            if args.clip != -1 :
                clip_gradient(optimizer,args.clip) #gradient clipping without normalization
            optimizer.step()

            avg_loss = (avg_loss*np.maximum(0,i) + loss.data.cpu().numpy())/(i+1)

            if i%5 == 0:
                with tf_summary_writer.as_default():
                    tf.summary.scalar('training loss',avg_loss,step=iteration)
            iteration += 1
        return iteration

    def validation(iteration):
        # initialize evaluation score
        model.eval()
        eval_score = dict()

        for metric in eval_metrics:
            eval_score[metric] = []

        for i, (img, que, sal, fix) in enumerate(testloader):
            # iterate through different frames of the same video sequence
            img, que = img.cuda(), que.cuda()
            pred = model(img,que)

            if len(pred) > 1:
                pred = pred.data.cpu().numpy().squeeze()
                sal = sal.numpy().squeeze()
                fix = fix.numpy().squeeze()
            else:
                pred = pred.data.cpu().numpy()
                sal = sal.numpy()
                fix = fix.numpy()
            for j in range(len(pred)):
                for k in range(len(pred[j])):
                    cur_pred = pred[j,k] # evaluate on the current frame
                    cur_pred = cv2.resize(cur_pred,(256,128))
                    if cur_pred.max()>0:
                        cur_pred /= cur_pred.max()

                    cur_sal = sal[j,k]
                    cur_fix = fix[j,k]
                    cur_pred = distortion_corr(cur_pred)
                    cur_sal = distortion_corr(cur_sal)
                    cur_fix = distortion_corr(cur_fix)

                    if args.center_bias:
                        cur_pred = add_center_bias(cur_pred)
                    eval_score['cc'].append(cal_cc_score(cur_pred,cur_sal))
                    eval_score['sim'].append(cal_sim_score(cur_pred,cur_sal))
                    eval_score['kld'].append(cal_kld_score(cur_pred,cur_sal))
                    eval_score['nss'].append(cal_nss_score(cur_pred,cur_fix))
                    eval_score['sauc'].append(cal_sauc_score(cur_pred,cur_fix,test_data.shuf_map))


        with tf_summary_writer.as_default():
            for metric in eval_score.keys():
                tf.summary.scalar(metric.upper(),np.mean(eval_score[metric]),step=iteration)

        return np.mean(eval_score['cc'])


    iteration = 0
    best_score = 0
    for epoch in range(args.epoch):
        adjust_learning_rate(optimizer, epoch+1)
        iteration = train(iteration)
        cur_score = validation(iteration)
        torch.save(model.state_dict(),os.path.join(args.checkpoint,'model.pth')) # for single-GPU training
        if cur_score > best_score:
            best_score = cur_score
            torch.save(model.state_dict(),os.path.join(args.checkpoint,'model_best.pth')) # for single-GPU training

# evaluation-only
def evaluation():
    test_data = aggregated_loader(args.img_dir,args.sal_dir,args.que_file,args.word2idx,'test',args.split_info,args.que_len,args.temporal_step,args.historical_step,transform,test_mode=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch, shuffle=False, num_workers=1) # fix the batch size for evaluation as 1
    model = SWM_agg(embed_size=args.embedding_size,vocab=train_data.word2idx)
    model.load_state_dict(torch.load(os.path.join(args.checkpoint,'model_best.pth')),strict=True)
    model = model.cuda()

    model.eval()
    eval_score = dict()
    for metric in eval_metrics:
        eval_score[metric] = []

    for i, (img, que, sal, fix) in enumerate(testloader):
        # iterate through different frames of the same video sequence
        img, que = img.cuda(), que.cuda()
        pred = model(img,que)

        if len(pred) > 1:
            pred = pred.data.cpu().numpy().squeeze()
            sal = sal.numpy().squeeze()
            fix = fix.numpy().squeeze()
        else:
            pred = pred.data.cpu().numpy()
            sal = sal.numpy()
            fix = fix.numpy()
        for j in range(len(pred)):
            for k in range(len(pred[j])):
                cur_pred = pred[j,k] # evaluate on the current frame
                cur_pred = cv2.resize(cur_pred,(256,128))
                if cur_pred.max()>0:
                    cur_pred /= cur_pred.max()

                cur_sal = sal[j,k]
                cur_fix = fix[j,k]
                cur_pred = distortion_corr(cur_pred)
                cur_sal = distortion_corr(cur_sal)
                cur_fix = distortion_corr(cur_fix)

                if args.center_bias:
                    cur_pred = add_center_bias(cur_pred)
                eval_score['cc'].append(cal_cc_score(cur_pred,cur_sal))
                eval_score['sim'].append(cal_sim_score(cur_pred,cur_sal))
                eval_score['kld'].append(cal_kld_score(cur_pred,cur_sal))
                eval_score['nss'].append(cal_nss_score(cur_pred,cur_fix))
                eval_score['sauc'].append(cal_sauc_score(cur_pred,cur_fix,test_data.shuf_map))

    print('Evaluation scores for aggregated attention')
    for metric in eval_score.keys():
        print('%s: %d' %(metric.upper(),np.mean(eval_score[metric])))

if args.mode == 'train':
    main()
else:
    evaluation()
