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
from dataloader import conditional_loader
from SWM import SWM
from loss_seq import NSS, CC, KLD, difference_loss_mse
from evaluation import cal_cc_score, cal_sim_score, cal_kld_score, cal_auc_score, cal_nss_score, add_center_bias, distortion_corr, cal_fix_mse, cal_sauc_score

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
parser.add_argument('--lr_decay', type=float, default=0.25, help='Learning rate decay factor')
parser.add_argument('--lr_decay_step', type=int, default=30, help='Learning rate decay step')
parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
parser.add_argument('--que_len', type=int, default=20, help='Maximum length for the questions')
parser.add_argument('--temporal_step', type=int, default=5, help='Temporal_step for sampling the frames')
parser.add_argument('--historical_step', type=int, default=4, help='Number of historical_step')
parser.add_argument('--resume', type=bool, default=False, help='Resume or not')
parser.add_argument('--width', type=int, default=512, help='Width of input')
parser.add_argument('--height', type=int, default=256, help='Height of input')
parser.add_argument('--center_bias', type=bool, default=True, help='Using center bias or not')
parser.add_argument('--alpha', type=float, default=0.5, help='Balance factor for the difference loss')
parser.add_argument('--save_dir', type=str, default=None, help='Directory for saving the predicted maps')


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
    if epoch >= 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * (args.lr_decay ** int(epoch/args.lr_decay_step))


def main():
    # IO
    tf_summary_writer = tf.summary.create_file_writer(args.checkpoint)
    train_data = conditional_loader(args.img_dir,args.sal_dir,args.que_file,args.word2idx,'train',args.split_info,args.que_len,args.temporal_step,args.historical_step,transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=1)
    test_data = conditional_loader(args.img_dir,args.sal_dir,args.que_file,args.word2idx,'test',args.split_info,args.que_len,args.temporal_step,args.historical_step,transform,test_mode=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch, shuffle=False, num_workers=1) # fix the batch size for evaluation as 1
    # model construction
    model = SWM(embed_size=args.embedding_size,vocab=train_data.word2idx)
    model = model.cuda()

    if args.resume:
        print('resumed from previous checkpoint')
        model.load_state_dict(torch.load(os.path.join(args.checkpoint,'model_best.pth')))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3) #1e-3 weight decay


    def train(iteration,epoch):
        model.train()
        avg_pos_loss = 0
        avg_neg_loss = 0
        avg_diff_loss = 0
        distor_corr_weight = torch.from_numpy(((np.arange(16)/16)*np.pi).astype('float32'))
        distor_corr_weight = torch.sin(distor_corr_weight).view(-1,1).expand(-1,32).unsqueeze(0).unsqueeze(0).cuda()

        # avg_vertical_loss = 0
        for i, (img, que, cur_sal_pos, cur_fix_pos, cur_sal_neg, cur_fix_neg) in enumerate(trainloader):
            if len(img) < args.batch:
                continue
            img, que, cur_sal_pos, cur_fix_pos, cur_sal_neg, cur_fix_neg = img.cuda(), que.cuda(), cur_sal_pos.cuda(), cur_fix_pos.cuda(), cur_sal_neg.cuda(), cur_fix_neg.cuda()
            optimizer.zero_grad()
            pred = model(img,que)

            pos_loss = -0.1*NSS(pred[:,:,0],cur_fix_pos) + KLD(pred[:,:,0],cur_sal_pos) - 0.1*CC(pred[:,:,0],cur_sal_pos) # AclNet setting
            neg_loss = -0.1*NSS(pred[:,:,1],cur_fix_neg) + KLD(pred[:,:,1],cur_sal_neg) - 0.1*CC(pred[:,:,1],cur_sal_neg) # AclNet setting
            diff_loss = 2*difference_loss_mse(pred,cur_sal_pos-cur_sal_neg,weight=torch.abs(cur_fix_pos-cur_fix_neg)) + CC(pred[:,:,0],pred[:,:,1],weight=torch.abs(cur_sal_pos-cur_sal_neg))
            loss = pos_loss + neg_loss + args.alpha*diff_loss
            loss.backward()

            if args.clip != -1 :
                clip_gradient(optimizer,args.clip) #gradient clipping without normalization
            optimizer.step()

            avg_pos_loss = (avg_pos_loss*np.maximum(0,i) + pos_loss.data.cpu().numpy())/(i+1)
            avg_neg_loss = (avg_neg_loss*np.maximum(0,i) + neg_loss.data.cpu().numpy())/(i+1)
            avg_diff_loss = (avg_diff_loss*np.maximum(0,i) + diff_loss.data.cpu().numpy())/(i+1)


            if i%5 == 0:
                with tf_summary_writer.as_default():
                    tf.summary.scalar('positive loss',avg_pos_loss,step=iteration)
                    tf.summary.scalar('negative loss',avg_neg_loss,step=iteration)
                    tf.summary.scalar('difference loss',avg_diff_loss,step=iteration)

            iteration += 1
        return iteration

    def validation(iteration):
        # initialize evaluation score
        model.eval()
        eval_score = dict()
        label_pool = ['correct','incorrect']

        for cond_label in label_pool:
            eval_score[cond_label] = dict()
            for metric in eval_metrics:
                eval_score[cond_label][metric] = []

        for i, (img, que, cur_sal_pos, cur_fix_pos, cur_sal_neg, cur_fix_neg) in enumerate(testloader):
            # iterate through different frames of the same video sequence
            img, que = img.cuda(), que.cuda()
            pred = model(img,que)

            sal, fix = [], []
            if len(pred) > 1:
                pred = pred.data.cpu().numpy().squeeze()
                sal.append(cur_sal_pos.numpy().squeeze())
                sal.append(cur_sal_neg.numpy().squeeze())
                fix.append(cur_fix_pos.numpy().squeeze())
                fix.append(cur_fix_neg.numpy().squeeze())
            else:
                pred = pred.data.cpu().numpy()
                sal.append(cur_sal_pos.numpy())
                sal.append(cur_sal_neg.numpy())
                fix.append(cur_fix_pos.numpy())
                fix.append(cur_fix_neg.numpy())

            for j in range(len(pred)):
                for k in range(len(pred[j])):
                    for cond_idx, cond_label in enumerate(label_pool):
                        cur_pred = pred[j,k,cond_idx] # only evaluate on the current frame
                        cur_pred = cv2.resize(cur_pred,(256,128))
                        cur_sal = sal[cond_idx][j,k]
                        cur_fix = fix[cond_idx][j,k]

                        cur_pred = distortion_corr(cur_pred)
                        cur_sal = distortion_corr(cur_sal)
                        cur_fix = distortion_corr(cur_fix)

                        if args.center_bias:
                            cur_pred = add_center_bias(cur_pred)
                        eval_score[cond_label]['cc'].append(cal_cc_score(cur_pred,cur_sal))
                        eval_score[cond_label]['sim'].append(cal_sim_score(cur_pred,cur_sal))
                        eval_score[cond_label]['kld'].append(cal_kld_score(cur_pred,cur_sal))
                        eval_score[cond_label]['nss'].append(cal_nss_score(cur_pred,cur_fix))
                        eval_score[cond_label]['sauc'].append(cal_sauc_score(cur_pred,cur_fix,test_data.shuf_map[cond_idx]))


        with tf_summary_writer.as_default():
            for cond_label in eval_score.keys():
                for metric in eval_score[cond_label].keys():
                    tf.summary.scalar(cond_label+'_'+metric.upper(),np.mean(eval_score[cond_label][metric]),step=iteration)

        return np.mean(eval_score['correct']['cc'])


    iteration = 0
    best_score = 0

    for epoch in range(args.epoch):
        adjust_learning_rate(optimizer, epoch+1)
        iteration = train(iteration,epoch+1)
        cur_score = validation(iteration)
        torch.save(model.state_dict(),os.path.join(args.checkpoint,'model.pth')) # for single-GPU training
        if cur_score > best_score:
            best_score = cur_score
            torch.save(model.state_dict(),os.path.join(args.checkpoint,'model_best.pth')) # for single-GPU training

def evaluation():
    test_data = conditional_loader(args.img_dir,args.sal_dir,args.que_file,args.word2idx,'test',args.split_info,args.que_len,args.temporal_step,args.historical_step,transform,test_mode=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch, shuffle=False, num_workers=1) # fix the batch size for evaluation as 1

    model = SWM(embed_size=args.embedding_size,vocab=train_data.word2idx)
    model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(args.checkpoint,'model_best.pth')))

    model.eval()
    eval_score = dict()
    label_pool = ['correct','incorrect']

    for cond_label in label_pool:
        eval_score[cond_label] = dict()
        for metric in eval_metrics:
            eval_score[cond_label][metric] = []

    for i, (img, que, cur_sal_pos, cur_fix_pos, cur_sal_neg, cur_fix_neg) in enumerate(testloader):
        # iterate through different frames of the same video sequence
        img, que = img.cuda(), que.cuda()
        pred = model(img,que)

        sal, fix = [], []
        if len(pred) > 1:
            pred = pred.data.cpu().numpy().squeeze()
            sal.append(cur_sal_pos.numpy().squeeze())
            sal.append(cur_sal_neg.numpy().squeeze())
            fix.append(cur_fix_pos.numpy().squeeze())
            fix.append(cur_fix_neg.numpy().squeeze())
        else:
            pred = pred.data.cpu().numpy()
            sal.append(cur_sal_pos.numpy())
            sal.append(cur_sal_neg.numpy())
            fix.append(cur_fix_pos.numpy())
            fix.append(cur_fix_neg.numpy())

        for j in range(len(pred)):
            for k in range(len(pred[j])):
                for cond_idx, cond_label in enumerate(label_pool):
                    cur_pred = pred[j,k,cond_idx] # only evaluate on the current frame
                    cur_pred = cv2.resize(cur_pred,(256,128))
                    cur_sal = sal[cond_idx][j,k]
                    cur_fix = fix[cond_idx][j,k]

                    cur_pred = distortion_corr(cur_pred)
                    cur_sal = distortion_corr(cur_sal)
                    cur_fix = distortion_corr(cur_fix)

                    if args.center_bias:
                        cur_pred = add_center_bias(cur_pred)
                    eval_score[cond_label]['cc'].append(cal_cc_score(cur_pred,cur_sal))
                    eval_score[cond_label]['sim'].append(cal_sim_score(cur_pred,cur_sal))
                    eval_score[cond_label]['kld'].append(cal_kld_score(cur_pred,cur_sal))
                    eval_score[cond_label]['nss'].append(cal_nss_score(cur_pred,cur_fix))
                    eval_score[cond_label]['sauc'].append(cal_sauc_score(cur_pred,cur_fix,test_data.shuf_map[cond_idx]))

    for cond_label in eval_score.keys():
        print('Evaluation scores for %s attention' %cond_label)
        for metric in eval_score[cond_label].keys():
            print('%s: %f' %(metric.upper(),np.mean(eval_score[cond_label][metric])))
        print('\n')

if args.mode == 'train':
    main()
else:
    evaluation()
