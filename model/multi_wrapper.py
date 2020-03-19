import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np
from skipthoughts import BayesianUniSkip
from torchvision import models

epsilon = 1e-16

class Attention(nn.Module):
	def __init__(self,):
		super(Attention,self).__init__()
		self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.conv1 = nn.Conv2d(512,64,kernel_size=1,stride=1,padding=0,bias=True)
		self.conv2 = nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1,bias=True)
		self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.conv3 = nn.Conv2d(128,64,kernel_size=1,stride=1,padding=0,bias=True)
		self.conv4 = nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1,bias=True)
		self.conv5 = nn.Conv2d(128,1,kernel_size=1,stride=1,padding=0,bias=True)

	def forward(self,x):
		x = self.pool1(x)
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = self.pool2(x)
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = F.relu(self.conv5(x))
		x = F.interpolate(x,scale_factor=4)
		return torch.sigmoid(x)

class dcn_vgg(nn.Module):
	def __init__(self,):
		super(dcn_vgg,self).__init__()
		self.backbone = nn.Sequential(*list(models.vgg16(pretrained=True).features.children())[:-1])
		self.spatial_attention = Attention()

	def forward(self,x):
		x = self.backbone(x)
		att = self.spatial_attention(x)
		x = x + torch.mul(x,att.expand_as(x))
		return x


class Conv_LSTM_multi(nn.Module):
	def __init__(self,embed_size=256):
		super(Conv_LSTM_multi,self).__init__()
		self.input_x = nn.Conv2d(512,embed_size,kernel_size=3,stride=1,padding=1,bias=True)
		self.forget_x = nn.Conv2d(512,embed_size,kernel_size=3,stride=1,padding=1,bias=True)
		self.output_x = nn.Conv2d(512,embed_size,kernel_size=3,stride=1,padding=1,bias=True)
		self.memory_x = nn.Conv2d(512,embed_size,kernel_size=3,stride=1,padding=1,bias=True)
		self.input_h = nn.Conv2d(embed_size,embed_size,kernel_size=3,stride=1,padding=1,bias=True)
		self.forget_h = nn.Conv2d(embed_size,embed_size,kernel_size=3,stride=1,padding=1,bias=True)
		self.output_h = nn.Conv2d(embed_size,embed_size,kernel_size=3,stride=1,padding=1,bias=True)
		self.memory_h = nn.Conv2d(embed_size,embed_size,kernel_size=3,stride=1,padding=1,bias=True)
		self.input_c = nn.Conv2d(embed_size,embed_size,kernel_size=3,stride=1,padding=1,bias=True)
		self.forget_c = nn.Conv2d(embed_size,embed_size,kernel_size=3,stride=1,padding=1,bias=True)
		self.output_c = nn.Conv2d(embed_size,embed_size,kernel_size=3,stride=1,padding=1,bias=True)

		self.input_pos = nn.Linear(embed_size,embed_size)
		self.forget_pos = nn.Linear(embed_size,embed_size)
		self.output_pos = nn.Linear(embed_size,embed_size)
		self.input_neg = nn.Linear(embed_size,embed_size)
		self.forget_neg = nn.Linear(embed_size,embed_size)
		self.output_neg = nn.Linear(embed_size,embed_size)


	def forward(self,x,state,semantic_pos,semantic_neg):
		h, c = state
		i = torch.sigmoid(self.input_x(x) + self.input_h(h) + self.input_c(c) + self.input_pos(semantic_pos).unsqueeze(-1).unsqueeze(-1).expand_as(h)+self.input_neg(semantic_neg).unsqueeze(-1).unsqueeze(-1).expand_as(h))
		f = torch.sigmoid(self.forget_x(x) + self.forget_h(h) + self.forget_c(c) + self.forget_pos(semantic_pos).unsqueeze(-1).unsqueeze(-1).expand_as(h)+ self.forget_neg(semantic_neg).unsqueeze(-1).unsqueeze(-1).expand_as(h))
		g = torch.tanh(self.memory_x(x) + self.memory_h(h))

		next_c = torch.mul(f,c) + torch.mul(i,g)
		o = torch.sigmoid(self.output_x(x) + self.output_h(h) + self.output_c(next_c) + self.output_pos(semantic_pos).unsqueeze(-1).unsqueeze(-1).expand_as(h) + self.output_neg(semantic_neg).unsqueeze(-1).unsqueeze(-1).expand_as(h))
		h = torch.mul(o,next_c)
		state = (h,next_c)

		return state

class semantic_att(nn.Module):
	def __init__(self,embedding_size=512):
		super(semantic_att,self).__init__()
		self.semantic_q = nn.Linear(embedding_size,embedding_size)
		self.semantic_pool = nn.Linear(embedding_size,embedding_size)
		self.semantic_cur = nn.Linear(embedding_size,embedding_size)
		self.semantic_att = nn.Linear(embedding_size,1)

	def forward(self,semantic,q,cur_v):
		q = F.relu(self.semantic_q(q)).unsqueeze(1).expand_as(semantic)
		v_pool = F.relu(self.semantic_pool(semantic))
		v_cur = F.relu(self.semantic_cur(cur_v)).unsqueeze(1).expand_as(semantic)
		semantic_att = F.softmax(self.semantic_att(q+v_pool+v_cur),dim=1).expand_as(semantic)
		semantic = torch.mul(semantic_att,semantic).sum(1)

		return semantic

class SWM_agg(nn.Module):
	def __init__(self,embed_size=256,vocab=None):
		super(SWM_agg,self).__init__()
		self.backbone = dcn_vgg()
		self.embed_size = embed_size
		self.rnn = Conv_LSTM_multi(embed_size)
		self.q_model = BayesianUniSkip(dir_st='/srv/chenshi/data/Skip-thoughts-pretrained' ,vocab=vocab) # using pretrained skip-thought
		self.q_embed = nn.Linear(2400,embed_size) # skip-thought vectors have pre-defined size 2400
		self.sal_pred = nn.Conv2d(embed_size,2,kernel_size=1,stride=1,padding=0,bias=True)
		self.semantic_att = semantic_att(embed_size)
		self.semantic_embed = nn.Linear(512,embed_size)

		# weight for combining the two maps
		self.weight_q = nn.Linear(embed_size,embed_size)
		self.weight_v = nn.Linear(embed_size*2,embed_size)
		self.weight = nn.Linear(embed_size,2)

	def init_hidden_state(self,batch,height,width,embedding_size=256):
		init_h = torch.zeros(batch,embedding_size,height,width).cuda()
		init_m = torch.zeros(batch,embedding_size,height,width).cuda()
		return (init_h,init_m)

	def get_semantic(self,sal_map,semantic_feature):
		semantic_feature_pos = sal_map[:,0].unsqueeze(1).expand_as(semantic_feature) * semantic_feature
		semantic_feature_neg = sal_map[:,1].unsqueeze(1).expand_as(semantic_feature) * semantic_feature
		semantic_feature_pos = semantic_feature_pos.view(semantic_feature.size(0),semantic_feature.size(1),-1).mean(-1)  # for sigmoid output
		semantic_feature_neg = semantic_feature_neg.view(semantic_feature.size(0),semantic_feature.size(1),-1).mean(-1)  # for sigmoid output

		return semantic_feature_pos, semantic_feature_neg

	def forward(self,x,que):
		if len(que) == 1:
			q = self.q_model(que,[len(que[0])])
		else:
			q = self.q_model(que)
		q = F.relu(self.q_embed(q)) # for semantic attention with hadamard product

		batch, seq, c, h, w = x.size()
		state = self.init_hidden_state(batch,int(h/16),int(w/16),self.embed_size)
		semantic_mem_pos = torch.zeros(batch,self.embed_size).cuda()
		semantic_mem_neg = torch.zeros(batch,self.embed_size).cuda()

		output = []
		semantic_pool_pos = []
		semantic_pool_neg = []
		for i in range(seq):
			cur_v = x[:,i].contiguous()
			cur_v = self.backbone(cur_v)

			state = self.rnn(cur_v,state,semantic_mem_pos,semantic_mem_neg)

			cur_sal = state[0] + q.unsqueeze(-1).unsqueeze(-1).expand_as(state[0])
			cur_sal = self.sal_pred(cur_sal)
			# semantic memory module
			rand_prob = np.random.rand(1,)[0]
			sal_map = torch.sigmoid(cur_sal)
			semantic_feature_pos, semantic_feature_neg = self.get_semantic(sal_map,cur_v)
			semantic_feature_pos = F.relu(self.semantic_embed(semantic_feature_pos))
			semantic_feature_neg = F.relu(self.semantic_embed(semantic_feature_neg))

			semantic_pool_pos.append(semantic_feature_pos)
			semantic_pool_neg.append(semantic_feature_neg)
			semantic_mem_pos = self.semantic_att(torch.cat([ _.unsqueeze(1) for _ in semantic_pool_pos], 1),q,semantic_feature_pos)
			semantic_mem_neg = self.semantic_att(torch.cat([ _.unsqueeze(1) for _ in semantic_pool_neg], 1),q,semantic_feature_neg)

			cur_sal = torch.sigmoid(cur_sal)
			# obtaining the aggregated saliency map
			weight_q = F.relu(self.weight_q(q))
			weight_v = F.relu(self.weight_v(torch.cat((semantic_mem_pos,semantic_mem_neg),dim=-1)))
			cur_weight = torch.sigmoid(self.weight(weight_q+weight_v))
			cur_sal = torch.mul(cur_sal,cur_weight.unsqueeze(-1).unsqueeze(-1).expand_as(cur_sal))
			cur_sal = cur_sal[:,0] + cur_sal[:,1]
			batch_, h_, w_ = cur_sal.size()
			cur_sal = cur_sal.view(batch_,h_*w_)
			cur_sal = torch.div(cur_sal,cur_sal.max(-1,keepdim=True)[0].expand_as(cur_sal)+epsilon)
			cur_sal = cur_sal.view(batch_,h_,w_)
			output.append(cur_sal.unsqueeze(1))

		output = torch.cat([_ for _ in output],dim=1)

		return output
