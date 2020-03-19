import os
from glob import glob
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import cv2
import json
import time

eps = 1e-16

class conditional_loader(data.Dataset):
	def __init__(self,img_dir,sal_dir,que_file,word2idx,split,split_info,que_len=15,temporal_step=5,historical_step=9,transform=None,test_mode=False):
		self.img_dir = img_dir
		self.sal_dir = sal_dir
		self.que_file = json.load(open(que_file))
		self.word2idx = json.load(open(word2idx))
		self.split = split
		self.split_info = json.load(open(split_info))
		self.que_len = que_len
		self.temporal_step = temporal_step
		self.historical_step = historical_step
		self.transform = transform
		self.test_mode = test_mode
		self.init_img_dir()
		self.init_data()

	def init_data(self,):
		self.img = []
		self.question = []
		self.sal_map = []
		self.fixation = []
		pos_shuf_map = np.zeros([128,256])
		neg_shuf_map = np.zeros([128,256])

		for cur_anno in self.que_file:
			cur_id = cur_anno['qid']
			if self.split_info[cur_id]['split'] != self.split or not self.split_info[cur_id]['valid_correctness']:
				continue
			saliency_pos = glob(os.path.join(self.sal_dir,cur_id,'saliency_map','correct','*'))
			saliency_neg = glob(os.path.join(self.sal_dir,cur_id,'saliency_map','incorrect','*'))
			fixation_pos = glob(os.path.join(self.sal_dir,cur_id,'fixation','correct','*'))
			fixation_neg = glob(os.path.join(self.sal_dir,cur_id,'fixation','incorrect','*'))
			frame_idx = [os.path.basename(cur)[:-4] for cur in saliency_pos]

			# downsampling the maps
			valid_frame = frame_idx[::self.temporal_step] # selecting valid frames with pre-defined step
			valid_saliency_map_pos = saliency_pos[::self.temporal_step]
			valid_saliency_map_neg = saliency_neg[::self.temporal_step]
			valid_fixation_map_pos = fixation_pos[::self.temporal_step]
			valid_fixation_map_neg = fixation_neg[::self.temporal_step]

			cur_que = cur_anno['question']
			cur_que = self.convert_idx(cur_que.split(' '))

			if self.test_mode:
				# exclude the first fixation to remove bias
				for i in range(self.historical_step+2,len(valid_frame),self.historical_step+1):
					self.img.append([os.path.join(self.img_dir,cur_id,str(cur_frame)+'.png') for cur_frame in valid_frame[i-self.historical_step-1:i]])
					self.question.append(cur_que)
					self.sal_map.append(([cur for cur in valid_saliency_map_pos[i-self.historical_step-1:i]],[cur for cur in valid_saliency_map_neg[i-self.historical_step-1:i]]))
					self.fixation.append(([cur for cur in valid_fixation_map_pos[i-self.historical_step-1:i]],[cur for cur in valid_fixation_map_neg[i-self.historical_step-1:i]]))

				if i < len(valid_frame)-1:
					self.img.append([os.path.join(self.img_dir,cur_id,str(cur_frame)+'.png') for cur_frame in valid_frame[len(valid_frame)-self.historical_step-1:]])
					self.question.append(cur_que)
					self.sal_map.append(([cur for cur in valid_saliency_map_pos[len(valid_frame)-self.historical_step-1:]],[cur for cur in valid_saliency_map_neg[len(valid_frame)-self.historical_step-1:]]))
					self.fixation.append(([cur for cur in valid_fixation_map_pos[len(valid_frame)-self.historical_step-1:]],[cur for cur in valid_fixation_map_neg[len(valid_frame)-self.historical_step-1:]]))

				# aggrate the fixations for sAUC evaluation
				for i in range(len(valid_saliency_map_pos[1:])):
						tmp_pos_fix = cv2.imread(valid_fixation_map_pos[i+1])[:,:,0].astype('float32')
						tmp_pos_fix[tmp_pos_fix>0] = 1
						pos_shuf_map += tmp_pos_fix
						tmp_neg_fix = cv2.imread(valid_fixation_map_neg[i+1])[:,:,0].astype('float32')
						tmp_neg_fix[tmp_neg_fix>0] = 1
						neg_shuf_map += tmp_neg_fix

				self.shuf_map = [pos_shuf_map,neg_shuf_map]

			else:
				self.img.append([os.path.join(self.img_dir,cur_id,str(cur_frame)+'.png') for cur_frame in valid_frame[1:]])
				self.question.append(cur_que)
				self.sal_map.append(([cur for cur in valid_saliency_map_pos[1:]],[cur for cur in valid_saliency_map_neg[1:]]))
				self.fixation.append(([cur for cur in valid_fixation_map_pos[1:]],[cur for cur in valid_fixation_map_neg[1:]]))

	def convert_idx(self,sentence):
		idx = []
		for word in sentence:
			if word in self.word2idx:
				idx.append(self.word2idx[word])
			else:
				if (word+'s' in self.word2idx):
					idx.append(self.word2idx[word+'s'])
				elif word[-1]=='s' and word[:-1] in self.word2idx:
					idx.append(self.word2idx[word[:-1]])
				elif word in ['women','men']:
					word = 'woman' if word=='women' else 'man'
					idx.append(self.word2idx[word])
				else:
					idx.append(self.word2idx['UNK'])
		return idx

	# randomly samping k frames
	def get_random_seq(self,seq,sal,fix):
		if len(seq)>=self.historical_step+1:
			start_idx = np.random.randint(0,len(seq)-self.historical_step)
			selected_seq = seq[start_idx:start_idx+self.historical_step+1]
			selected_sal_pos = sal[0][start_idx:start_idx+self.historical_step+1]
			selected_fix_pos = fix[0][start_idx:start_idx+self.historical_step+1]
			selected_sal_neg = sal[1][start_idx:start_idx+self.historical_step+1]
			selected_fix_neg = fix[1][start_idx:start_idx+self.historical_step+1]
		else:
			# for large temporal step that exceeds the video length, duplicate the first (second) frame
			selected_seq = [seq[0]]*(self.historical_step+1-len(seq)) + seq
			selected_sal_pos = [sal[0][0]]*(self.historical_step+1-len(sal[0])) + sal[0]
			selected_fix_pos = [fix[0][0]]*(self.historical_step+1-len(fix[0])) + fix[0]
			selected_sal_neg = [sal[1][0]]*(self.historical_step+1-len(sal[1])) + sal[1]
			selected_fix_neg = [fix[1][0]]*(self.historical_step+1-len(fix[1])) + fix[1]

		return selected_seq,selected_sal_pos,selected_fix_pos, selected_sal_neg,selected_fix_neg

	def __getitem__(self,index):
		img_seq = self.img[index]
		sal_seq = self.sal_map[index]
		fix_seq = self.fixation[index]
		if not self.test_mode:
			# for training, randomly sample the frames
			img_seq, sal_seq_pos, fix_seq_pos, sal_seq_neg, fix_seq_neg = self.get_random_seq(img_seq, sal_seq, fix_seq)
		else:
			# for testing, iterate all frames within a video
			sal_seq_pos, fix_seq_pos, sal_seq_neg, fix_seq_neg = sal_seq[0], fix_seq[0], sal_seq[1], fix_seq[1]

		cur_data = []
		for i in range(len(img_seq)):
			cur_img = Image.open(img_seq[i]).convert('RGB')
			if self.transform is not None:
				cur_img = self.transform(cur_img)
			cur_data.append(cur_img)
		cur_data = torch.cat([cur.unsqueeze(0) for cur in cur_data],dim=0)
		cur_que = np.array(self.question[index]).astype('int')
		if len(cur_que) < self.que_len:
			cur_que = np.concatenate((cur_que,np.zeros([self.que_len-len(cur_que),]).astype('int')))
		cur_que = torch.from_numpy(cur_que)

		cur_sal_pos = []
		cur_sal_neg = []
		cur_fix_pos = []
		cur_fix_neg = []

		for i in range(len(sal_seq_pos)):
			tmp_sal_pos = cv2.imread(sal_seq_pos[i])[:,:,0].astype('float32')
			tmp_sal_neg = cv2.imread(sal_seq_neg[i])[:,:,0].astype('float32')
			tmp_fix_pos = cv2.imread(fix_seq_pos[i])[:,:,0].astype('float32')
			tmp_fix_neg = cv2.imread(fix_seq_neg[i])[:,:,0].astype('float32')

			# downsampling for training
			if not self.test_mode:
				tmp_sal_pos = cv2.resize(tmp_sal_pos,(32,16)) # (32,16) for (512,256)
				tmp_sal_neg = cv2.resize(tmp_sal_neg,(32,16))
				tmp_fix_pos = cv2.resize(tmp_fix_pos,(32,16)) # (32,16) for (512,256)
				tmp_fix_neg = cv2.resize(tmp_fix_neg,(32,16))

			# data normalization
			tmp_sal_pos /= tmp_sal_pos.max()
			tmp_sal_neg /= tmp_sal_neg.max()
			tmp_fix_pos[tmp_fix_pos>0] = 1
			tmp_fix_neg[tmp_fix_neg>0] = 1

			cur_sal_pos.append(tmp_sal_pos)
			cur_sal_neg.append(tmp_sal_neg)
			cur_fix_pos.append(tmp_fix_pos)
			cur_fix_neg.append(tmp_fix_neg)

		cur_sal_pos = torch.from_numpy(np.array(cur_sal_pos))
		cur_sal_neg = torch.from_numpy(np.array(cur_sal_neg))
		cur_fix_pos = torch.from_numpy(np.array(cur_fix_pos))
		cur_fix_neg = torch.from_numpy(np.array(cur_fix_neg))

		return cur_data, cur_que, cur_sal_pos, cur_fix_pos, cur_sal_neg, cur_fix_neg

	def __len__(self,):
		return len(self.img)


class aggregated_loader(data.Dataset):
	def __init__(self,img_dir,sal_dir,que_file,word2idx,split,split_info,que_len=15,temporal_step=5,historical_step=9,transform=None,test_mode=False):
		self.img_dir = img_dir
		self.sal_dir = sal_dir
		self.que_file = json.load(open(que_file))
		self.word2idx = json.load(open(word2idx))
		self.split = split
		self.split_info = json.load(open(split_info))
		self.que_len = que_len
		self.temporal_step = temporal_step
		self.historical_step = historical_step
		self.transform = transform
		self.test_mode = test_mode
		self.init_img_dir()
		self.init_data()

	def init_img_dir(self,):
		batches = glob(os.path.join(self.img_dir,'*'))
		self.qid2dir = dict()
		for cur_batch in batches:
			ids = glob(os.path.join(cur_batch,'*'))
			for cur_id in ids:
				id_name = os.path.basename(cur_id)
				self.qid2dir[id_name] = cur_id

	def init_data(self,):
		self.img = []
		self.question = []
		self.sal_map = []
		self.fixation = []
		self.shuf_map = np.zeros([128,256])

		for cur_anno in self.que_file:
			cur_id = cur_anno['qid']
			if self.split_info[cur_id]['split'] != self.split:
				continue
			saliency_map = glob(os.path.join(self.sal_dir,cur_id,'saliency_map','all','*'))
			fixation = glob(os.path.join(self.sal_dir,cur_id,'fixation','all','*'))
			frame_idx = [os.path.basename(cur)[:-4] for cur in saliency_map]

			valid_frame = frame_idx[::self.temporal_step] # selecting valid frames with pre-defined step
			valid_saliency_map = saliency_map[::self.temporal_step]
			valid_fixation_map = fixation[::self.temporal_step]

			cur_que = cur_anno['question']
			cur_que = self.convert_idx(cur_que.split(' '))

			if self.test_mode:
				for i in range(self.historical_step+2,len(valid_frame),self.historical_step+1):
					self.img.append([os.path.join(self.img_dir,cur_id,str(cur_frame)+'.png') for cur_frame in valid_frame[i-self.historical_step-1:i]])
					self.sal_map.append([cur for cur in valid_saliency_map[i-self.historical_step-1:i]])
					self.fixation.append([cur for cur in valid_fixation_map[i-self.historical_step-1:i]])
					self.question.append(cur_que)
				if i < len(valid_frame)-1:
					self.img.append([os.path.join(cur_id,str(cur_frame)+'.png') for cur_frame in valid_frame[len(valid_frame)-self.historical_step-1:]])
					self.sal_map.append([cur for cur in valid_saliency_map[len(valid_frame)-self.historical_step-1:]])
					self.fixation.append([cur for cur in valid_fixation_map[len(valid_frame)-self.historical_step-1:]])
					self.question.append(cur_que)

					# aggrate the fixations for sAUC evaluation
					for i in range(len(valid_fixation_map[1:])):
							tmp_fix = cv2.imread(valid_fixation_map[i+1])[:,:,0].astype('float32')
							tmp_fix[tmp_fix>0] = 1
							self.shuf_map += tmp_fix
			else:
				self.img.append([os.path.join(self.img_dir,cur_id,str(cur_frame)+'.png') for cur_frame in valid_frame[1:]])
				self.sal_map.append([cur for cur in valid_saliency_map[1:]])
				self.fixation.append([cur for cur in valid_fixation_map[1:]])
				self.question.append(cur_que)

	def convert_idx(self,sentence):
		idx = []
		for word in sentence:
			if word in self.word2idx:
				idx.append(self.word2idx[word])
			else:
				if (word+'s' in self.word2idx):
					idx.append(self.word2idx[word+'s'])
				elif word[-1]=='s' and word[:-1] in self.word2idx:
					idx.append(self.word2idx[word[:-1]])
				elif word in ['women','men']:
					word = 'woman' if word=='women' else 'man'
					idx.append(self.word2idx[word])
				else:
					idx.append(self.word2idx['UNK'])
		return idx

	def get_random_seq(self,seq,sal,fix):
		if len(seq)>=self.historical_step+1:
			start_idx = np.random.randint(0,len(seq)-self.historical_step)
			selected_seq = seq[start_idx:start_idx+self.historical_step+1]
			selected_sal = sal[start_idx:start_idx+self.historical_step+1]
			selected_fix = fix[start_idx:start_idx+self.historical_step+1]
		else:
			selected_seq = [seq[0]]*(self.historical_step+1-len(seq)) + seq
			selected_sal = [sal[0]]*(self.historical_step+1-len(sal)) + sal
			selected_fix = [fix[0]]*(self.historical_step+1-len(fix)) + fix

		return selected_seq,selected_sal,selected_fix

	def __getitem__(self,index):
		img_seq = self.img[index]
		sal_seq = self.sal_map[index]
		fix_seq = self.fixation[index]
		if not self.test_mode:
			img_seq, sal_seq, fix_seq = self.get_random_seq(img_seq, sal_seq, fix_seq)

		cur_data = []
		for i in range(len(img_seq)):
			cur_img = Image.open(img_seq[i]).convert('RGB')
			if self.transform is not None:
				cur_img = self.transform(cur_img)
			cur_data.append(cur_img)
		cur_data = torch.cat([cur.unsqueeze(0) for cur in cur_data],dim=0)
		cur_que = np.array(self.question[index]).astype('int')
		if len(cur_que) < self.que_len:
			cur_que = np.concatenate((cur_que,np.zeros([self.que_len-len(cur_que),]).astype('int')))
		cur_que = torch.from_numpy(cur_que)
		cur_sal = []
		cur_fix = []

		for i in range(len(sal_seq)):
			tmp_sal = cv2.imread(sal_seq[i])[:,:,0].astype('float32')
			tmp_fix = cv2.imread(fix_seq[i])[:,:,0].astype('float32')

			# downsampling for training
			if not self.test_mode:
				tmp_sal = cv2.resize(tmp_sal,(32,16)) # (32,16) for (512,256)
				tmp_fix = cv2.resize(tmp_fix,(32,16))

			# data normalization
			tmp_sal /= tmp_sal.max()
			tmp_fix[tmp_fix>0] = 1

			cur_sal.append(tmp_sal)
			cur_fix.append(tmp_fix)

		cur_sal = torch.from_numpy(np.array(cur_sal))
		cur_fix = torch.from_numpy(np.array(cur_fix))

		return cur_data, cur_que, cur_sal, cur_fix

	def __len__(self,):
		return len(self.img)
