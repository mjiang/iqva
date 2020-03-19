from glob import glob
import os
import argparse
import json
import operator
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Processing the question info files')
parser.add_argument('--que_dir', type=str, default='./data_info', help='directory to the question')
parser.add_argument('--save_dir', type=str, default='./data_info', help='directory for saving the data')
parser.add_argument('--num_word', type=int, default=250, help='Number of words in the dictionary')
args = parser.parse_args()

word_count = dict()
q_len = []

# using consolidated data
que_info = json.load(open(os.path.join(args.que_dir,'question_info_latest.json')))
for i in range(len(que_info)):
	cur_que = que_info[i]['question']
	cur_que = cur_que.split(' ')
	cur_que = [cur_word for cur_word in cur_que if cur_word not in ['',' ']]
	q_len.append(len(cur_que))
	for word in cur_que:
		if word not in word_count:
			word_count[word] = 1
		else:
			word_count[word] += 1

word_count = sorted(word_count.items(),key=operator.itemgetter(1))
word_count.reverse()

word2idx = dict()
count = []
for idx in range(len(word_count)):
	if idx+2>args.num_word: # reserving one key for UNK
		break
	word2idx[word_count[idx][0]] = idx + 1
	count.append(word_count[idx][1])
word2idx['UNK'] = args.num_word
print('selected %d out of %d words' %(args.num_word,len(word_count)))
print('Average frequency is %f' %np.mean(count))
print('Least frequency is %d' %count[-1])
print('Average question length is %f' %np.mean(q_len))

with open(os.path.join(args.save_dir,'word2idx.json'),'w') as f:
	json.dump(word2idx,f)

