# AiR: Immersive Question-directed Visual Attention
This is the official repository for Immersive Question-directed Visual Attention (IQVA) datasets. It provides the first visual attention dataset that takes into account the correctness of attention, and a framework to simultaneously predict both the correct and incorrect attentions. An example illustrating the correctness of attention in the Immersive Question Answering context is shown below:

<!-- ![teaser](data/teaser.jpg?raw=true) -->

### Reference
If you use our code or data, please cite our paper:
```
@InProceedings{IQVA,
author = {Jiang, Ming and Chen, Shi and Yang, Jinhui and Zhao, Qi.},
title = {Fantastic Answers and Where to Find Them: Immersive Question-Directed Visual Attention},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2020}
}
```

### Disclaimer
We adopt the implementation of Skip-Thought from [this repository](https://github.com/Cadene/skip-thoughts.torch/tree/master/pytorch). Please refer to these links for further README information.

### Requirements
1. Requirements for Pytorch. We use Pytorch 1.2.0 in our experiments.
2. Requirements for Tensorflow. We only use the tensorboard for visualization.
3. Python 3.6+

### IQVA Dataset
Our data is available at https://drive.google.com/file/d/1gJQdvBmqQIXOVZcXdvVlx1_0ufRaUsyS/view?usp=sharing. For each trial (i.e., a single question), the saliency maps and raw fixation maps are stored in different folder. We provide both the maps aggregated across all participants and those for different groups if applicable (i.e., participants with correct and incorrect answers). Information about the questions and data splits are stored in `question_info.json` and `split_info.json`, respectively. In our experiments for correctness-aware attention prediction, we only use the questions with human accuracy between 20-80% , and they are highlighted with `valid_correctness=1` in `split_info.json`. For the experiments for attention prediction regardless of correctness, we use all data.

Note that we do not provide the raw Youtube videos, but instead include their video id in `question_info.json`. The saliency maps and fixation maps are named based on the frame ids of the corresponding videos, thus it should be straightforward to retrieve the video inputs accordingly.   

### Data Pre-processing
1. Download our IQVA dataset, and unzip it to the root directory of this project.
2. Download the corresponding videos, and retrieve the video frames used in our dataset (stored as JPG images in `$IMG_DIR`).
3. Pre-process the questions to obtain a word dictionary:
  ```
  python process_question --que_dir $Question_FILE
  ```

### Correctness-aware Attention Prediction
For training our model for simultaneously predicting visual attentions for correct and incorrect answers:
```
python main_corr.py --mode train --img_dir $IMG_DIR --sal_dir ./data --que_file ./data_info/question_info.json --word2idx ./data_info/word2idx.json --checkpoint $CHECKPOINT_DIR --split_info ./data_info/split_info.json
```

To evaluate the performance on the test set, simply set `--mode eval`.

### Attention Prediction Regardless of Correctness
The model is initialized with the weights previously trained for correctness-aware attention prediction, please follow the instruction above to fully train the model. After that, copy the weights to a new checkpoint directory:
```
cp $CHECKPOINT_SOURCE/model_best.pth $CHECKPOINT_TARGET/pretrained.pth
```

Then the training process can be called:
```
python main_agg.py --mode train --img_dir $IMG_DIR --sal_dir ./data --que_file ./data_info/question_info.json --word2idx ./data_info/word2idx.json --checkpoint $CHECKPOINT_DIR --split_info ./data_info/split_info.json
```

The evaluation process is the same as correctness-aware attention prediction.
