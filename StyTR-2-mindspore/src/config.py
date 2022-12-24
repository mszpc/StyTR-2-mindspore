from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from easydict import EasyDict as edict

config = edict()
config.modelname = 'Sty-trans2'
config.save_dir = 'save_models'
config.IS_MODELART = False

config.MODELARTS = edict()
config.MODELARTS.data_path = '/cache/dataset'
config.MODELARTS.CACHE_INPUT = '/cache/dataset'
config.MODELARTS.CACHE_OUTPUT = '/cache/output'

config.DATASET = edict()
config.DATASET.data_path = 'dataset'
config.DATASET.content_train = '/COCO/train'
config.DATASET.content_val = '/COCO/test'
config.DATASET.style_train = '/wikiart/train'
config.DATASET.style_val = '/wikiart/test'


config.TRAIN = edict()
config.TRAIN.save_freq = 1000
config.TRAIN.with_eval = True
config.TRAIN.batch_size = 4

# 学习率相关参数
config.TRAIN.lr = 0.0001 # 0.001
config.TRAIN.warmup_epochs = 1
config.TRAIN.T_max = 5e-4
config.TRAIN.eta_min = 5e-6

# Adam优化器使用weight decay
config.TRAIN.WD = 0.0000 # 0.00005
config.TRAIN.END_EPOCH = 20

# 损失函数权重
config.loss = edict()
config.loss.content_weight = 7.0
config.loss.style_weight = 10.0
