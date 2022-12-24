from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time

import numpy as np
import mindspore as ms
from mindspore import Tensor, Model, save_checkpoint, context
from mindspore.communication.management import get_rank, init, get_group_size
from mindspore.context import ParallelMode
import mindspore.nn as nn
from mindspore.train.callback import TimeMonitor, LossMonitor, ModelCheckpoint, CheckpointConfig

from src.config import config
from src.loss import StyTRWithLoss
from models import StyTR
from models import transformer
from src.dataset import Create_Content_Style_Dateset
from src.utils.lr_scheduler import warmup_cosine_annealing_lr

parser = argparse.ArgumentParser(description='Train StyTR2 network')

parser.add_argument('--opts', help="Modify config options using the command-line",
                    default=None, nargs=argparse.REMAINDER)
parser.add_argument('--dataDir', help='data directory', type=str, default='')
parser.add_argument('--train_url', required=False, default=None, help='Location of training outputs.')
parser.add_argument('--data_url', required=False, default=config.MODELARTS.CACHE_INPUT, help='Location of data.')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend'],
                    help='device where the code will be implemented')

args = parser.parse_args()


def train():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    # 初始化模型存放目录
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    content_style_train_ds = Create_Content_Style_Dateset(is_train=True)
    dataset_size = content_style_train_ds.get_dataset_size()

    vgg = StyTR.vgg
    vgg = vgg[:44]

    decoder = StyTR.Decoder(True)
    Trans = transformer.Transformer()
    embedding = StyTR.PatchEmbed()

    lr = warmup_cosine_annealing_lr(config.TRAIN.lr, dataset_size,
                                    config.TRAIN.warmup_epochs, config.TRAIN.END_EPOCH,
                                    config.TRAIN.T_max, config.TRAIN.eta_min)

    optimizer = nn.Adam([{'params': Trans.trainable_params()},
                         {'params': decoder.trainable_params()},
                         {'params': embedding.trainable_params()},
                         ], learning_rate=lr)

    stytran = StyTR.StyTrans(decoder, embedding, Trans)
    net_with_loss = StyTRWithLoss(vgg, stytran)
    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossMonitor()  # per_print_times=dataset_size
    callback_list = [time_cb, loss_cb]
    StyTR_model = Model(net_with_loss, optimizer=optimizer, amp_level="O0")

    print("************ Start training now ************")
    print('start training, epoch size = %d' % config.TRAIN.END_EPOCH)
    StyTR_model.train(config.TRAIN.END_EPOCH, content_style_train_ds, dataset_sink_mode=True, callbacks=callback_list)
    print("************ Training complete ************")
    # save sty-tran model
    save_checkpoint(decoder, config.save_dir + '/decoder.ckpt')
    save_checkpoint(Trans, config.save_dir + '/transformer.ckpt')
    save_checkpoint(embedding, config.save_dir + '/embedding.ckpt')
    # save vgg model
    save_checkpoint(vgg, config.save_dir + '/vgg.ckpt')

if __name__ == '__main__':
    train()
