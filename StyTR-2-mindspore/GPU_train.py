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
from src.dataset import Create_ContentDateset, Create_StyleDateset
from src.utils.lr_scheduler import warmup_cosine_annealing_lr


def train():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    # 初始化模型存放目录
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    content_train_ds = Create_ContentDateset(is_train=True)
    style_train_ds = Create_StyleDateset(is_train=True)
    dataset_size = min(content_train_ds.get_dataset_size(), style_train_ds.get_dataset_size())

    content_dataloader = content_train_ds.create_dict_iterator()
    style_dataloader = style_train_ds.create_dict_iterator()

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

    manager = nn.DynamicLossScaleUpdateCell(loss_scale_value=2 ** 12, scale_factor=2, scale_window=1000)
    StyTR_model = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer=optimizer, scale_sense=manager)

    print("************ Start training now ************")
    print('start training, epoch size = %d' % config.TRAIN.END_EPOCH)
    step = 0
    for epoch in range(config.TRAIN.END_EPOCH):
        for i, (dcontent, dstyle) in enumerate(zip(content_dataloader, style_dataloader)):
            step_begin_time = time.time()
            content = dcontent['content']
            style = dstyle['style']
            loss = StyTR_model(content, style)
            step_end_time = time.time()
            print('step:', step, 'epoch:', epoch, 'batch:', i, 'loss:', loss[0].sum())
            print('step time is', step_end_time - step_begin_time)
            # if (step + 1) % config.TRAIN.save_freq == 0:
            #     save_checkpoint(decoder, config.save_dir + '/decoder_' + str(step + 1) + '.ckpt')
            #     save_checkpoint(Trans, config.save_dir + '/transformer_' + str(step + 1) + '.ckpt')
            #     save_checkpoint(embedding, config.save_dir + '/embedding_' + str(step + 1) + '.ckpt')
            step += 1

    # save sty-tran model
    save_checkpoint(decoder, config.save_dir + '/decoder.ckpt')
    save_checkpoint(Trans, config.save_dir + '/transformer.ckpt')
    save_checkpoint(embedding, config.save_dir + '/embedding.ckpt')
    # save vgg model
    save_checkpoint(vgg, config.save_dir + '/vgg.ckpt')

    print("************ Training complete ************")


if __name__ == '__main__':
    train()
