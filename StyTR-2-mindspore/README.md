# 目录

<!-- TOC -->

- [目录](#目录)
- [StyTr2说明](#StyTr2说明)
- [模型架构](#模型架构)
- [任务介绍](#任务介绍)
- [结果展示](#结果展示)
- [代码结构](#代码结构)
- [数据集](#数据集)
    - [Content——COCO](#Content——COCO)
    - [Styles——Wikiart](#Styles——Wikiart)
- [超参数设置](#超参数设置)
- [训练和评估](#训练和评估)
    - [Train（训练）](#Train（训练）)
    - [Evaluate（评估）](#Evaluate（评估）)
<!-- /TOC -->


# StyTr2说明

由于典型的基于 CNN 的风格化方法获取的内容表示是不准确的，会容易导致内容泄漏的问题。因此为了解决这个问题，StyTr2通过提出一种基于Transformer的方法，将输入图像的long-range依赖性考虑到无偏风格转移。

[论文 paper](https://arxiv.org/abs/2105.14576)：Yingying Deng, Fan Tang, Weiming Dong, Chongyang Ma, Xingjia Pan, Lei Wang, Changsheng Xu

# 模型架构

![net](./README_img/net.png)

模型主要包括三部分：内容 Transformer 编码器、风格 Transformer 编码器 以及 Transformer 解码器。  
其中，内容 Transformer 编码器和风格 Transformer 编码器分别用来编码content内容域和style风格域的图片的长程信息（embedding之后的信息），这种编码方式可以有效**避免细节丢失问题**。将content和style图片分别进行编码之后，输入StyTr网络后，最终再经过Transformer 解码器用来**将内容特征转换为带有风格图片特征的风格化结果**。

# 任务介绍

在本次作业任务中，我们成功复现了CVPR 2022 中该篇论文的 : Image Style Transfer with Transformers模型，参考pytorch版本的官方代码复现出基于华为AI计算框架 mindspore 版本的代码，完成了在mindspore框架下 网络模型的构建、训练以及最终结果的测试，完成了图像的多样风格迁移任务，成功复现了StyTr2论文。

通过输入内容图content以及风格图style，经过训练的模型之后，得到以style图风格的content图片。

# 结果展示

![result](./README_img/result.png)

# 代码结构

```
├─StyTr2(mindspore)
   ├─calculate_FID                  #计算FID指标相关
   │    ├─inception.py             #inception模型定义
   │    └─FID.py                   #计算FID的运行文件
   ├─calculate_IS                    #计算IS指标相关
   │    ├─inception.py             #inception模型定义
   │    ├─inceptionv3.ckpt          #inception模型权重文件
   │    └─IS.py                   #计算IS的运行文件
   ├─input
   │    ├─content                 #content内容图输入
   │    └─style                   #style风格图输入
   ├─models
   │    ├─model_utils.py          # 搭建model所用到的工具
   │    ├─StyTR.py                # StyRT网络模型定义
   │    ├─transformer.py          # transformer模型定义
   │    └─ViT_helper.py           # transformer模型定义辅助工具
   ├─output 
   ├─save_models                  # 保存的模型权重文件
   │    ├─decoder.ckpt             
   │    ├─embedding.ckpt           
   │    └─transformer.ckpt        
   └─src
   │    ├─config.py               # config参数设置 
   │    ├─dataset.py              # 数据集dataset准备
   │    ├─loss.py                 # 损失函数 loss function 定义
   │    └─utils     
   │        ├─function.py         # 训练所用到的工具函数
   │        └─lr_scheduler.py     # 学习率更新函数
   ├─debug.py                     # debug调试脚本
   ├─evaluate.py                  # 最终测试脚本
   ├─GPU_train.py                 # GPU训练环境的训练启动代码
   ├─modelarts_train.py           # 使用华为云Modelarts的Ascend时的训练启动代码
   ├─moxing_adapter.py            # 华为云Modelarts创建任务的辅助文件
   ├─NPU_train.py                 # NPU环境的训练启动代码
   ├─train.py                     # 本机训练启动代码
   └─requirements.txt             # requirements
```



# 数据集

```
├─dataset
│    ├─COCO       
│    │   ├─README.txt
│    │   └─train
│    └─wikiart
│        ├─README.txt
└─       └─train
```

## Content——COCO

  内容图片我们选用的是 **COCO2014** 数据集，该数据集图像包括91类目标，328,000影像和2,500,000个label。目前为止有语义分割的最大数据集，提供的类别有80 类，有超过33 万张图片，其中20 万张有标注，整个数据集中个体的数目超过150 万个。

  COCO文件夹下包含`/train`文件夹，包含若干图片作为content用于训练。

## Styles——Wikiart

  风格图片我们选择的是 **WikiArt** 数据集，该数据集是一个大型绘画作品数据集，主要用于数字化艺术品归档与检索系统的开发。该数据集收录了 195 位艺术家的绘画作品，其中 42,129 幅用于训练，10,628 幅用于测试。

  wikiart文件夹下包含`/train`文件夹，包含若干图片作为style用于训练。



# 超参数设置
（详见config.py文件）

```
# 基本参数设置
config = edict()
config.modelname = 'Sty-trans2'        #模型名称
config.save_dir = 'save_models'        #本机训练模型保存路径
config.IS_MODELART = False             #是否为华为云Modelarts训练

#华为云Modelarts训练参数设置
config.MODELARTS = edict()
config.MODELARTS.data_path = '/cache/dataset'    #平台dataset路径
config.MODELARTS.CACHE_INPUT = '/cache/dataset'  #平台input输入路径
config.MODELARTS.CACHE_OUTPUT = '/cache/output'  #平台output输出路径

#dataset数据集加载参数设置
config.DATASET = edict()
config.DATASET.data_path = 'dataset'
config.DATASET.content_train = '/COCO/train'     #content 训练数据加载路径
config.DATASET.content_val = '/COCO/test'        #content 测试数据加载路径
config.DATASET.style_train = '/wikiart/train'    #style 训练数据加载路径
config.DATASET.style_val = '/wikiart/test'       #style 测试数据加载路径

#训练参数设置
config.TRAIN = edict()
config.TRAIN.save_freq = 1000              # 模型保存频率
config.TRAIN.with_eval = True              
config.TRAIN.batch_size = 4                # 批大小
config.TRAIN.lr = 0.0001                   # 初始学习率
config.TRAIN.warmup_epochs = 1             # 学习率上升到最大值时的epoch数
config.TRAIN.T_max = 5e-4                  # 学习率上升到的最大值
config.TRAIN.eta_min = 5e-6                # 学习率下降后的最小值
config.TRAIN.WD = 0.0000 # 0.00005         # 优化器权重衰减值
config.TRAIN.END_EPOCH = 20                # 训练epoch数

# 损失函数权重设置
config.loss = edict()
config.loss.content_weight = 7.0           # 计算content损失的权重
config.loss.style_weight = 10.0            # 计算style损失的权重
```



# 训练和评估


## Train（训练）

三种训练平台的训练运行脚本 `GPU_train.py`\ `NPU_train.py`\ `modelart_train.py`

### 1. 初始化模型存放目录、设置模式并配置相关训练信息 

——**优化器**：`Adam` 

——**损失函数**：`\src\loss.py`

（1）内容感知损失函数（2）风格感知损失函数（3）身份感知损失函数

——**学习率函数**：` \src\utils\lr_scheduler` 

预热余弦退火学习率策略

###  2. 训练模型

```
——GPU：GPU_train.py
      For循环迭代训练内容和风格数据集中较小的数据集，两者一一对应、绑定传入模型进行训练
——NPU：NPU_train.py
      调用StyTR_model.train进行模型训练
——ModelArt：modelart_train.py
      在华为云上用Acend卡跑，调用端口进行模型训练
```
### 3. 保存sty-tran模型和VGG模型

## Evaluate（评估）

###  图片存放位置

```
——输入：input文件夹
——输出：output文件夹
```

### 测试主函数

`evaluate.py`

### 预训练模型

`vgg-model, vit_embedding, decoder, Transformer_module`

