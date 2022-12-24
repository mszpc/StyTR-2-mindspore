from pathlib import Path
import os
from PIL import Image

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from mindspore import dataset as ds
from mindspore.communication import init, get_rank, get_group_size
from mindspore.dataset.vision import c_transforms, py_transforms
from mindspore.dataset.transforms.c_transforms import Compose

from src.config import config


def train_transform():
    transform_list = [
        c_transforms.Resize(size=(512, 512)),
        c_transforms.RandomCrop(256),
        py_transforms.ToTensor()
    ]
    return Compose(transform_list)


def content_transform_function():
    transforms_list = [py_transforms.ToTensor()]
    transform = Compose(transforms_list)
    return transform


def style_transform_function(h, w):
    transform_list = [c_transforms.CenterCrop((h, w)), py_transforms.ToTensor()]
    transform = Compose(transform_list)
    return transform


class FolderDataGenerator():
    def __init__(self, root, transform):
        super(FolderDataGenerator, self).__init__()
        self.root = root
        self.path = os.listdir(self.root)
        if os.path.isdir(os.path.join(self.root, self.path[0])):
            self.paths = []
            for file_name in os.listdir(self.root):
                for file_name1 in os.listdir(os.path.join(self.root, file_name)):
                    self.paths.append(self.root + "/" + file_name + "/" + file_name1)
        else:
            self.paths = list(Path(self.root).glob('*'))
        self.transform = transform
        print(self.root, len(self.paths))

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)


def Create_ContentDateset(is_train):
    if config.IS_MODELART:
        root = config.MODELARTS.CACHE_INPUT
    else:
        root = config.DATASET.data_path

    if is_train:
        root = root + config.DATASET.content_train
    else:
        root = root + config.DATASET.content_val

    transform = train_transform()

    DataGenerator = FolderDataGenerator(root=root, transform=transform)

    print(DataGenerator.__len__())
    if is_train:
        dataset = ds.GeneratorDataset(DataGenerator, python_multiprocessing=True, shuffle=True,
                                      # shard_id=get_rank(), num_shards=get_group_size(),
                                      column_names=['content'], num_parallel_workers=4)
        dataset = dataset.batch(batch_size=config.TRAIN.batch_size, drop_remainder=False)
    else:
        dataset = ds.GeneratorDataset(DataGenerator, python_multiprocessing=True, shuffle=True,
                                      column_names=['content'], num_parallel_workers=4)
        dataset = dataset.batch(batch_size=config.TRAIN.batch_size, drop_remainder=False)

    return dataset


def Create_StyleDateset(is_train):
    if config.IS_MODELART:
        root = config.MODELARTS.CACHE_INPUT
    else:
        root = config.DATASET.data_path

    if is_train:
        root = root + config.DATASET.style_train
    else:
        root = root + config.DATASET.style_val

    transform = train_transform()

    DataGenerator = FolderDataGenerator(root=root, transform=transform)

    print(DataGenerator.__len__())
    if is_train:
        dataset = ds.GeneratorDataset(DataGenerator, python_multiprocessing=True, shuffle=True,
                                      # shard_id=get_rank(), num_shards=get_group_size(),
                                      column_names=['style'], num_parallel_workers=4)
        dataset = dataset.batch(batch_size=config.TRAIN.batch_size, drop_remainder=False)
    else:
        dataset = ds.GeneratorDataset(DataGenerator, python_multiprocessing=True, shuffle=True,
                                      column_names=['style'], num_parallel_workers=4)
        dataset = dataset.batch(batch_size=config.TRAIN.batch_size, drop_remainder=False)

    return dataset


class Content_Style_DataGenerator():
    def __init__(self, content_root, style_root, content_transforms, style_transforms):
        super(Content_Style_DataGenerator, self).__init__()
        self.content_root = content_root
        self.content_path = os.listdir(self.content_root)
        if os.path.isdir(os.path.join(self.content_root, self.content_path[0])):
            self.content_paths = []
            for file_name in os.listdir(self.content_root):
                for file_name1 in os.listdir(os.path.join(self.content_root, file_name)):
                    self.content_paths.append(self.content_root + "/" + file_name + "/" + file_name1)
        else:
            self.content_paths = list(Path(self.content_root).glob('*'))
        self.content_transforms = content_transforms
        print("self.content_paths: ", self.content_root, len(self.content_paths))

        self.style_root = style_root
        self.style_path = os.listdir(self.style_root)
        if os.path.isdir(os.path.join(self.style_root, self.style_path[0])):
            self.style_paths = []
            for file_name in os.listdir(self.style_root):
                for file_name1 in os.listdir(os.path.join(self.style_root, file_name)):
                    self.style_paths.append(self.style_root + "/" + file_name + "/" + file_name1)
        else:
            self.style_paths = list(Path(self.style_root).glob('*'))
        self.style_transforms = style_transforms
        print("self.style_paths: ", self.style_root, len(self.style_paths))

    def __getitem__(self, index):
        content_path = self.content_paths[index]
        content_img = Image.open(str(content_path)).convert('RGB')
        content_img = self.content_transforms(content_img)

        style_path = self.style_paths[index]
        style_img = Image.open(str(style_path)).convert('RGB')
        style_img = self.style_transforms(style_img)
        return content_img, style_img

    def __len__(self):
        min_len = min(len(self.content_paths), len(self.style_paths))
        return min_len


def Create_Content_Style_Dateset(is_train):
    if config.IS_MODELART:
        root = config.MODELARTS.CACHE_INPUT
    else:
        root = config.DATASET.data_path

    if is_train:
        style_root = root + config.DATASET.style_train
        content_root = root + config.DATASET.content_train
    else:
        style_root = root + config.DATASET.style_val
        content_root = root + config.DATASET.content_val

    content_transforms = train_transform()
    style_transforms = train_transform()

    DataGenerator = Content_Style_DataGenerator(content_root=content_root, style_root=style_root,
                                                content_transforms=content_transforms,
                                                style_transforms=style_transforms)

    if is_train:
        dataset = ds.GeneratorDataset(DataGenerator, python_multiprocessing=True, shuffle=True,
                                      # shard_id=get_rank(), num_shards=get_group_size(),
                                      column_names=['content', 'style'], num_parallel_workers=4)
        dataset = dataset.batch(batch_size=config.TRAIN.batch_size, drop_remainder=False)
    else:
        dataset = ds.GeneratorDataset(DataGenerator, python_multiprocessing=True, shuffle=True,
                                      column_names=['content', 'style'], num_parallel_workers=4)
        dataset = dataset.batch(batch_size=config.TRAIN.batch_size, drop_remainder=False)

    return dataset
