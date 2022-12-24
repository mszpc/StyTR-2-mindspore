import os
import cv2
import numpy as np
import mindspore.numpy as msnp
from scipy.stats import entropy
import mindspore as ms
from mindspore import dataset, dtype
from mindspore.dataset.transforms import c_transforms as c_tr
import mindspore.dataset.vision.c_transforms as tr
import mindspore.ops.function as F

from calculate_IS.inception import InceptionV3


def get_pred(net, x):
    x = net(x)
    return F.softmax(x, axis=1)


class ImageDataset:
    def __init__(self, ds_path):
        self.ds_path = ds_path
        self.img_path_list = os.listdir(self.ds_path)

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        return np.array(cv2.imread(self.ds_path + self.img_path_list[idx]))[:, 1024:, :]


def get_image_dataset(ds_path, batch_size=20):
    mean_inception = [0.485, 0.456, 0.406]
    std_inception = [0.229, 0.224, 0.225]
    ds = dataset.GeneratorDataset(ImageDataset(ds_path), ['image'], shuffle=False, num_parallel_workers=4)
    transforms_test = [
        c_tr.TypeCast(dtype.float32),
        tr.Resize([299, 299]),
        tr.Normalize(mean_inception, std_inception),
        tr.HWC2CHW()
    ]
    ds = ds.map(transforms_test, 'image')
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds


def get_is_score(ds_path='./output/'):
    # context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU", device_id=0)
    net = InceptionV3()
    param_dict = ms.load_checkpoint(
        'inceptionv3.ckpt')
    ms.load_param_into_net(net, param_dict)

    batch_size = 20
    ds = get_image_dataset(ds_path, batch_size)
    total_len = ds.get_dataset_size() * ds.get_batch_size()
    out = msnp.zeros((total_len, 2048), dtype=dtype.float32)

    for idx, img in enumerate(ds):
        out[idx * batch_size:(idx + 1) * batch_size] = get_pred(net, img[0])
        print('idx =', idx)
    all_avg = F.mean(out, 0).asnumpy()
    kl_every = np.zeros(total_len, dtype='float32')
    for i in range(total_len):
        kl_every[i] = entropy(out[i].asnumpy(), all_avg)

    return float(np.exp(np.mean(kl_every)))


if __name__ == "__main__":
    is_score = get_is_score('../output/')
    print("IS score: %.3f" % is_score)
