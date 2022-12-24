import os
import cv2
import numpy as np
import mindspore.numpy as msnp
from scipy.stats import entropy
import mindspore as ms
from scipy import linalg
from mindspore import dataset, dtype
from mindspore.dataset.transforms import c_transforms as c_tr
import mindspore.dataset.vision.c_transforms as tr
import mindspore.ops as F

from inception import InceptionV3


def get_pred(net, x):
    x = net(x)
    return F.Softmax(axis=1)(x)


class ImageDataset:
    def __init__(self, ds_path):
        self.ds_path = ds_path
        self.img_path_list = os.listdir(self.ds_path)

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        return np.array(cv2.imread(self.ds_path + self.img_path_list[idx]))[:, :512, :],np.array(cv2.imread(self.ds_path + self.img_path_list[idx]))[:, 512:1024, :],np.array(cv2.imread(self.ds_path + self.img_path_list[idx]))[:, 1024:, :]


def get_image_dataset(ds_path, batch_size=20):
    mean_inception = [0.485, 0.456, 0.406]
    std_inception = [0.229, 0.224, 0.225]
    ds = dataset.GeneratorDataset(ImageDataset(ds_path), ['image1','image2','image3'], shuffle=False)
    transforms_test = [
        c_tr.TypeCast(dtype.float32),
        tr.Resize([299, 299]),
        tr.Normalize(mean_inception, std_inception),
        tr.HWC2CHW()
    ]
    ds = ds.map(transforms_test, ['image1'])
    ds = ds.map(transforms_test, ['image2'])
    ds = ds.map(transforms_test, ['image3'])
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds

def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu - mu2) ** 2) + np.trace(cov + cov2 - 2 * cc)
    return np.real(dist)

def get_fid_score(ds_path='../output/'):
    # context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU", device_id=0)
    net = InceptionV3()
    param_dict = ms.load_checkpoint(
        '../calculate_IS/inceptionv3.ckpt')
    ms.load_param_into_net(net, param_dict)

    batch_size = 20
    ds = get_image_dataset(ds_path, batch_size)
    total_len = ds.get_dataset_size() * ds.get_batch_size()
    out1 = msnp.zeros((total_len, 2048), dtype=dtype.float32)
    out2 = msnp.zeros((total_len, 2048), dtype=dtype.float32)
    out3 = msnp.zeros((total_len, 2048), dtype=dtype.float32)

    for idx, img in enumerate(ds):
        out1[idx * batch_size:(idx + 1) * batch_size],out2[idx * batch_size:(idx + 1) * batch_size],out3[idx * batch_size:(idx + 1) * batch_size] = get_pred(net, img[0]),get_pred(net, img[1]),get_pred(net, img[2])
        print('idx =', idx)
    avg1 = msnp.mean(out1, 0).asnumpy()
    print(np.shape(avg1))
    avg2 = msnp.mean(out2, 0).asnumpy()
    print(np.shape(avg2))
    avg3 = msnp.mean(out3, 0).asnumpy()
    print(np.shape(avg3))
    cov1 = msnp.cov(out1, rowvar=False).asnumpy()
    print(np.shape(cov1))
    cov2 = msnp.cov(out2, rowvar=False).asnumpy()
    print(np.shape(cov2))
    cov3 = msnp.cov(out3, rowvar=False).asnumpy()
    print(np.shape(cov3))
    fid1 = frechet_distance(avg1, cov1, avg3, cov3)
    fid2 = frechet_distance(avg2, cov2, avg3, cov3)
    return fid1,fid2


if __name__ == "__main__":
    fid1,fid2 = get_fid_score('../output/')
    print("FID1 score: %.3f" % fid1)
    print("FID2 score: %.3f" % fid2)
