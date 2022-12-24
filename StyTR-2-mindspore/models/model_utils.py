from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore import dtype as mstype

import numpy as np
# class AdaptiveAvgpool(Cell):
#     def __init__(self):
#         super(AdaptiveAvgpool, self).__init__()
#     def construct(self, input):
#         return AdaptiveAvgpool_function(input)
#
# def floor(x):
#     return int(x)
#
# def ceil(x):
#     out = int(x)
#     if out == x:
#         return out
#     else:
#         return out + 1
#
# def AdaptiveAvgpool_function(input,target_size=(18,18)):
#     h=target_size[0]
#     w=target_size[1]
#     s1=[]
#     e1=[]
#     s2=[]
#     e2=[]
#     shape=input.shape
#
#     # return ops.ReduceMean(keep_dims=True)(input,(-2,-1))
#
#     for i in range(w):
#         s1.append(floor(i*shape[-1]/w))
#         e1.append(ceil((i+1)*shape[-1]/w))
#     for i in range(h):
#         s2.append(floor(i*shape[-2]/h))
#         e2.append(ceil((i+1)*shape[-2]/h))
#     pool2=[]
#     for i_h in range(h):
#         pool=[]
#         for i_w in range(w):
#             pool.append(ops.ReduceMean(keep_dims=True)(input[:,:,s2[i_h]:e2[i_h],s1[i_w]:e1[i_w]],(-2,-1)))
#         pool1=pool[0]
#         for i_w in range(w-1):
#             pool1=ops.Concat(axis=-1)((pool1,pool[i_w+1]))
#         pool2.append(pool1)
#     out=pool2[0]
#     for i_h in range(h-1):
#         out=ops.Concat(axis=-2)((out,pool2[i_h+1]))
#     return out
#

class AdaptiveAvgPool2d(Cell):

    def __init__(self, input_shape, output_size):
        """Initialize AdaptiveAvgPool2d."""
        super(AdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size
        self.shape = ops.Shape()
        self.input_shape = input_shape
        self.H = self.output_size[0]
        self.W = self.output_size[1]

        self.H_start = np.arange(start=0, stop=self.H, dtype=np.float32) * (self.input_shape[-2] / self.H)
        self.H_end = np.ceil(((np.arange(start=0, stop=self.H, dtype=np.float32)+1) * (self.input_shape[-2] / self.H)))

        self.W_start = np.arange(start=0, stop=self.W, dtype=np.float32) * (self.input_shape[-1] / self.W)
        self.W_end = np.ceil(((np.arange(start=0, stop=self.W, dtype=np.float32)+1) * (self.input_shape[-1] / self.W)))

        self.H_start=list(self.H_start)
        self.H_start = [int(i) for i in self.H_start]
        self.H_end=list(self.H_end)
        self.H_end = [int(i) for i in self.H_end]
        self.W_start=list(self.W_start)
        self.W_start = [int(i) for i in self.W_start]
        self.W_end=list(self.W_end)
        self.W_end = [int(i) for i in self.W_end]


    def construct(self, inputs):
        pooled2 = []
        for idx_H in range(self.H):
            pooled1 = []
            for idx_W in range(self.W):
                h_s = self.H_start[idx_H]
                h_e = self.H_end[idx_H]
                w_s = self.W_start[idx_W]
                w_e = self.W_end[idx_W]
                res = inputs[:, :, h_s:h_e, w_s:w_e]
                pooled1.append(ops.ReduceMean(keep_dims=True)(res, (-2, -1)))
            pooled1 = ops.Concat(-1)(pooled1)
            pooled2.append(pooled1)
        pooled2 = ops.Concat(-2)(pooled2)
        return pooled2

if __name__ == '__main__':

    from mindspore import context
    context.set_context(mode=1)
    # import numpy as np
    from mindspore import Tensor
    # [4,512,32,32]
    input_np = np.zeros(shape=(4,512,32,32), dtype=np.float32)
    input = Tensor.from_numpy(input_np)
    # layer = AdaptiveAvgPool2d(input_shape=(4,512,32,32),output_size=(18,18))
    from mindspore import nn

    pad = nn.Pad(paddings=((0,0),(0,0),(3, 3), (3, 3)), mode="REFLECT")
    input = pad(input)
    layer = nn.AvgPool2d(kernel_size=(3, 3), stride=2,)
    output = layer(input)
    print(output.shape)