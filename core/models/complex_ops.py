import torch
import torch.nn as nn1
import torchcomplex.nn as nn
import torchcomplex
import numpy as np
#from torch import linalg as LA
#import torch.autograd as autograd
import math
#from .downsampler import Downsampler
#from torch.nn import Parameter
#import torch.nn.utils.spectral_norm as sp_norm
#from deepsplines.ds_modules import dsnn
#import ptwt
#import pywt

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)

torchcomplex.nn.Module.add = add_module

class Concat(nn1.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            #print("sss", idx)
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
                np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])
        print(f"Concat: {inputs_}")
        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)

def act(act_fun='LeakyReLU', chns=128):
    '''
        Either string defining an activation function or module (e.g. nn.ReLU)
    '''
    if isinstance(act_fun, str):
        if act_fun == 'ReLU':
            return nn.CReLU(inplace=True)
        elif act_fun == 'none':
            return nn1.Sequential()
        else:
            assert False
    else:
        return act_fun()
        
        
def bn(num_features):
#    return MeanOnlyBatchNorm(num_features) #nn.BatchNorm2d(num_features)
    return nn.BatchNorm2d(num_features)
    
    
def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', c=0, norm_method='spectral', downsample_mode='stride', is_last=False):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    #if pad == 'reflection':
    #    padder = nn.ReflectionPad2d(to_pad)
    #    to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad,  bias=bias)
#    print(f"convolver: {convolver.weight}")
    if is_last:
       layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
       return nn.Sequential(*layers)

    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)
