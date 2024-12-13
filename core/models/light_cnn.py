import sys

import torch
import torch.nn as nn
import numpy as np
from .common import *

sys.path.append('../')
from pruning.DAM import * 

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)
torch.nn.Module.add = add_module


class ConvDecoder(nn.Module):
    def __init__(self, args, num_layers, num_channels, num_output_channels, out_size, in_size, act_func='ReLU', upsample_mode='nearest', norm_func='bn', need_dropout=False,
                 need_sigmoid=False):
        super(ConvDecoder, self).__init__()

        ### parameter setup
        kernel_size = 3
        strides = [1] * (num_layers - 1)

        norm_layer = nn.BatchNorm2d if norm_func == 'bn' else nn.InstanceNorm2d

        ### compute up-sampling factor from one layer to another
        scale_x, scale_y = (out_size[0] / in_size[0]) ** (1. / (num_layers - 1)), (out_size[1] / in_size[1]) ** (
                    1. / (num_layers - 1))
        hidden_size = [(int(np.ceil(scale_x ** n * in_size[0])),
                        int(np.ceil(scale_y ** n * in_size[1]))) for n in range(1, (num_layers - 1))] + [out_size]
        print(hidden_size)
        ### hidden layers
        self.net = nn.Sequential()
        '''       
        print('=============================== DIP ==================================================')
#        print(f'num_scales:{len(num_channels_down)} num_channels:{num_channels_up[0]} out_size:{out_size} in_size:{out_size}\n'
        f'act_func:{act_fun} upsample_mode:{upsample_mode} need_dropout:{need_dropout} need_sigmoid:{need_sigmoid}')
        print('=============================== DIP ==================================================')
        '''
        # Rectangle window
        w1 = torch.tensor([0.5, 0.5])
        # Triangle window
        w2 = torch.tensor([0.25, 0.5, 0.25])
        # -20dB LPF 7 taps
        w3 = torch.tensor([-0.1070,    0.0000,    0.3389,    0.5360,    0.3389,    0.0000,   -0.1070])

        w41 = torch.tensor([ 0.0027,    0.0140,    0.0113,   -0.0217,   -0.0556,   -0.0218,    0.1123,    0.2801,    0.3572,    0.2801,    0.1123 ,  -0.0218,   -0.0556,   -0.0217,    0.0113])

        w14 = torch.tensor([-0.0054,-0.0518,0.2554,0.6036,0.2554,-0.0518, -0.0054])

        w4 = torch.tensor([0.002745,0.014011,0.011312,-0.021707,-0.055602,-0.021802,0.112306,0.280146,0.357183,0.280146,0.112306,-0.021802,-0.055602,-0.021707,0.011312,0.014011,0.002745])
        w5 = torch.tensor([-0.001921,-0.004291,0.009947,0.021970,-0.022680,-0.073998,0.034907,0.306100,0.459933,0.306100,0.034907,-0.073998,-0.022680,0.021970,0.009947,-0.004291,-0.001921]) # -60
        w6 = torch.tensor([0.000015,0.000541,0.003707,0.014130,0.037396,0.075367,0.121291,0.159962,0.175182,0.159962,0.121291,0.075367,0.037396,0.014130,0.003707,0.000541,0.000015]) # -100

        w_list = [w3, w3, w3]

        for i in range(num_layers - 1):
            if upsample_mode == 'bilinear' or upsample_mode == 'nearest':
                self.net.add(nn.Upsample(size=hidden_size[i], mode=upsample_mode))
            elif upsample_mode == 'transposed':
                if 'knee' in args.task:
                  if i >= num_layers - 5:
                     self.net.add(nn.ConvTranspose2d(num_channels, num_channels, kernel_size=2, stride=2))
                else:
                  self.net.add(nn.ConvTranspose2d(num_channels, num_channels, kernel_size=2, stride=2))
            elif upsample_mode == 'LPF6':          
                if 'knee' in args.task:
                  if i >= num_layers - 3:
                     self.net.add(InsertZeros(2, 2, gain=1.0))
                     self.net.add(lowpass_conv3(num_channels, w6, pad_mode = 'reflect', gain=4.0))
                elif 'brain' in args.task:
                  if i >= num_layers - 6:
                     self.net.add(InsertZeros(2, 2, gain=1.0))
                     self.net.add(lowpass_conv3(num_channels, w6, pad_mode = 'reflect', gain=4.0))
                else:
                  self.net.add(InsertZeros(2, 2, gain=1.0))
                  self.net.add(lowpass_conv3(num_channels, w6, pad_mode = 'reflect', gain=4.0))
            elif upsample_mode == 'None':
               pass
            else:
               raise NotImplementedError("No such upsampling implemented.")

            if i == 0:
               conv = nn.Conv2d(args.input_dim, num_channels, kernel_size, strides[i], padding=(kernel_size - 1) // 2, bias=True)
            else:
               conv = nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size - 1) // 2,
                             bias=True)
            self.net.add(conv)
#            self.net.add(learnable_zero_insertion_lowpass_conv(num_channels, pad_size=(kernel_size - 1) // 2, outsize=hidden_size[i][0]))
            self.net.add(act(act_func))
            self.net.add(norm_layer(num_channels)) #nn.BatchNorm2d(num_channels, affine=True))
            if need_dropout:
                self.net.add(nn.Dropout2d(0.3))
        ### final layer
        self.net.add(
            nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size - 1) // 2, bias=True))
       # self.net.add(zero_insertion_lowpass_conv(num_channels, pad_size=(kernel_size - 1) // 2))
        self.net.add(act(act_func))
        self.net.add(norm_layer(num_channels))
        self.net.add(nn.Conv2d(num_channels, num_output_channels, 1, 1, padding=0, bias=True))
         
        if need_sigmoid:
          self.net.add(nn.Sigmoid())

    def forward(self, x, scale_out=1):
        return self.net(x) * scale_out
        
class ConvDecoder_pip(nn.Module):
    def __init__(self, input_dim, num_layers, num_channels, num_output_channels, out_size, in_size, kernel_size=1, act_func='ReLU', upsample_mode='nearest', norm_func='bn', need_dropout=False,
                 need_sigmoid=False):
        super(ConvDecoder_pip, self).__init__()

        ### parameter setup
        strides = [1] * (num_layers - 1)

        norm_layer = nn.BatchNorm2d if norm_func == 'bn' else nn.InstanceNorm2d

        ### compute up-sampling factor from one layer to another
        scale_x, scale_y = (out_size[0] / in_size[0]) ** (1. / (num_layers - 1)), (out_size[1] / in_size[1]) ** (
                    1. / (num_layers - 1))
        hidden_size = [(int(np.ceil(scale_x ** n * in_size[0])),
                        int(np.ceil(scale_y ** n * in_size[1]))) for n in range(1, (num_layers - 1))] + [out_size]
        print(hidden_size)
        ### hidden layers
        self.net = nn.Sequential()

        for i in range(num_layers - 1):

#            self.net.add(nn.Upsample(size=hidden_size[i], mode=upsample_mode))
            if i==0:
               conv = nn.Conv2d(input_dim, num_channels, kernel_size, strides[i], padding=(kernel_size - 1) // 2,
                             bias=True)
            else:
               conv = nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size - 1) // 2,
                             bias=True)
            self.net.add(conv)
#            self.net.add(learnable_zero_insertion_lowpass_conv(num_channels, pad_size=(kernel_size - 1) // 2, outsize=hidden_size[i][0]))
            self.net.add(act(act_func))
            self.net.add(norm_layer(num_channels)) #nn.BatchNorm2d(num_channels, affine=True))
            if need_dropout:
                self.net.add(nn.Dropout2d(0.3))
        ### final layer
        self.net.add(
            nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size - 1) // 2, bias=True))
       # self.net.add(zero_insertion_lowpass_conv(num_channels, pad_size=(kernel_size - 1) // 2))
        self.net.add(act(act_func))
        self.net.add(norm_layer(num_channels))
        self.net.add(nn.Conv2d(num_channels, num_output_channels, 1, 1, padding=0, bias=True))
         
        if need_sigmoid:
          self.net.add(nn.Sigmoid())

    def forward(self, x, scale_out=1):
        return self.net(x) * scale_out

class ConvDecoder_improved(nn.Module):
    def __init__(self, num_layers, num_channels, num_output_channels, out_size, in_size, act_func='ReLU', upsample_mode='nearest', norm_func='bn', need_dropout=False,
                 need_sigmoid=False, num_ups=4):
        super(ConvDecoder_improved, self).__init__()

        ### parameter setup
        kernel_size = 3
        strides = [1] * (num_layers - 1)

        norm_layer = nn.BatchNorm2d if norm_func == 'bn' else nn.InstanceNorm2d

        ### compute up-sampling factor from one layer to another
        scale_x, scale_y = (out_size[0] / in_size[0]) ** (1. / (num_layers - 1)), (out_size[1] / in_size[1]) ** (
                    1. / (num_layers - 1))
        hidden_size = [(int(np.ceil(scale_x ** n * in_size[0])),
                        int(np.ceil(scale_y ** n * in_size[1]))) for n in range(1, (num_layers - 1))] + [out_size]

        ### hidden layers
        self.net = nn.Sequential()

       # for _ in range(4):
        #    self.net.add(nn.Conv2d(num_channels, num_channels, kernel_size, 1, padding=(kernel_size - 1) // 2, bias=True))
         #   self.net.add(act(act_func))
          #  self.net.add(norm_layer(num_channels))

        for i in range(num_layers - 1):
            if i == num_layers - 3:
##            if i==num_layers-3:
#               self.net.add(nn.Upsample(size=hidden_size[i], mode=upsample_mode))
               self.net.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            if i == num_layers - 2:
               self.net.add(nn.Upsample(size=out_size, mode=upsample_mode))
            conv = nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size - 1) // 2,
                             bias=True)
            self.net.add(conv)
            self.net.add(act(act_func))
            self.net.add(norm_layer(num_channels)) #nn.BatchNorm2d(num_channels, affine=True))
            if need_dropout:
                self.net.add(nn.Dropout2d(0.3))
        ### final layer
        self.net.add(
            nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size - 1) // 2, bias=True))
        self.net.add(norm_layer(num_channels))
        self.net.add(nn.Conv2d(num_channels, num_output_channels, 1, 1, padding=0, bias=True))
        if need_sigmoid:
            self.net.add(nn.Sigmoid())

    def forward(self, x, scale_out=1):
        return self.net(x) * scale_out

"""
Prunable with DAM
"Learning Compact Representations of Neural Networks using DiscriminAtive Masking (DAM)"
"""
class ConvDecoder_DAM(nn.Module):
    def __init__(self, num_layers, num_channels, num_output_channels, out_size, in_size, act_func='ReLU', upsample_mode='nearest', norm_func='bn', need_dropout=False,
                 need_sigmoid=False):
        super(ConvDecoder_DAM, self).__init__()

        ### parameter setup
        kernel_size = 3
        strides = [1] * (num_layers - 1)

        norm_layer = nn.BatchNorm2d if norm_func == 'bn' else nn.InstanceNorm2d

        ### compute up-sampling factor from one layer to another
        scale_x, scale_y = (out_size[0] / in_size[0]) ** (1. / (num_layers - 1)), (out_size[1] / in_size[1]) ** (
                    1. / (num_layers - 1))
        hidden_size = [(int(np.ceil(scale_x ** n * in_size[0])),
                        int(np.ceil(scale_y ** n * in_size[1]))) for n in range(1, (num_layers - 1))] + [out_size]

        ### hidden layers
        self.net = nn.Sequential()
        for i in range(num_layers - 1):

            self.net.add(nn.Upsample(size=hidden_size[i], mode=upsample_mode))

            conv = nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size - 1) // 2,
                             bias=True)
            self.net.add(conv)
            self.net.add(act(act_func))
            self.net.add(norm_layer(num_channels))
            self.net.add(DAM_2d(num_channels))
            if need_dropout:
                self.net.add(nn.Dropout2d(0.3))
        ### final layer
        self.net.add(
            nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size - 1) // 2, bias=True))
        self.net.add(norm_layer(num_channels))
        self.net.add(nn.Conv2d(num_channels, num_output_channels, 1, 1, padding=0, bias=True))
        if need_sigmoid:
            self.net.add(nn.Sigmoid())

    def forward(self, x, scale_out=1):
        return self.net(x) * scale_out

"""
Y-shape with two different branches leading to different outputs.
Weights are forced to be orthogonal by min | WW' - I |
"""
class BiConvDecoder(nn.Module):
    def __init__(self, num_layers, num_channels, num_output_channels, out_size, in_size, act_func='ReLU', upsample_mode='nearest', need_dropout=False,
                 need_sigmoid=False):
        super(BiConvDecoder, self).__init__()

        ### parameter setup
        kernel_size = 3
        strides = [1] * (num_layers - 1)

        ### compute up-sampling factor from one layer to another
        scale_x, scale_y = (out_size[0] / in_size[0]) ** (1. / (num_layers - 1)), (out_size[1] / in_size[1]) ** (
                    1. / (num_layers - 1))
        hidden_size = [(int(np.ceil(scale_x ** n * in_size[0])),
                        int(np.ceil(scale_y ** n * in_size[1]))) for n in range(1, (num_layers - 1))] + [out_size]

        ### hidden layers
        self.net = nn.Sequential()
        for i in range(num_layers - 1):

            self.net.add(nn.Upsample(size=hidden_size[i], mode=upsample_mode))

            conv = nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size - 1) // 2,
                             bias=True)
            self.net.add(conv)
            self.net.add(act(act_func))
            self.net.add(nn.BatchNorm2d(num_channels, affine=True))
            if need_dropout:
                self.net.add(nn.Dropout2d(0.3))
        ### final layer
        self.net.add(
            nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size - 1) // 2, bias=True))
        self.net.add(nn.BatchNorm2d(num_channels, affine=True))
        self.net.add(nn.Conv2d(num_channels, num_output_channels, 1, 1, padding=0, bias=True))
        if need_sigmoid:
            self.net.add(nn.Sigmoid())

    def forward(self, x, scale_out=1):
        return self.net(x) * scale_out





