import torchcomplex.nn as nn
from complex_ops import *
#import torch.nn as nn
import torch
#from .common import *
from torch.nn.utils import skip_init

def complex_skip(
        out_size,num_input_channels=2, num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4],
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, lipschitz_constant=0, lipschitz_reg=0, norm_method='spectral',
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='ReLU',
        need1x1_up=True, need_dropout=False, need_tanh=False):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
#    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    if lipschitz_reg:
       lipschitz_constant = -1

    n_scales = len(num_channels_down)

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode   = [upsample_mode]*n_scales

    if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
        downsample_mode   = [downsample_mode]*n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
        filter_size_down   = [filter_size_down]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales

    last_scale = n_scales - 1

    cur_depth = None

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)
        
        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:
            convlayer = conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad, c=lipschitz_constant, norm_method=norm_method)
#            print(f"convlayer: {convlayer.weight}")
            skip.add(convlayer)
            skip.add(bn(num_channels_skip[i]))
#            print(f"bn: bn(num_channels_skip[i])")
            skip.add(act(act_fun))
              
        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, c=lipschitz_constant, norm_method=norm_method, downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
        if need_dropout:
           deeper.add(nn.Dropout2d(p=0.3))

        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad, c=lipschitz_constant, norm_method=norm_method))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
        #if need_dropout:
        #   deeper.add(nn.Dropout2d(p=0.3))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        if i == 0:
            deeper.add(nn.Upsample(size=out_size, mode=upsample_mode[i]))
        else:
            deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad, c=lipschitz_constant, norm_method=norm_method))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))
#        if need_dropout:
#           model_tmp.add(nn.Dropout2d(p=0.3))

        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad, c=lipschitz_constant, norm_method=norm_method))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad, c=lipschitz_constant, norm_method=norm_method, is_last=True))
    if need_sigmoid:
        model.add(nn.Sigmoid())
    elif need_tanh:
        model.add(nn.Tanh())

    return model
    
    
def complex_UniNet2(args):

    num_channels = args.dim

    net_T1 = complex_skip(args.in_size,args.input_dim, args.out_chns,
        num_channels_down = [num_channels] * args.num_scales,
        num_channels_up = [num_channels] * args.num_scales,
        num_channels_skip = [num_channels*0] * int(args.num_scales - args.num_skips) + [4] * int(args.num_skips),
        filter_size_up = args.filter_size_up, filter_size_down = args.filter_size_down,
        upsample_mode=args.upsample_mode, filter_skip_size=1,
        need_sigmoid=args.need_sigmoid, need_bias=True, lipschitz_constant=args.Lipschitz_constant, lipschitz_reg=args.Lipschitz_reg, pad=args.pad, act_fun=args.act_func, need_dropout=args.need_dropout, need_tanh=args.need_tanh)

    net_T2 = complex_skip(args.in_size,args.input_dim, args.out_chns,
        num_channels_down = [num_channels] * args.num_scales,
        num_channels_up = [num_channels] * args.num_scales,
        num_channels_skip = [num_channels*0] * int(args.num_scales - args.num_skips) + [4] * int(args.num_skips),
        filter_size_up = args.filter_size_up, filter_size_down = args.filter_size_down,
        upsample_mode=args.upsample_mode, filter_skip_size=1,
        need_sigmoid=args.need_sigmoid, need_bias=True, lipschitz_constant=args.Lipschitz_constant, lipschitz_reg=args.Lipschitz_reg, pad=args.pad, act_fun=args.act_func, need_dropout=args.need_dropout, need_tanh=args.need_tanh)

    if args.learned_scaling:
       net_factor = complex_skip(args.in_size,args.input_dim, 2,
          num_channels_down = [num_channels] * args.num_scales,
          num_channels_up = [num_channels] * args.num_scales,
          num_channels_skip = [num_channels*0] * int(args.num_scales - args.num_skips) + [4] * int(args.num_skips),
          filter_size_up = args.filter_size_up, filter_size_down = args.filter_size_down,
          upsample_mode=args.upsample_mode, filter_skip_size=1,
          need_sigmoid=args.need_sigmoid, need_bias=True, lipschitz_constant=args.Lipschitz_constant, lipschitz_reg=args.Lipschitz_reg, pad=args.pad, act_fun=args.act_func, need_dropout=args.need_dropout, need_tanh=args.need_tanh)
    else:
       net_factor = None
          
    return net_T1, net_T2
    
class complex_UniNet(nn.Module):
    def __init__(self, args):
        super(complex_UniNet, self).__init__()
        num_channels = args.dim

        self.args = args
  

        self.net_T1 = complex_skip(args.in_size,args.input_dim, args.out_chns, 
           num_channels_down = [num_channels] * args.num_scales,
           num_channels_up = [num_channels] * args.num_scales,
           num_channels_skip = [num_channels*0] * int(args.num_scales - args.num_skips) + [4] * int(args.num_skips),  
           filter_size_up = args.filter_size_up, filter_size_down = args.filter_size_down, 
           upsample_mode=args.upsample_mode, filter_skip_size=1,
           need_sigmoid=args.need_sigmoid, need_bias=True, lipschitz_constant=args.Lipschitz_constant, lipschitz_reg=args.Lipschitz_reg, pad=args.pad, act_fun=args.act_func, need_dropout=args.need_dropout, need_tanh=args.need_tanh)

        self.net_T2 = complex_skip(args.in_size,args.input_dim, args.out_chns, 
           num_channels_down = [num_channels] * args.num_scales,
           num_channels_up = [num_channels] * args.num_scales,
           num_channels_skip = [num_channels*0] * int(args.num_scales - args.num_skips) + [4] * int(args.num_skips),  
           filter_size_up = args.filter_size_up, filter_size_down = args.filter_size_down, 
           upsample_mode=args.upsample_mode, filter_skip_size=1,
           need_sigmoid=args.need_sigmoid, need_bias=True, lipschitz_constant=args.Lipschitz_constant, lipschitz_reg=args.Lipschitz_reg, pad=args.pad, act_fun=args.act_func, need_dropout=args.need_dropout, need_tanh=args.need_tanh)

        if args.learned_scaling:                     
           self.net_factor = complex_skip(args.in_size,args.input_dim, 2, 
              num_channels_down = [num_channels] * args.num_scales,
              num_channels_up = [num_channels] * args.num_scales,
              num_channels_skip = [num_channels*0] * int(args.num_scales - args.num_skips) + [4] * int(args.num_skips),  
              filter_size_up = args.filter_size_up, filter_size_down = args.filter_size_down, 
              upsample_mode=args.upsample_mode, filter_skip_size=1,
              need_sigmoid=args.need_sigmoid, need_bias=True, lipschitz_constant=args.Lipschitz_constant, lipschitz_reg=args.Lipschitz_reg, pad=args.pad, act_fun=args.act_func, need_dropout=args.need_dropout, need_tanh=args.need_tanh)
       
        '''
        elif args.scale_transform:
           self.transform = Net_1by1(args.mrf_time_pts, args.mrf_time_pts, ngf=args.FNN_dim, FNN_depth=args.FNN_depth)
  #     self.factor = nn.Parameter(torch.ones(1)*2300)
        '''
        '''
        self.M0 = nn.Parameter(torch.zeros(1,1, 256, 256))
        '''
    def forward(self, x):       
#        for _, param in self.net_T1.named_parameters():
#           if torch.sum(torch.abs(param))>0:
#              print(param)
     
        T1 = self.net_T1(x) 
        T2 = self.net_T2(x) 
        if self.args.learned_scaling:
           factor = self.net_factor(x) # (1, 2, 256, 256)
           
        else:
           factor = None
        return torch.cat([T1, T2], 1), factor


'''
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--out_chns', default=2)
parser.add_argument('--in_size', default=[256,256])
parser.add_argument('--num_scales', default=2)
parser.add_argument('--num_skips', default=2)
parser.add_argument('--filter_size_up', default=3)
parser.add_argument('--filter_size_down', default=3)
parser.add_argument('--pad', default='zero')
parser.add_argument('--Lipschitz_constant', default=0)
parser.add_argument('--Lipschitz_reg', default=0)
parser.add_argument('--need_sigmoid', default=False)
parser.add_argument('--upsample_mode', default='nearest')
parser.add_argument('--learned_scaling', default=False)
parser.add_argument('--dim', default=64)
parser.add_argument('--input_dim', default=256)
parser.add_argument('--act_func', default='ReLU')
parser.add_argument('--need_dropout', default=False)
parser.add_argument('--need_tanh', default=False)
parser.add_argument('--need_bias', default=True)

args = parser.parse_args()

input = torch.view_as_complex(torch.rand(1,256,256,256,2))
net = complex_UniNet2(args)
out = net(input)
print(out)
'''
'''
net_T1, net_T2 = complex_UniNet(args)
out_T1 = net_T1(input)
out_T2 = net_T2(input)
print(out_T1)
'''

