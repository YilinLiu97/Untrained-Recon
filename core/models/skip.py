import torch
import torch.nn as nn
from .common import *
from deepsplines.ds_modules import dsnn
from .helper_shi import *
import antialiased_cnns
from models.resnet import ResNet
from unet3d import UNet3D, ResidualUNetSE3D

def DIP(
        num_input_channels=2, num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128],
        num_channels_skip=[4, 4, 4, 4, 4],
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True,
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU',
        need1x1_up=True):
    """Assembles encoder-decoder with skip connections.
    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')
    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down)

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode = [upsample_mode] * n_scales

    if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
        downsample_mode = [downsample_mode] * n_scales

    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
        filter_size_down = [filter_size_down] * n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up = [filter_size_up] * n_scales

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
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))

        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad,
                        downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
#        model_tmp.add(lowpass(num_channels_up[i])) # low pass
        model_tmp.add(act(act_fun))

        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model


def skip(
        out_size,num_input_channels=2, num_output_channels=3, 
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, lipschitz_constant=0, lipschitz_reg=0, norm_method='spectral',
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
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
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad, c=lipschitz_constant, norm_method=norm_method))
            skip.add(bn(num_channels_skip[i]))
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
        if need_dropout:
           deeper.add(nn.Dropout2d(p=0.3))

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
        if need_dropout: 
           model_tmp.add(nn.Dropout2d(p=0.3))

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
    
def skip_fixed_scaled_upsample(
        out_size,num_input_channels=2, num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4],
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, lipschitz_constant=0, lipschitz_reg=0,
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU',
        need1x1_up=True, need_dropout=False, need_tanh=False):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

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
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad, c=lipschitz_constant))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))
              
        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, c=lipschitz_constant, downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
        if need_dropout:
           deeper.add(nn.Dropout2d(p=0.3))

        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad, c=lipschitz_constant))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
        if need_dropout:
           deeper.add(nn.Dropout2d(p=0.3))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        
        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad, c=lipschitz_constant))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))
        if need_dropout:
           model_tmp.add(nn.Dropout2d(p=0.3))

        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad, c=lipschitz_constant))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad, c=lipschitz_constant, is_last=True))
    if need_sigmoid:
        model.add(nn.Sigmoid())
    elif need_tanh:
        model.add(nn.Tanh())

    return model


class Net_1by1(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, FNN_depth=4, norm_layer=nn.BatchNorm2d, use_bias=False, use_dropout=False):
        super(Net_1by1, self).__init__()
        
        model = [
        nn.Conv2d(input_nc, ngf, kernel_size=1, padding=0, bias=use_bias),
        norm_layer(ngf),
        nn.ReLU(True)
        ]
        # Originally 4 norm_layer

        for k in range(FNN_depth-1):
            model.append(nn.Conv2d(ngf, self.ngf, kernel_size=1, padding=0, bias=use_bias)) 
            model.append(norm_layer(ngf))
            model.append(nn.ReLU(True))
        
        model.append(nn.Conv2d(ngf, output_nc, kernel_size=1, padding=0))

        # model = [i for i in model if i is not None]
        self.model = nn.Sequential(*model)

    def forward(self, input):
       return self.model(input)   

class UniNet(nn.Module):
    def __init__(self, args):
        super(UniNet, self).__init__()
        num_channels = args.dim

        self.args = args

        self.net_T1 = skip(args.in_size,args.input_dim, args.out_chns, 
           num_channels_down = [num_channels] * args.num_scales,
           num_channels_up = [num_channels] * args.num_scales,
           num_channels_skip = [num_channels*0] * int(args.num_scales - args.num_skips) + [4] * int(args.num_skips),  
           filter_size_up = args.filter_size_up, filter_size_down = args.filter_size_down, 
           upsample_mode=args.upsample_mode, filter_skip_size=1,
           need_sigmoid=args.need_sigmoid, need_bias=True, lipschitz_constant=args.Lipschitz_constant, lipschitz_reg=args.Lipschitz_reg, pad=args.pad, act_fun=args.act_func, need_dropout=args.need_dropout, need_tanh=args.need_tanh)

        self.net_T2 = skip(args.in_size,args.input_dim, args.out_chns, 
           num_channels_down = [num_channels] * args.num_scales,
           num_channels_up = [num_channels] * args.num_scales,
           num_channels_skip = [num_channels*0] * int(args.num_scales - args.num_skips) + [4] * int(args.num_skips),  
           filter_size_up = args.filter_size_up, filter_size_down = args.filter_size_down, 
           upsample_mode=args.upsample_mode, filter_skip_size=1,
           need_sigmoid=args.need_sigmoid, need_bias=True, lipschitz_constant=args.Lipschitz_constant, lipschitz_reg=args.Lipschitz_reg, pad=args.pad, act_fun=args.act_func, need_dropout=args.need_dropout, need_tanh=args.need_tanh) 

        if args.learned_scaling:                     
           self.net_factor = skip(args.in_size,args.input_dim, 2, 
              num_channels_down = [num_channels] * args.num_scales,
              num_channels_up = [num_channels] * args.num_scales,
              num_channels_skip = [num_channels*0] * int(args.num_scales - args.num_skips) + [4] * int(args.num_skips),  
              filter_size_up = args.filter_size_up, filter_size_down = args.filter_size_down, 
              upsample_mode=args.upsample_mode, filter_skip_size=1,
              need_sigmoid=args.need_sigmoid, need_bias=True, lipschitz_constant=args.Lipschitz_constant, lipschitz_reg=args.Lipschitz_reg, pad=args.pad, act_fun=args.act_func, need_dropout=False, need_tanh=False)
        '''
        elif args.scale_transform:
           self.transform = Net_1by1(args.mrf_time_pts, args.mrf_time_pts, ngf=args.FNN_dim, FNN_depth=args.FNN_depth)
  #     self.factor = nn.Parameter(torch.ones(1)*2300)
        '''
        '''
        self.M0 = nn.Parameter(torch.zeros(1,1, 256, 256))
        '''
    def forward(self, x):
        T1 = self.net_T1(x) 
        T2 = self.net_T2(x) 
        if self.args.learned_scaling:
           factor = self.net_factor(x) # (1, 2, 256, 256)
           
        else:
           factor = None
        return torch.cat([T1, T2], 1), factor
        
class UniResNet(nn.Module):
    def __init__(self, args):
        super(UniResNet, self).__init__()
        num_channels = args.dim

        self.args = args

        self.net_T1 = ResNet(args.input_dim, args.out_chns, args.num_res_blocks, args.dim, need_sigmoid=False, act_fun=args.act_func)

        self.net_T2 = ResNet(args.input_dim, args.out_chns, args.num_res_blocks, args.dim, need_sigmoid=False, act_fun=args.act_func)

        if args.learned_scaling:
           self.net_factor = ResNet(args.input_dim, args.out_chns, args.num_res_blocks, args.dim, need_sigmoid=False, act_fun=args.act_func)
        '''
        elif args.scale_transform:
           self.transform = Net_1by1(args.mrf_time_pts, args.mrf_time_pts, ngf=args.FNN_dim, FNN_depth=args.FNN_depth)
  #     self.factor = nn.Parameter(torch.ones(1)*2300)
        '''
        '''
        self.M0 = nn.Parameter(torch.zeros(1,1, 256, 256))
        '''
    def forward(self, x):
        T1 = self.net_T1(x)
        T2 = self.net_T2(x)
        if self.args.learned_scaling:
           factor = self.net_factor(x) # (1, 2, 256, 256)
           
        else:
           factor = None
        return torch.cat([T1, T2], 1), factor


class UniNet3D(nn.Module):
    def __init__(self, args):
        super(UniNet3D, self).__init__()
        num_channels = args.dim

        self.args = args

        self.net_T1 = UNet3D(args.input_dim, args.out_chns, final_sigmoid=args.need_tanh, f_maps=args.dim)
        self.net_T2 = UNet3D(args.input_dim, args.out_chns, final_sigmoid=args.need_tanh, f_maps=args.dim)

        if args.learned_scaling:
           self.net_factor = UNet3D(args.input_dim, args.out_chns)
     
    def forward(self, x):
        T1 = self.net_T1(x)
        T2 = self.net_T2(x)
        if self.args.learned_scaling:
           factor = self.net_factor(x) # (1, 2, 256, 256)           
        else:
           factor = None
        return torch.cat([T1, T2], 1), factor
        

class UNet3D(nn.Module):
    def __init__(self, args):
        super(UNet3D, self).__init__()
        num_channels = args.dim

        self.args = args

        self.net = UNet3D(args.input_dim, args.out_chns, f_maps=args.dim)

    def forward(self, x):
        return self.net(x), None


def skip_SVD(
        out_size,num_input_channels=2, num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4],
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, lipschitz_constant=0, lipschitz_reg=0,
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU',
        need1x1_up=True, need_dropout=False, need_tanh=False):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

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
            skip.add(SVD_Conv2d(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))
              
        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        deeper.add(SVD_Conv2d(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
        if need_dropout:
           deeper.add(nn.Dropout2d(p=0.3))

        deeper.add(SVD_Conv2d(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
        if need_dropout:
           deeper.add(nn.Dropout2d(p=0.3))

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

        model_tmp.add(SVD_Conv2d(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))
        if need_dropout:
           model_tmp.add(nn.Dropout2d(p=0.3))

        if need1x1_up:
            model_tmp.add(SVD_Conv2d(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad, c=lipschitz_constant, is_last=True))
    if need_sigmoid:
        model.add(nn.Sigmoid())
    elif need_tanh:
        model.add(nn.Tanh())

    return model

class deepspline_skip(dsnn.DSModule):
    def __init__(self, out_size,num_input_channels=2, num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4],
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, lipschitz_constant=0, lipschitz_reg=0,
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='DeepSpline',
        need1x1_up=True):

        super(deepspline_skip, self).__init__()        
        """Assembles encoder-decoder with skip connections.

        Arguments:
            act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
            pad (string): zero|reflection (default: 'zero')
            upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
            downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

        """
        assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

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

        self.model = nn.Sequential()
        model_tmp = self.model

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
                skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad, c=lipschitz_constant))
                skip.add(bn(num_channels_skip[i]))
                skip.add(act(act_fun,chns=num_channels_skip[i]))
                  
            # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

            deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, c=lipschitz_constant, downsample_mode=downsample_mode[i]))
            deeper.add(bn(num_channels_down[i]))
            deeper.add(act(act_fun,chns=num_channels_down[i]))
    #        deeper.add(nn.Dropout2d(p=0.3))

            deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad, c=lipschitz_constant))
            deeper.add(bn(num_channels_down[i]))
            deeper.add(act(act_fun,chns=num_channels_down[i]))
    #        deeper.add(nn.Dropout2d(p=0.3))

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

            model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad, c=lipschitz_constant))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun,chns=num_channels_up[i]))
     #       model_tmp.add(nn.Dropout2d(p=0.3))

            if need1x1_up:
                model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad, c=lipschitz_constant))
                model_tmp.add(bn(num_channels_up[i]))
                model_tmp.add(act(act_fun,chns=num_channels_up[i]))

            input_depth = num_channels_down[i]
            model_tmp = deeper_main

        self.model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad, c=0, is_last=True))
        if need_sigmoid:
            self.model.add(nn.Sigmoid())

    def forward(self, x):
        return self.model(x)


def transposed_skip(
        out_size,num_input_channels=2, num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4],
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, lipschitz_constant=0, lipschitz_reg=0, norm_method='spectral',
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU',
        need1x1_up=True, need_dropout=False, need_tanh=False):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

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
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))
              
        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
#        deeper.add(nn.Dropout2d(p=0.3))

        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
#        deeper.add(nn.Dropout2d(p=0.3))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        if i == 0:
           deeper.add(nn.Upsample(size=out_size, mode='nearest'))
        else:
           deeper.add(nn.ConvTranspose2d(num_channels_up[i], num_channels_up[i], kernel_size=2, stride=2))

        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))
 #       model_tmp.add(nn.Dropout2d(p=0.3))

        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad, is_last=True))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model

def skip_lpf(
        out_size,num_input_channels=2, num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4],
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, lipschitz_constant=0, lipschitz_reg=0,
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU',
        need1x1_up=True, need_dropout=False, need_tanh=False, need_relu=False):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)
    
    print('=============================== DIP ==================================================')
    print(f'num_scales:{len(num_channels_down)} num_channels:{num_channels_up[0]} out_size:{out_size} in_size:{out_size}\n'
    f'act_func:{act_fun} upsample_mode:{upsample_mode} need_dropout:{need_dropout} need_sigmoid:{need_sigmoid}')
    print('=============================== DIP ==================================================')
    
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
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad, c=lipschitz_constant))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))
              
        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, c=lipschitz_constant, downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
        if need_dropout:
           deeper.add(nn.Dropout2d(p=0.3))

        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad, c=lipschitz_constant))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
        if need_dropout:
           deeper.add(nn.Dropout2d(p=0.3))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        if i == 0:
            deeper.add(nn.Upsample(size=out_size, mode='nearest'))
        else:
            if upsample_mode[i] == 'nearest' or upsample_mode[i] == 'bilinear' or upsample_mode[i] == 'bicubic':
                deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))
            else:
                deeper.add(InsertZeros(2, 2, gain=1.0))
                if upsample_mode[i] == 'LPF1':
                    deeper.add(lowpass_conv3(num_channels_up[i], w1, pad_mode = 'reflect', gain=4.0))
                    print(f"used LPF1 !!!!!!!!!!!!!!!!!!!1")
                elif upsample_mode[i] == 'LPF2':
                    deeper.add(lowpass_conv3(num_channels_up[i], w2, pad_mode = 'reflect', gain=4.0))
                    print(f"used LPF2 !!!!!!!!!!!!!!!!!!!!")
                elif upsample_mode[i] == 'LPF3':
                    deeper.add(lowpass_conv3(num_channels_up[i], w3, pad_mode = 'reflect', gain=4.0))
                    print(f"used LPF3 !!!!!!!!!!!!!!!!!!!!")
                elif upsample_mode[i] == 'LPF4':
                    deeper.add(lowpass_conv3(num_channels_up[i], w4, pad_mode = 'reflect', gain=4.0))
                    print(f"used LPF4 !!!!!!!!!!!!!!!!!!!!")
                elif upsample_mode[i] == 'LPF14':
                    deeper.add(lowpass_conv3(num_channels_up[i], w14, pad_mode = 'reflect', gain=4.0))
                    print(f"used LPF14 !!!!!!!!!!!!!!!!!!!!")
                elif upsample_mode[i] == 'LPF5':
                    deeper.add(lowpass_conv3(num_channels_up[i], w5, pad_mode = 'zeros', gain=4.0))
                    print(f"used LPF5 !!!!!!!!!!!!!!!!!!!!")
                elif upsample_mode[i] == 'LPF6':
                    deeper.add(lowpass_conv3(num_channels_up[i], w6, pad_mode = 'zeros', gain=4.0))
                    print(f"used LPF6 !!!!!!!!!!!!!!!!!!!!")
                else:
                    print(f'Upsample Mode:{upsample_mode} not supported! Use Zero Insertion Only!')
#                deeper.add(nn.Upsample(scale_factor=2, mode='bilinear'))
                #self.net.add(nn.Upsample(size=hidden_size[i], mode='bilinear'))
                #self.net.add(lowpass_conv3(num_channels, w2, pad_mode = 'reflect', gain=4.0))
        

        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad, c=lipschitz_constant))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))
        if need_dropout:
           model_tmp.add(nn.Dropout2d(p=0.3))

        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad, c=lipschitz_constant))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad, c=lipschitz_constant, is_last=True))
    if need_sigmoid:
        model.add(nn.Sigmoid())
    elif need_tanh:
        model.add(nn.Tanh())
    elif need_relu:
        model.add(nn.ReLU(True))

    return model

def gaussian_skip(
        out_size,num_input_channels=2, num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4],
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, lipschitz_constant=0, lipschitz_reg=0, norm_method='spectral',
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU',
        need1x1_up=True, need_dropout=False, need_tanh=False, gaussian_up_ks=3, gaussian_up_sigma=0.1):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down)

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode   = [upsample_mode]*n_scales

    if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
        downsample_mode   = [downsample_mode]*n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
        filter_size_down   = [filter_size_down]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales

    if not (isinstance(gaussian_up_ks, list) or isinstance(gaussian_up_ks, tuple)) :
        gaussian_up_ks   = [gaussian_up_ks]*n_scales

    if not (isinstance(gaussian_up_sigma, list) or isinstance(gaussian_up_sigma, tuple)) :
        gaussian_up_sigma   = [gaussian_up_sigma]*n_scales

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
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))
              
        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
#        deeper.add(nn.Dropout2d(p=0.3))

        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
#        deeper.add(nn.Dropout2d(p=0.3))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        #if i == 0:
         #  deeper.add(nn.Upsample(size=out_size, mode='nearest'))
        #else:
        print(f"num_channels_down: {num_channels_down}, i: {i}, gaussian_up_ks: {gaussian_up_ks}, gaussian_up_sigma: {gaussian_up_sigma}")
        deeper.add(gaussian(num_channels_up[i], kernel_width=gaussian_up_ks[i], sigma=gaussian_up_sigma[i]))

        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))
 #       model_tmp.add(nn.Dropout2d(p=0.3))

        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad, is_last=True))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model
    
def kaiser_skip(
        out_size,num_input_channels=2, num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4],
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, lipschitz_constant=0, lipschitz_reg=0, norm_method='spectral',
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU',
        need1x1_up=True, need_dropout=False, need_tanh=False, kaiser_up_ks=3, kaiser_up_beta=3, antialiased=False):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down)

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode   = [upsample_mode]*n_scales

    if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
        downsample_mode   = [downsample_mode]*n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
        filter_size_down   = [filter_size_down]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales

    if not (isinstance(kaiser_up_ks, list) or isinstance(kaiser_up_ks, tuple)) :
        kaiser_up_ks   = [kaiser_up_ks]*n_scales

    if not (isinstance(kaiser_up_beta, list) or isinstance(kaiser_up_beta, tuple)) :
        kaiser_up_beta   = [kaiser_up_beta]*n_scales

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
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))
              
        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        if not antialiased:
           deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
           deeper.add(bn(num_channels_down[i]))
           deeper.add(act(act_fun))
        else:
           deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 1, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
           deeper.add(bn(num_channels_down[i]))
           deeper.add(act(act_fun))
           deeper.add(antialiased_cnns.BlurPool(num_channels_down[i], stride=2))
#        deeper.add(nn.Dropout2d(p=0.3))

        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
#        deeper.add(nn.Dropout2d(p=0.3))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        #if i == 0:
         #  deeper.add(nn.Upsample(size=out_size, mode='nearest'))
        #else:
        print(f"num_channels_down: {num_channels_down}, i: {i}, kaiser_up_ks: {kaiser_up_ks}, kaiser_up_beta: {kaiser_up_beta}")
        deeper.add(kaiser(num_channels_up[i], kernel_width=kaiser_up_ks[i], beta=kaiser_up_beta[i]))

        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))
 #       model_tmp.add(nn.Dropout2d(p=0.3))

        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad, is_last=True))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model
    
    
def lowpassed_skip(
        out_size,num_input_channels=2, num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4],
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True,
        pad='zero', upsample_mode='transposed', downsample_mode='stride', act_fun='LeakyReLU',
        need1x1_up=True):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

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
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))
              
        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
#        deeper.add(nn.Dropout2d(p=0.3))

        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
#        deeper.add(nn.Dropout2d(p=0.3))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(zero_insertion_lowpass_conv(num_channels_up[i], pad_size=1))

        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))
 #       model_tmp.add(nn.Dropout2d(p=0.3))

        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad, is_last=True))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model

