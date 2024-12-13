import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import torch

def getModel(args):
    if args.model_type == 'ConvDecoder':
        from models.light_cnn import ConvDecoder
        model = ConvDecoder(args, args.num_layers, args.dim, args.out_chns, args.out_size,
                                args.in_size, args.act_func, args.upsample_mode, args.norm_func, args.need_dropout,args.need_sigmoid)

    elif args.model_type == 'ConvDecoder_LPF':
        from models.ConvDecoder import ConvDecoder
        model = ConvDecoder(args.num_layers, args.dim, args.out_chns, args.out_size,
                                args.in_size, args.act_func, args.upsample_mode, args.norm_func, args.need_dropout,args.need_sigmoid)

    elif args.model_type == 'ConvDecoder_ours':
       from models.light_cnn import ConvDecoder_improved
       model = ConvDecoder_improved(args.num_layers, args.dim, args.out_chns, args.out_size,
                                args.in_size, args.act_func, args.upsample_mode, args.norm_func, args.need_dropout,args.need_sigmoid)

    elif args.model_type == 'sup_unet':
        from get_unet import get_net
        model = get_net()

    elif args.model_type == 'Decoder_Shi':
        from models.decoder_shi import decoder
        model = decoder(args.dim, need_sigmoid=args.need_sigmoid)

    elif args.model_type == 'DIP_2': # the DIP setting in ConvDecoder Paper
        from models.skip import skip
        num_channels = 256
        model = skip(args.in_size,args.input_dim, args.out_chns, 
           num_channels_down = [num_channels] * 8,
           num_channels_up =   [num_channels] * 8,
           num_channels_skip =    [num_channels*0] * 6 + [4,4],  
           filter_size_up = 3, filter_size_down = 5, 
           upsample_mode='nearest', filter_skip_size=1,
           need_sigmoid=False, need_bias=True, pad='zero', act_fun='ReLU')

    elif args.model_type == 'DIP_2_scaled': # the DIP in ConvDecoder Paper but with customized scales
        from models.skip import skip
        num_channels = args.dim
        model = skip(args.in_size,args.input_dim, args.out_chns, 
           num_channels_down = [num_channels] * args.num_scales,
           num_channels_up = [num_channels] * args.num_scales,
           num_channels_skip =    [num_channels*0] * int(args.num_scales - args.num_skips) + [4] * int(args.num_skips),  
           filter_size_up = args.filter_size_up, filter_size_down = args.filter_size_down, 
           upsample_mode=args.upsample_mode, downsample_mode=args.downsample_mode, filter_skip_size=1,
           need_sigmoid=args.need_sigmoid, need_bias=args.need_bias, lipschitz_constant=args.Lipschitz_constant, lipschitz_reg=args.Lipschitz_reg, norm_method=args.weight_norm_method, pad=args.pad, act_fun=args.act_func, need_dropout=args.need_dropout, need_tanh=args.need_tanh)        

    elif args.model_type == 'DIP_hourglass':
        from models.skip import skip_hourglass
        num_channels = args.dim
        model = skip_hourglass(args.in_size,args.input_dim, args.out_chns, 
           num_channels_down = [16, 32, 64, 128, 128, 128][:args.num_scales],
           num_channels_up = [16, 32, 64, 128, 128, 128][:args.num_scales],
           num_channels_skip =    [num_channels*0] * int(args.num_scales - args.num_skips) + [4] * int(args.num_skips),  
           filter_size_up = args.filter_size_up, filter_size_down = args.filter_size_down, 
           upsample_mode=args.upsample_mode, downsample_mode=args.downsample_mode, filter_skip_size=1,
           need_sigmoid=args.need_sigmoid, need_bias=args.need_bias, lipschitz_constant=args.Lipschitz_constant, lipschitz_reg=args.Lipschitz_reg, norm_method=args.weight_norm_method, pad=args.pad, act_fun=args.act_func, need_dropout=args.need_dropout, need_tanh=args.need_tanh) 

    elif args.model_type == 'UNet':
        from models.unet import UNet
        model = UNet(num_input_channels=args.input_dim, num_output_channels=args.out_chns, 
                   feature_scale=8, more_layers=1, 
                   concat_x=False, upsample_mode='deconv', 
                   pad='zero', norm_layer=torch.nn.InstanceNorm2d, need_sigmoid=True, need_bias=True)

    elif args.model_type == 'ResNet':
        from models.resnet import ResNet
        model = ResNet(args.input_dim, args.out_chns, args.num_res_blocks, args.dim, need_sigmoid=True, act_fun=args.act_func)

    elif args.model_type == 'DIP_svd': # SVD decomposed CNN
        from models.skip import skip_SVD
        num_channels = args.dim
        model = skip_SVD(args.in_size,args.input_dim, args.out_chns, 
           num_channels_down = [num_channels] * args.num_scales,
           num_channels_up =   [num_channels] * args.num_scales,
           num_channels_skip =    [num_channels*0] * int(args.num_scales - args.num_skips) + [4] * int(args.num_skips),  
           filter_size_up = args.filter_size_up, filter_size_down = args.filter_size_down, 
           upsample_mode=args.upsample_mode, filter_skip_size=1,
           need_sigmoid=args.need_sigmoid, need_bias=args.need_bias, lipschitz_constant=args.Lipschitz_constant, lipschitz_reg=args.Lipschitz_reg, pad=args.pad, act_fun=args.act_func, need_dropout=args.need_dropout, need_tanh=args.need_tanh)


    elif args.model_type == 'UniNet':
        from models.skip import UniNet
        model = UniNet(args)

    elif args.model_type == 'UniResNet':
        from models.skip import UniResNet
        model = UniResNet(args)

    elif args.model_type == 'UniNet3D':
        from models.skip import UniNet3D
        model = UniNet3D(args)

    elif args.model_type == 'UNet3D':
        from models.skip import UNet3D
        model = UNet3D(args)

    elif args.model_type == 'complex_UniNet':
        from models.complex_skip import complex_UniNet
        model = complex_UniNet(args)

    elif args.model_type == 'CoordRegress':
        from models.skip import CoordRegress_skip
        num_channels = args.dim
        model = CoordRegress_skip(args.in_size,args.input_dim, args.out_chns, 
           num_channels_down = [num_channels] * args.num_scales,
           num_channels_up =   [num_channels] * args.num_scales,
           num_channels_skip =    [num_channels*0] * int(args.num_scales - args.num_skips) + [4] * int(args.num_skips),  
           filter_size_up = args.filter_size_up, filter_size_down = args.filter_size_down, 
           upsample_mode=args.upsample_mode, filter_skip_size=1,
           need_sigmoid=args.need_sigmoid, need_bias=True, lipschitz_constant=args.Lipschitz_constant, lipschitz_reg=args.Lipschitz_reg, pad=args.pad, act_fun=args.act_func, need_dropout=args.need_dropout, need_tanh=args.need_tanh)


    elif args.model_type == 'multiscale_mlp':
        from models.pip import skip_pip
        skip_n33d=args.dim
        skip_n33u=args.dim 
        skip_n11=4
        num_scales=5
        downsample_mode='stride' 
        n_channels = 3
        input_depth = 32
        pad = 'reflection'
        act_fun = 'LeakyReLU'
        upsample_mode = 'bilinear'
        model = skip_pip(args.input_dim, n_channels, num_channels_down = [skip_n33d]*args.num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*args.num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*args.num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=args.upsample_mode, downsample_mode=args.downsample_mode,
                                            need_sigmoid=args.need_sigmoid, need_bias=True, pad=args.pad, act_fun=args.act_func,
                   filter_size_down=1, filter_size_up=1, filter_skip_size=1)


    elif args.model_type == 'pure_mlp':
        from models.light_cnn import ConvDecoder_pip
        ks = 1
        model = ConvDecoder_pip(args.input_dim, args.num_layers, args.dim, args.out_chns, args.out_size,
                                args.out_size, kernel_size=ks, act_func=args.act_func, upsample_mode=args.upsample_mode, need_dropout=False,need_sigmoid=args.need_sigmoid) 
    elif args.model_type == 'vit':
        from models.vit import MiniViT2
        model = MiniViT2(args.input_dim)

    elif args.model_type == 'DIP_ds': # the DIP in ConvDecoder Paper but with customized scales
        from models.skip import deepspline_skip
        num_channels = args.dim
        model = deepspline_skip(args.in_size,num_channels, args.out_chns, 
           num_channels_down = [num_channels] * args.num_scales,
           num_channels_up =   [num_channels] * args.num_scales,
           num_channels_skip =    [num_channels*0] * int(args.num_scales - args.num_skips) + [4] * int(args.num_skips),  
           filter_size_up = args.filter_size_up, filter_size_down = args.filter_size_down, 
           upsample_mode=args.upsample_mode, filter_skip_size=1,
           need_sigmoid=args.need_sigmoid, need_bias=True, lipschitz_constant=args.Lipschitz_constant, lipschitz_reg=args.Lipschitz_reg, pad=args.pad, act_fun='DeepSpline')
      
    elif args.model_type == 'DIP_scaled_transposed': # the DIP in ConvDecoder Paper but with customized scales
        from models.skip import transposed_skip
        num_channels = args.dim
        model = transposed_skip(args.in_size,num_channels, args.out_chns, 
           num_channels_down = [num_channels] * args.num_scales,
           num_channels_up =   [num_channels] * args.num_scales,
           num_channels_skip =    [num_channels*0] * int(args.num_scales - args.num_skips) + [4] * int(args.num_skips),  
           filter_size_up = args.filter_size_up, filter_size_down = args.filter_size_down, 
           upsample_mode=args.upsample_mode, downsample_mode=args.downsample_mode,filter_skip_size=1,
           need_sigmoid=args.need_sigmoid, need_bias=True, pad=args.pad, act_fun=args.act_func)  

    elif args.model_type == 'DIP_LPF':
        from models.skip import skip_lpf
        num_channels = args.dim
        model = skip_lpf(args.in_size,args.input_dim, args.out_chns, 
           num_channels_down = [num_channels] * args.num_scales,
           num_channels_up = [num_channels] * args.num_scales,
           num_channels_skip = [num_channels*0] * int(args.num_scales - args.num_skips) + [4] * int(args.num_skips),  
           filter_size_up = args.filter_size_up, filter_size_down = args.filter_size_down, 
           upsample_mode=args.upsample_mode, filter_skip_size=1,
           need_sigmoid=args.need_sigmoid, need_bias=True, lipschitz_constant=args.Lipschitz_constant, lipschitz_reg=args.Lipschitz_reg, pad=args.pad, act_fun=args.act_func, need_dropout=args.need_dropout, need_tanh=args.need_tanh)

    elif args.model_type == 'DIP_gaussian':
        from models.skip import gaussian_skip
        num_channels = args.dim
        model = gaussian_skip(args.in_size,args.input_dim, args.out_chns, 
           num_channels_down = [num_channels] * args.num_scales,
           num_channels_up = [num_channels] * args.num_scales,
           num_channels_skip = [num_channels*0] * int(args.num_scales - args.num_skips) + [4] * int(args.num_skips),  
           filter_size_up = args.filter_size_up, filter_size_down = args.filter_size_down, 
           upsample_mode=args.upsample_mode, filter_skip_size=1,
           need_sigmoid=args.need_sigmoid, need_bias=True, lipschitz_constant=args.Lipschitz_constant, lipschitz_reg=args.Lipschitz_reg, pad=args.pad, act_fun=args.act_func, need_dropout=args.need_dropout, need_tanh=args.need_tanh, gaussian_up_ks=args.gaussian_up_ks, gaussian_up_sigma=args.gaussian_up_sigma)

    elif args.model_type == 'DIP_kaiser':
        from models.skip import kaiser_skip
        num_channels = args.dim
        model = kaiser_skip(args.in_size,args.input_dim, args.out_chns, 
           num_channels_down = [num_channels] * args.num_scales,
           num_channels_up = [num_channels] * args.num_scales,
           num_channels_skip = [num_channels*0] * int(args.num_scales - args.num_skips) + [4] * int(args.num_skips),  
           filter_size_up = args.filter_size_up, filter_size_down = args.filter_size_down, 
           upsample_mode=args.upsample_mode, filter_skip_size=1,
           need_sigmoid=args.need_sigmoid, need_bias=True, lipschitz_constant=args.Lipschitz_constant, lipschitz_reg=args.Lipschitz_reg, pad=args.pad, act_fun=args.act_func, need_dropout=args.need_dropout, need_tanh=args.need_tanh, kaiser_up_ks=args.kaiser_up_ks, kaiser_up_beta=args.kaiser_up_beta, antialiased=args.antialiased_down)


    elif args.model_type == 'DIP_scaled_lowpassed': # the DIP in ConvDecoder Paper but with customized scales
        from models.skip import lowpassed_skip
        num_channels = args.dim
        model = lowpassed_skip(args.in_size,num_channels, args.out_chns, 
           num_channels_down = [num_channels] * args.num_scales,
           num_channels_up =   [num_channels] * args.num_scales,
           num_channels_skip =    [num_channels*0] * int(args.num_scales - args.num_skips) + [4] * int(args.num_skips),  
           filter_size_up = args.filter_size_up, filter_size_down = args.filter_size_down, 
           upsample_mode=args.upsample_mode, filter_skip_size=1,
           need_sigmoid=args.need_sigmoid, need_bias=True, pad=args.pad, act_fun=args.act_func)

    elif args.model_type == 'DIP_orig':
        from models.skip import DIP
        skip_n33d, skip_n33u, skip_n11 = 128, 128, 4
        model = DIP(args.input_dim, args.out_chns, num_channels_down = [skip_n33d]*5 if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*5 if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*5 if isinstance(skip_n11, int) else skip_n11,
                                            filter_size_up=3, filter_size_down=3, filter_skip_size=1,
                                            upsample_mode='bilinear', downsample_mode='stride',
                                            need_sigmoid=args.need_sigmoid, need_bias=True, pad='reflection', act_fun='LeakyReLU')

    elif args.model_type == 'small_DIP':
        from models.skip import skip
        num_channels = 256
        model = skip(args.in_size,num_channels, args.out_chns, 
              num_channels_down = [num_channels] * 2,
              num_channels_up =   [num_channels] * 2,
              num_channels_skip =    [num_channels*0] * 0 + [4,4],  
              filter_size_up = 3, filter_size_down = 5, 
              upsample_mode='nearest', filter_skip_size=1,
              need_sigmoid=False, need_bias=True, pad="zero", act_fun='ReLU')

    elif args.model_type == 'DD_orig':
        from models.DD import decodernw
        num_channels = args.dim
        model = decodernw(args.out_chns,
        [num_channels]*args.num_layers
        ) 
    elif args.model_type == 'DD':        
        from models.decoder_skip import skipdecoder
        num_channels = args.dim
        model = skipdecoder(args, args.out_size,
        args.in_size,
        args.out_chns, 
        args.num_layers,
        num_channels,
        skips=False,
        need_last=True,
        need_sigmoid=args.need_sigmoid,
        upsample_mode=args.upsample_mode) 
        
    elif args.model_type == 'PruneDIP':
        from models.prune_dip import Unet_struc
        print('<<<<<<<<<<<<<<<<<<<< exp_name 2:', args.exp_name)
        model = Unet_struc(args.exp_name, args.out_size, args.dim,
        output_nc=args.out_chns,
        ngf=args.dim,
        upsample_mode=args.upsample_mode, adaptive_halting=True, ponder_lbda_p=args.exit_layer_idx_prior, ponder_epsilon=0.05)

    elif args.model_type == 'hidden_net':
        from models.hidden_net import hidden_skip
        num_channels = args.dim
        model = hidden_skip(args.in_size,num_channels, args.out_chns, 
           num_channels_down = [num_channels] * args.num_scales,
           num_channels_up =   [num_channels] * args.num_scales,
           num_channels_skip =    [num_channels*0] * int(args.num_scales - args.num_skips) + [4] * int(args.num_skips),  
           filter_size_up = args.filter_size_up, filter_size_down = args.filter_size_down, 
           upsample_mode=args.upsample_mode, filter_skip_size=1,
           need_sigmoid=args.need_sigmoid, need_bias=True, pad=args.pad, act_fun=args.act_func)

    elif args.model_type == 'Unet_small':
        from models.unet import Unet_small
        model = Unet_small(args.dim,args.out_chns)

    else:
        raise NotImplementedError("No such model implemented.")
    return model
