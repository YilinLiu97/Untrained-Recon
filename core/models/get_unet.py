from unet.functions.train_unet import UnetMRIModel
from unet.functions.helpers import *
from unet.functions.include import *

import torch
import numpy as np

dtype = torch.cuda.FloatTensor

class args():
    def __init__(self):
        self.num_pools = 4
        self.drop_prob = 0
        self.num_chans = 64
        self.in_chans = 2

### create the model

def get_net():
  config = args()
  net = UnetMRIModel(config)
  ckp = torch.load('/shenlab/lab_stor/yilinliu/ttt/unet/unet_model_checkpoints/t1pre_supervised.pt', map_location='cpu')
  net.load_state_dict(ckp['model_state_dict'])
  return net

def scale_factor(slice, net, mask2d):
    
    slice = np.stack((slice.real, slice.imag), axis=-1)
    slice_tt = torch.from_numpy(slice).type(dtype)
    input = rss_torch(ifft2(slice_tt))[None,:]
    print(f"input: {input.shape}")
    input = torch.moveaxis(input , -1, 1 )
    print(f"input: {input.shape}")
    print(f"net: {type(net)}")

    for k in range(5):
        ###
        output = torch.moveaxis( net(input) , 1, -1 ) 
        print(f"output: {output.shape}")
        if k == 0:
            slice_tt_scale = slice_tt * torch.norm(output.detach()) / torch.norm(input.detach())
        else:
            slice_tt_scale *= torch.norm(output.detach()) / torch.norm(input.detach())
        input = torch.moveaxis( rss_torch(ifft2(slice_tt_scale))[None,:] , -1, 1 )

    scale = (torch.norm(slice_tt_scale) / torch.norm(slice_tt)).data.cpu().numpy()[np.newaxis][0]
    return scale

'''
sr, si = torch.rand(15,640,320), torch.rand(15,640,320)
slice = sr + 1j*si
net = get_net()

 
print(scale_factor(slice, net, torch.rand(1,640,320)))
'''
