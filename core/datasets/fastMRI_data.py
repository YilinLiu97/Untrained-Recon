import torch
from torch.utils.data import Dataset

import os
import numpy as np
import h5py

from utils.mri_utils import *
from utils.common_utils import *

class fastMRIData_DIP_multicoil(Dataset):
    def __init__(self, args):

        files = os.listdir(args.folder_path)
        #files.sort()     
        self.samples = []
        # get only the central slice from each subject/file
        for i in range(2): #(len(files)):
            #print(files[i])
            try:
              f = h5py.File(args.folder_path + files[i], 'r')
            except:
              print(f"Error: failed to read {files[i]}. Skipped!")
              continue
            slicenu = f["kspace"].shape[0] // 2 
            print(("Reconstruct the %d slice") % (slicenu))
            slice_ksp = f["kspace"][slicenu]
            print("slice_ksp: ", slice_ksp.shape)
            kdata = np.stack((slice_ksp.real, slice_ksp.imag), axis=-1)
          #  csm = f["csm"][:] if 'csm' in f.keys() else torch.ones([slice_ksp.shape[0]*2] + list(slice_ksp.shape[-2:]))
            if args.use_csm:
               csmfile = f"{args.csm_folder_path}/csm_{files[i]}"
               cf = h5py.File(csmfile, 'r')
               cr, ci = cf[f"csm{slicenu}_r"][:], cf[f"csm{slicenu}_i"][:]
               csm = np.stack((cr, ci), axis=-1) # (coils, x, y, 2)
               print(f"loaded csm file {csmfile}: {csm.shape}")
            else:
               csm = torch.ones([slice_ksp.shape[0]*2] + list(slice_ksp.shape[-2:]))

            orig = f["reconstruction_rss"][slicenu]

            slice_ksp_torchtensor = torch.from_numpy(kdata)
            masked_kspace, mask, mask1d, mask2d = get_mask(slice_ksp_torchtensor, slice_ksp, factor=args.ac_factor, cent=args.center_frac, low_freq_only_mask=args.low_freq_only_mask, inverse_mask=args.inverse_mask, seed=args.seed)
            print(f"masked_kspace: {masked_kspace.shape}")
     
            undersampled_recon, input_images = self.simple_recon(masked_kspace)

            self.samples.append({
                'slice_ksp': masked_kspace,
                'orig': orig,
                'csm' : csm,
                'slice_ksp_torchtensor': slice_ksp_torchtensor,
                'undersampled_recon': undersampled_recon,
                'input_images': input_images,
                'mask': mask,
                'mask1d': mask1d,
                'mask2d': mask2d,
                'filename': files[i] 
            })

    def simple_recon(self, kspace):
        multi_imgs = ifft2(kspace)
        imgs = []
        for img in multi_imgs.detach().cpu():
            imgs += [ img[:,:,0].numpy() , img[:,:,1].numpy() ]
        chn_imgs = channels2imgs(np.array(imgs))
        combined = root_sum_of_squares(torch.from_numpy(chn_imgs))
        cropped = crop_center(combined, 320,320)
        return cropped, np.array(imgs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



class fastMRIData_pretrain_multicoil(Dataset):
    """
    Patrain on some subjects (all slices) and validation on others.
    """
    def __init__(self, args, net):

        self.args = args
        files = os.listdir(args.folder_path)
     
        self.samples = []
        # get all the slices from each subject/file
        for i in range(len(files)):
            try:
              f = h5py.File(args.folder_path + files[i], 'r')
            except:
              print(f"Error: {files[i]} cannot be read. Skipped!")
              continue
            
            vol = f["kspace"][:]
            print(f"{files[i]}: {vol.shape}")
            
            for slicenu in range(len(vol)):
                slice_ksp = vol[slicenu]
                kdata = np.stack((slice_ksp.real, slice_ksp.imag), axis=-1)
                slice_ksp_torchtensor = torch.from_numpy(kdata)
                
                masked_kspace, mask, mask1d, mask2d = get_mask(slice_ksp_torchtensor, slice_ksp, factor=args.ac_factor, cent=args.center_frac, low_freq_only_mask=args.low_freq_only_mask, inverse_mask=args.inverse_mask, seed=args.seed)
                
#                save = h5py.File(f'../data/fastMRI/mask2d_inverse_{args.inverse_mask}_lowFreqOnly_{args.low_freq_only_mask}_{args.ac_factor}x_{args.center_frac}_seed{args.seed}.h5', 'w')
#                save.create_dataset('mask', data=mask2d)
#                save.close()
             
                undersampled_recon, self.input_images = self.simple_recon(masked_kspace)
#                print(f"masked_kspace: {masked_kspace.is_cuda}")
                orig = f["reconstruction_rss"][slicenu] # ground truth
                
                csm = f["csm"][:] if 'csm' in f.keys() else torch.ones([slice_ksp.shape[0]*2] + list(slice_ksp.shape[-2:]))
                
                net_input, scaling_factor = self.get_net_input(masked_kspace, net)

                self.samples.append({
                'slice_ksp': masked_kspace,
                'orig': orig,
                'csm' : csm,
                'slice_ksp_torchtensor': slice_ksp_torchtensor,
                'undersampled_recon': undersampled_recon,
                'net_input': net_input,
                'scaling_factor': scaling_factor,
                'mask': mask,
                'mask1d': mask1d,
                'mask2d': mask2d,
                'filename': files[i]
                })

    def simple_recon(self, kspace):
        multi_imgs = ifft2(kspace)
        imgs = []
        for img in multi_imgs.detach().cpu():
            imgs += [ img[:,:,0].numpy() , img[:,:,1].numpy() ]
        chn_imgs = channels2imgs(np.array(imgs))
        combined = root_sum_of_squares(torch.from_numpy(chn_imgs))
        cropped = crop_center(combined, 320,320)
        return cropped, np.array(imgs)
        
    def get_net_input(self, kspace, net):
        if self.args.use_image_input:
            return self.input_images, 1
        else:
            freq_dict = {
            'method': 'log',
            'cosine_only': False,
            'n_freqs': self.args.n_freqs,
            'base': self.args.base,
        }
            ni, args = get_noise(self.args.input_dim, self.args.noise_method, self.args.in_size, self.args.noise_type, freq_dict=freq_dict,
                                 args=self.args)
            args.input_dim = ni.shape[1]
            # scaling factor
            scaling_factor = get_scale_factor(net, ni, kspace)
            return ni, scaling_factor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class fastMRIData_pretrain_singlecoil(Dataset):
    """
    Patrain on some subjects (all slices) and validation on others.
    """
    def __init__(self, args, net):

        self.args = args
        files = os.listdir(args.folder_path)
     
        self.samples = []
        # get all the slices from each subject/file
        for i in range(len(files)):
            try:
              f = h5py.File(args.folder_path + files[i], 'r')
            except:
              print(f"Error: {files[i]} cannot be read. Skipped!")
              continue
            
            vol = f["kspace"][:]
            print(f"{files[i]}: {vol.shape}")
            
            for slicenu in range(10,len(vol)-5):
                slice_ksp = vol[slicenu, np.newaxis,...]
                kdata = np.stack((slice_ksp.real, slice_ksp.imag), axis=-1)
                slice_ksp_torchtensor = torch.from_numpy(kdata)
                
                masked_kspace, mask, mask1d, mask2d = get_mask(slice_ksp_torchtensor, slice_ksp, factor=args.ac_factor, cent=args.center_frac, low_freq_only_mask=args.low_freq_only_mask, inverse_mask=args.inverse_mask, seed=args.seed)
             
                undersampled_recon, self.input_images = self.simple_recon(masked_kspace[0])

                orig = f["reconstruction_rss"][slicenu] # ground truth
                
                net_input, scaling_factor = self.get_net_input(masked_kspace, net)

                self.samples.append({
                'slice_ksp': masked_kspace,
                'orig': orig,
                'slice_ksp_torchtensor': slice_ksp_torchtensor,
                'undersampled_recon': undersampled_recon,
                'net_input': net_input,
                'scaling_factor': scaling_factor,
                'mask': mask,
                'mask1d': mask1d,
                'mask2d': mask2d,
                'filename': files[i]
                })

    def simple_recon(self, kspace):
        img = ifft2(kspace).data.cpu().numpy()
        imgc = img[...,0] + 1j*img[...,1]
        img = np.abs(imgc)
        cropped = crop_center(img, 320,320)
        return cropped, img
        
    def get_net_input(self, kspace, net):
        if self.args.use_image_input:
            return self.input_images, 1
        else:
            freq_dict = {
            'method': 'log',
            'cosine_only': False,
            'n_freqs': self.args.n_freqs,
            'base': self.args.base,
        }
            ni, args = get_noise(self.args.input_dim, self.args.noise_method, self.args.in_size, self.args.noise_type, freq_dict=freq_dict,
                                 args=self.args)
            args.input_dim = ni.shape[1]
            # scaling factor
            scaling_factor = get_scale_factor(net, ni, kspace)
            return ni, scaling_factor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]






