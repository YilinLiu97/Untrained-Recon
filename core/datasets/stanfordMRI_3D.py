import torch
from torch.utils.data import Dataset

import os
import numpy as np
import h5py

from utils.mri_utils import *

class stanfordMRI3D_multicoil(Dataset):
    def __init__(self, args):

        files = os.listdir(args.folder_path)
     
        self.samples = []
        # get only the central slice from each subject/file
        for i in range(len(files)):
            print(files[i])
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

            if args.use_csm:
               csmfile = f"{args.csm_folder_path}/csm_{files[i]}"
               cf = h5py.File(csmfile, 'r')
               cr, ci = cf[f"csm{slicenu}_r"][:], cf[f"csm{slicenu}_i"][:]
               csm = np.stack((cr, ci), axis=-1) # (coils, x, y, 2)
               print(f"loaded csm file {csmfile}: {csm.shape}")
            else:
               csm = torch.ones([slice_ksp.shape[0]*2] + list(slice_ksp.shape[-2:]))

            orig = f["reconstruction_rss"][slicenu]
            target_size = orig.shape

            slice_ksp_torchtensor = torch.from_numpy(kdata)
            masked_kspace, mask, mask1d, mask2d = get_mask(slice_ksp_torchtensor, slice_ksp, factor=args.ac_factor, cent=args.center_frac, low_freq_only_mask=args.low_freq_only_mask, inverse_mask=args.inverse_mask, seed=args.seed)
            print(f"masked_kspace: {masked_kspace.shape}")
     
            undersampled_recon, input_images = self.simple_recon(masked_kspace)

            save=h5py.File(f'/shenlab/lab_stor/yilinliu/SelfRecon/core/datasets/check_stanford/{files[i]}', 'w')
            save.create_dataset('data', data=undersampled_recon)
            save.create_dataset('orig', data=orig)
            save.close()


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

    def simple_recon(self, kspace, target_size=(320,320)):
        multi_imgs = ifft2(kspace)
        imgs = []
        for img in multi_imgs.detach().cpu():
            imgs += [ img[:,:,0].numpy() , img[:,:,1].numpy() ]
        chn_imgs = channels2imgs(np.array(imgs))
        combined = root_sum_of_squares(torch.from_numpy(chn_imgs))
        cropped = crop_center(combined, target_size[0], target_size[1])
        return cropped, np.array(imgs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



