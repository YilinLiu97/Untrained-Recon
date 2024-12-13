import torch
from torch.utils.data import Dataset

import os
import numpy as np
import h5py
import sys
sys.path.append('./')
from utils.mrf_utils import *

"""
Using inputs of SCQ --> T1, T2
"""
import scipy.io as sio
class mrfData_imMRF_3D_sagittal(Dataset):
    def __init__(self, args):

        self.samples = []

        f = h5py.File(os.path.join(args.folder_path, 'imMRF_GRAPP2_PF_quarterpoints_noSVD.mat'), 'r')
     
        data = f['imMRF_all'][:args.mrf_time_pts,args.mrf_slice_start:args.mrf_slice_end+1] # (192, 3, 256, 256)
        d_real, d_imag = np.flip(data['real'][:], axis=(2,3)), np.flip(data['imag'][:], axis=(2,3))
      
      #  slice_im = np.stack((d_real, d_imag), axis=-1).copy() # (nt, 3, 256, 256, 2)
        slice_im = []

        for n, (dr, di) in enumerate(zip(d_real.transpose(1,0,2,3), d_imag.transpose(1,0,2,3))):
            slice_im += [np.stack((dr, di), axis=-1)]
        
        slice_im = np.array(slice_im) # (3,192,256,256,2)
        print(f"{slice_im.shape}")
        slice_im_torchtensor = torch.from_numpy(slice_im)


        mf = h5py.File(os.path.join(args.gt_mask_path, 'mask3D_tight_100th_v2.h5'), 'r')
        tmask = mf['mask'][:].copy() #np.fliplr(np.flipud(mf['mask_tight'][:])).copy() # (256, 256)
        
        tf = h5py.File(os.path.join(args.folder_path, 'patternmatching_all_v2.h5'), 'r') 
        t1map = tf['t1'][...,args.mrf_slice_start:args.mrf_slice_end+1] #.transpose(2,0,1)
        t2map = tf['t2'][...,args.mrf_slice_start:args.mrf_slice_end+1] #.transpose(2,0,1) # (slices,256,256)

        '''
        mf = h5py.File(os.path.join(args.folder_path, 'm0_2304pnts.h5'), 'r')
        mr = np.fliplr(np.flipud(mf['real'][:]))
        mi = np.fliplr(np.flipud(mf['imag'][:]))
        m0map = np.stack((mr, mi), -1).copy() # (256, 256, 2)
        mc = mr + 1j * mi        
        '''
        print(f"Reconstruct: {slice_im.shape}")

        save = h5py.File('data_check.h5', 'w')
        save.create_dataset('mask', data=tmask)
        save.create_dataset('slice', data=slice_im[0,...])
        save.close()


        self.samples.append({
                'slice_ksp': slice_im * tmask[np.newaxis,np.newaxis,...,np.newaxis] if args.use_tissue_mask else slice_im,
                'kmask': np.ones((slice_im_torchtensor.shape)),
                'tmask': tmask,
                'm0map': np.ones((tmask.shape)),
                't1map': t1map,
                't2map': t2map,
                'cmap': np.ones((slice_im.shape))[...,np.newaxis],
                'filename': 'subj53_1152pts'
        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


"""
Using inputs of SCQ --> T1, T2
"""
import scipy.io as sio
class mrfData_imMRF_3D_coronal(Dataset):
    def __init__(self, args):

        self.samples = []

        f = h5py.File(os.path.join(args.folder_path, 'imMRF_GRAPP2_PF_quarterpoints_noSVD.mat'), 'r')
     
        data = f['imMRF_all'][:args.mrf_time_pts,:,args.mrf_slice_start:args.mrf_slice_end+1,:].transpose(0,2,1,3) # (192, 176, 3, 256) --> (192,3,176,256)
        d_real, d_imag = np.flip(np.rot90(data['real'][:], axes=(2,3)), axis=2), np.flip(np.rot90(data['imag'][:], axes=(2,3)), axis=2)
      
      #  slice_im = np.stack((d_real, d_imag), axis=-1).copy() # (nt, 3, 256, 256, 2)
        slice_im = []
        for n, (dr, di) in enumerate(zip(d_real.transpose(1,0,2,3), d_imag.transpose(1,0,2,3))):
            slice_im += [np.stack((dr, di), axis=-1)]
        
        slice_im = np.array(slice_im) # (3,192,256,256,2)
        slice_im_torchtensor = torch.from_numpy(slice_im)


        mf = h5py.File(os.path.join(args.gt_mask_path, 'coronal_100th_mask.h5'), 'r')
        tmask = np.flip(np.rot90(mf['mask'][:]), axis=1).copy() #np.fliplr(np.flipud(mf['mask_tight'][:])).copy() # (256, 256)
        
        tf = h5py.File(os.path.join(args.folder_path, 'patternmatching_all_v2.h5'), 'r')
        t1map = tf['t1'][args.mrf_slice_start:args.mrf_slice_end+1,:,:].copy() #.transpose(2,0,1)
        t2map = tf['t2'][args.mrf_slice_start:args.mrf_slice_end+1,:,:].copy() #.transpose(2,0,1) # (slices,256,256)

        '''
        mf = h5py.File(os.path.join(args.folder_path, 'm0_2304pnts.h5'), 'r')
        mr = np.fliplr(np.flipud(mf['real'][:]))
        mi = np.fliplr(np.flipud(mf['imag'][:]))
        m0map = np.stack((mr, mi), -1).copy() # (256, 256, 2)
        mc = mr + 1j * mi
        '''
        print(f"Reconstruct: {slice_im.shape} | tmaps: {t1map.shape}")

        save = h5py.File('data_check.h5', 'w')
        save.create_dataset('mask', data=tmask)
        save.create_dataset('slice', data=slice_im[0,...])
        save.create_dataset('t1map', data=t1map)
        save.close()


        self.samples.append({
                'slice_ksp': slice_im * tmask[np.newaxis,np.newaxis,...,np.newaxis] if args.use_tissue_mask else slice_im,
                'kmask': np.ones((slice_im_torchtensor.shape)),
                'tmask': tmask,
                'm0map': np.ones((tmask.shape)),
                't1map': t1map,
                't2map': t2map,
                'cmap': np.ones((slice_im.shape))[...,np.newaxis],
                'filename': 'subj53_1152pts'
        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


"""
Using inputs of SCQ --> T1, T2
"""
import scipy.io as sio
class mrfData_imMRF_3D_axial(Dataset):
    def __init__(self, args):

        self.samples = []

        f = h5py.File(os.path.join(args.folder_path, 'imMRF_GRAPP2_PF_quarterpoints_noSVD.mat'), 'r')
     
        data = f['imMRF_all'][:args.mrf_time_pts,:,:,args.mrf_slice_start:args.mrf_slice_end+1].transpose(0,3,1,2) # (192, 176, 256, 3) --> (192,3,176,256)
        d_real, d_imag = np.rot90(data['real'][:], axes=(2,3)), np.rot90(data['imag'][:], axes=(2,3))
      
      #  slice_im = np.stack((d_real, d_imag), axis=-1).copy() # (nt, 3, 256, 256, 2)
        slice_im = []
        for n, (dr, di) in enumerate(zip(d_real.transpose(1,0,2,3), d_imag.transpose(1,0,2,3))):
            slice_im += [np.stack((dr, di), axis=-1)]
        
        slice_im = np.array(slice_im) # (slices,192,256,256,2) 
        print(f"{slice_im.shape}")
        slice_im_torchtensor = torch.from_numpy(slice_im)


        mf = h5py.File(os.path.join(args.gt_mask_path, 'axial_100th_mask.h5'), 'r')
        tmask = np.rot90(mf['mask'][:]).copy() # (256, 256)
        
        tf = h5py.File(os.path.join(args.folder_path, 'patternmatching_all_v2.h5'), 'r') # (256, 256, 176)
        t1map = np.fliplr(np.flipud(tf['t1'][:,args.mrf_slice_start:args.mrf_slice_end+1,:].transpose(1,0,2))).copy()
        t2map = np.fliplr(np.flipud(tf['t2'][:,args.mrf_slice_start:args.mrf_slice_end+1,:].transpose(1,0,2))).copy() # (slices,256,256)


        '''
        mf = h5py.File(os.path.join(args.folder_path, 'm0_2304pnts.h5'), 'r')
        mr = np.fliplr(np.flipud(mf['real'][:]))
        mi = np.fliplr(np.flipud(mf['imag'][:]))
        m0map = np.stack((mr, mi), -1).copy() # (256, 256, 2)
        mc = mr + 1j * mi
        '''
        print(f"Reconstruct: {slice_im.shape} | t1map: {t1map.shape} | t2map: {t2map.shape}")

        self.samples.append({
                'slice_ksp': slice_im * tmask[np.newaxis,np.newaxis,...,np.newaxis] if args.use_tissue_mask else slice_im,
                'kmask': np.ones((slice_im_torchtensor.shape)),
                'tmask': tmask,
                'm0map': np.ones((tmask.shape)),
                't1map': t1map,
                't2map': t2map,
                'cmap': np.ones((slice_im.shape))[...,np.newaxis],
                'filename': 'subj53_1152pts'
        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
