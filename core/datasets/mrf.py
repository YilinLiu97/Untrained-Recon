import torch
from torch.utils.data import Dataset

import os
import numpy as np
import h5py
import sys
sys.path.append('./')
from utils.mrf_utils import *


"""
Real data
"""
class mrfData_coil_real(Dataset):
    def __init__(self, args):

        self.args = args

        files = os.listdir(args.folder_path)
        self.samples = []
        
        # get single-coil data
        f = h5py.File(os.path.join('/mnt/yapdata/data/yilinliu/datasets/MRF-DIP', 'kspace_multicoil_1152.h5'), 'r') # (nt, N, N, nc)
        dr, di = np.flip(f['real'][:args.mrf_time_pts], axis=(1,2)), np.flip(f['imag'][:args.mrf_time_pts], axis=(1,2))
#        f = h5py.File(os.path.join('/mnt/yapdata/data/yilinliu/datasets/MRF-DIP/144_8coils', 'kspace_multicoil_percount.h5'), 'r') # (nt, N, N, nc)
#        dr, di = np.flip(f['r'][:args.mrf_time_pts], axis=(1,2)), np.flip(f['i'][:args.mrf_time_pts], axis=(1,2))
        d_real, d_imag = dr[...,args.nth_coil], di[...,args.nth_coil]
        

        slice_ksp = np.stack((d_real, d_imag), axis=-1).copy() # (nt, 256, 256, nc, 2)
        slice_ksp_torchtensor = torch.from_numpy(slice_ksp)
        
        # create and apply kmask

        fm = h5py.File(os.path.join('/mnt/yapdata/data/yilinliu/datasets/MRF-DIP', 'actual_kmask.h5'), 'r')
        kmask = np.flip(fm['mask'][:args.mrf_time_pts], axis=(1,2))
        kmask = (kmask != 0.)
        kmask = torch.from_numpy(kmask.copy()).unsqueeze(-1)
        print(f"kmask: {kmask.shape}, {torch.unique(kmask)}")

        masked_kspace = slice_ksp_torchtensor * kmask
        print(f"masked: {masked_kspace.sum()}")

        # visualize
        self.visualize_imMRF(slice_ksp_torchtensor, masked_kspace, output_ksp=True)        

        '''
        save = h5py.File('/shenlab/lab_stor/yilinliu/SelfRecon/core/mrf-related/mask_actual.h5', 'w')
        save.create_dataset('mask', data=kmask.data.cpu().numpy())
        save.close()
        '''

        # tissue mask
        tmf = h5py.File(os.path.join(args.gt_mask_path, 'mask_tight.mat'), 'r')  #h5py.File('/mnt/yapdata/data/yilinliu/datasets/MRF-DIP/equal-spaced/tissue_mask.h5', 'r')
        tmask = np.fliplr(np.flipud(tmf['mask'][:])).copy()  #tmf['mask'][:] # (256, 256)
        # ground truth tissue maps
        tf = h5py.File('/mnt/yapdata/data/yilinliu/datasets/MRF-DIP/patternmatching_percoil.h5', 'r') # (32, 256, 256)
        t1map = np.fliplr(np.flipud(tf['t1'][args.nth_coil])).copy()
        t2map = np.fliplr(np.flipud(tf['t2'][args.nth_coil])).copy()
        
        
        # m0map
        mr = np.fliplr(np.flipud(tf['m0']['real'][args.nth_coil]))
        mi = np.fliplr(np.flipud(tf['m0']['imag'][args.nth_coil]))
        
        m0map = np.stack((mr, mi), -1).copy() # (256, 256, 2)

        
        print(f"Reconstruct: {slice_ksp.shape} | kmask: {kmask.shape} ")
                 
        self.samples.append({
                'slice_ksp': masked_kspace * args.manual_scaling if not args.dictionary_matching else mrf_ifft2(masked_kspace, None) * args.manual_scaling,
                'slice_ksp_torchtensor': slice_ksp_torchtensor,
                'kmask': kmask,
                'tmask': tmask,
                'm0map': m0map * args.manual_scaling,
                't1map': t1map,
                't2map': t2map,
                'cmap': np.ones((slice_ksp.shape))[...,np.newaxis],
                'filename': 'Subj53'
        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def visualize_imMRF(self, kdata, masked, output_ksp=False, cmap=None):
        if not torch.is_tensor(kdata):
           kdata = torch.from_numpy(kdata)
        im =  mrf_ifft2(kdata, cmap).data.cpu().numpy()
        masked_im = mrf_ifft2(masked, cmap).data.cpu().numpy()

        if output_ksp:
           savek = h5py.File(f"{self.args.save_path}/masked_kspace.h5", 'w')
           savek.create_dataset('kdata', data=masked.data.cpu().numpy())
           savek.close()

        save = h5py.File(f"{self.args.save_path}/imMRF.h5", 'w')
        save.create_dataset('fps', data=im)
        save.close()

        savem = h5py.File(f"{self.args.save_path}/imMRF_masked.h5", 'w')
        savem.create_dataset('fps', data=masked_im)
        savem.close()

"""
Synthetic per-coil data
"""
class mrfData_syn2(Dataset):
    def __init__(self, args):

        self.args = args

        files = os.listdir(args.folder_path)
        self.samples = []

        # get single-coil data
        f = h5py.File(os.path.join('/shenlab/lab_stor/yilinliu/make_data/percoil', 'full_ksp_from_dict_False.h5'), 'r') # (nt, N, N, nc)
        d_real, d_imag = f['r'][:args.mrf_time_pts], f['i'][:args.mrf_time_pts]
        dc = d_real + 1j*d_imag


        slice_ksp = np.stack((d_real, d_imag), axis=-1).copy() # (nt, 256, 256, nc, 2)
        slice_ksp_torchtensor = torch.from_numpy(slice_ksp)

        # create and apply kmask
        """
        fm = h5py.File(os.path.join(args.folder_path, 'mrf_kmask_144pts.h5'), 'r')
        km = fm['mask'][:] # (144, 256, 256)
         
        kmask = (1 - (km[:48] == 0)) # (nt, 256, 256)
        print(f"kmask (prev): {kmask.shape}")
             
        kmask = torch.from_numpy(kmask).type(torch.FloatTensor).unsqueeze(-1) # (nt, 256, 256, 1)

        rep = int(args.mrf_time_pts / 48)
        kmask = torch.tile(kmask, (rep, 1, 1, 1))
        print(f"kmask (after): {kmask.shape}")
        """

        fm = h5py.File(os.path.join('/mnt/yapdata/data/yilinliu/datasets/MRF-DIP', 'actual_kmask.h5'), 'r')
        kmask = np.flip(fm['mask'][:args.mrf_time_pts], axis=(1,2))
        kmask = (kmask != 0.)
        kmask = torch.from_numpy(kmask.copy()).unsqueeze(-1)
        print(f"kmask: {kmask.shape}, {torch.unique(kmask)}")

        masked_kspace = slice_ksp_torchtensor #* kmask
        print(f"masked: {masked_kspace.sum()}")

        # visualize
        self.visualize_imMRF(slice_ksp_torchtensor, masked_kspace)

        save = h5py.File('/shenlab/lab_stor/yilinliu/SelfRecon/core/mrf-related/mask_actual.h5', 'w')
        save.create_dataset('mask', data=kmask.data.cpu().numpy())
        save.close()

        # tissue mask
        mf = h5py.File(os.path.join(args.gt_mask_path, 'mask.mat'), 'r')
        tmask = np.fliplr(np.flipud(mf['mask'][:])).copy() # (256, 256)
        # ground truth tissue maps
        tf = h5py.File('/mnt/yapdata/data/yilinliu/datasets/MRF-DIP/patternmatching_percoil.h5', 'r') # (32, 256, 256)    #h5py.File(os.path.join(args.gt_mask_path, 'patternmatching_2304.mat'), 'r')
        t1map = np.fliplr(np.flipud(tf['t1'][args.nth_coil])).copy()
        t2map = np.fliplr(np.flipud(tf['t2'][args.nth_coil])).copy()

        '''
        # coil map
        #        cf = h5py.File(os.path.join(args.folder_path, 'cmap_32coils.h5'), 'r')
        cf = h5py.File(os.path.join(args.folder_path, 'cmap_32coils.h5'), 'r')
        cr, ci = np.flip(cf['real'][...,4], axis=(0,1)), np.flip(cf['imag'][...,4], axis=(0,1))
        cmap = np.stack((cr, ci), -1).copy() # when single: (256, 256, 2). (256, 256, nc, 2)
        cmc = cr + 1j * ci # (256, 256, nc)

        cf = h5py.File(os.path.join(args.folder_path, 'singlecoil_map.h5'), 'w')
        cf.create_dataset('r', data=cmc.real)
        cf.create_dataset('i', data=cmc.imag)
        cf.close()
        '''

        # m0map
        #mf = h5py.File(os.path.join(args.folder_path, 'm0_2304pts.h5'), 'r')
        mr = np.fliplr(np.flipud(tf['m0']['real'][args.nth_coil]))
        mi = np.fliplr(np.flipud(tf['m0']['imag'][args.nth_coil]))

        print(f"m0_coil: {np.unique(mr)}, {np.unique(mi)}")
        m0map = np.stack((mr, mi), -1).copy() # (256, 256, 2)


        print(f"Reconstruct: {slice_ksp.shape} | kmask: {kmask.shape} ")
                 
        self.samples.append({
                'slice_ksp': masked_kspace * 1000000000000,
                'slice_ksp_torchtensor': slice_ksp_torchtensor,
                'kmask': kmask,
                'tmask': tmask,
                'm0map': m0map * 1000000000000,
                't1map': t1map,
                't2map': t2map,
                'cmap': np.ones((slice_ksp.shape))[...,np.newaxis],
                'filename': 'Subj53'
        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def visualize_imMRF(self, kdata, masked, cmap=None):
        if not torch.is_tensor(kdata):
           kdata = torch.from_numpy(kdata)
        im =  mrf_ifft2(kdata, cmap).data.cpu().numpy()
        masked_im = mrf_ifft2(masked, cmap).data.cpu().numpy()

        save = h5py.File(f"{self.args.save_path}/imMRF.h5", 'w')
        save.create_dataset('fps', data=im)
        save.close()

        savem = h5py.File(f"{self.args.save_path}/imMRF_masked.h5", 'w')
        savem.create_dataset('fps', data=masked_im)
        savem.close()
        

"""
Synthetic per-coil data in kspace
"""
class mrfData_coil_syn(Dataset):
    def __init__(self, args):

        files = os.listdir(args.folder_path)
        self.samples = []

        self.args = args

        f = h5py.File(os.path.join('/shenlab/lab_stor/yilinliu/make_data/percoil2', f'full_ksp_from_dict_False_{args.nth_coil+1}.h5'), 'r')
#        f = h5py.File(os.path.join('/shenlab/lab_stor/yilinliu/make_data/all_percoil_m0', 'full_ksp_from_dict_False.h5'), 'r')
#        f = h5py.File(os.path.join('/shenlab/lab_stor/yilinliu/make_data/all_coil_m0_scaled_by_5thcoil', 'full_ksp_from_dict_False.h5'), 'r')
        d_real, d_imag = f['r'][:args.mrf_time_pts], f['i'][:args.mrf_time_pts]
        dc = d_real + 1j * d_imag

        slice_ksp = np.stack((d_real, d_imag), axis=-1).copy() # (nt, 256, 256, nc, 2)
        slice_ksp_torchtensor = torch.from_numpy(slice_ksp)
        print(f"slice_ksp_torchtensor: {slice_ksp_torchtensor.shape}")
     
        fm = h5py.File(os.path.join('/mnt/yapdata/data/yilinliu/datasets/MRF-DIP', 'actual_kmask.h5'), 'r')
        kmask = np.flip(fm['mask'][:args.mrf_time_pts], axis=(1,2))
        kmask = (kmask != 0.)
        kmask = torch.from_numpy(kmask.copy()).unsqueeze(-1)
        print(f"kmask: {kmask.shape}, {torch.unique(kmask)}")

        masked_kspace = slice_ksp_torchtensor * kmask
        print(f"masked: {masked_kspace.sum()}")

        # visualize
#        self.visualize_imMRF(slice_ksp_torchtensor, masked_kspace)

        save = h5py.File('/shenlab/lab_stor/yilinliu/SelfRecon/core/mrf-related/mask_actual.h5', 'w')
        save.create_dataset('mask', data=kmask.data.cpu().numpy())
        save.close()

        '''
        masked_fps = mrf_ifft2(masked_kspace, None)
        save = h5py.File('/shenlab/lab_stor/yilinliu/SelfRecon/core/mrf-related/masked_dict_fps.h5', 'w')
        save.create_dataset('r', data=masked_fps[...,0].data.cpu().numpy())
        save.create_dataset('i', data=masked_fps[...,1].data.cpu().numpy())
        save.close()
        '''
        # tissue mask
        tmf = h5py.File('/mnt/yapdata/data/yilinliu/datasets/MRF-DIP/equal-spaced/tissue_mask.h5', 'r')
        tmask = np.fliplr(np.flipud(tmf['mask'][:])).copy() # (256, 256)
        # ground truth tissue maps
        tf = h5py.File('/mnt/yapdata/data/yilinliu/datasets/MRF-DIP/patternmatching_percoil.h5', 'r') # (32, 256, 256)    #h5py.File(os.path.join(args.gt_mask_path, 'pattern>
        t1map = np.fliplr(np.flipud(tf['t1'][args.nth_coil])).copy()
        t2map = np.fliplr(np.flipud(tf['t2'][args.nth_coil])).copy()

        # m0map
        #mf = h5py.File(os.path.join(args.folder_path, 'm0_2304pts.h5'), 'r')
        mr = np.fliplr(np.flipud(tf['m0']['real'][args.nth_coil]))
        mi = np.fliplr(np.flipud(tf['m0']['imag'][args.nth_coil]))
        
        print(f"m0_coil: {np.unique(mr)}, {np.unique(mi)}")
        m0map = np.stack((mr, mi), -1).copy() # (256, 256, 2)

 
        savem = h5py.File('/shenlab/lab_stor/yilinliu/SelfRecon/core/mrf-related/m0_singlecoil.h5','w')
        savem.create_dataset('m0', data=m0map)
        savem.close()

        save = h5py.File('/shenlab/lab_stor/yilinliu/SelfRecon/core/mrf-related/m0map.h5', 'w')
        save.create_dataset('r', data=mr)
        save.create_dataset('i', data=mi)
        save.close()
        
        print(f"Reconstruct: {slice_ksp.shape} | kmask: {kmask.shape} ")
                 
        self.samples.append({
                'slice_ksp': masked_kspace * args.manual_scaling if not args.dictionary_matching else mrf_ifft2(masked_kspace, None) * args.manual_scaling,
                'slice_ksp_torchtensor': slice_ksp_torchtensor,
                'kmask': kmask,
                'tmask': tmask,
                'm0map': m0map * args.manual_scaling,
                't1map': t1map,
                't2map': t2map,
                'cmap': np.ones((slice_ksp.shape))[...,np.newaxis],
                'filename': 'Subj53'
        })


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def visualize_imMRF(self, kdata, masked, cmap=None):
        if not torch.is_tensor(kdata):
           kdata = torch.from_numpy(kdata)
        im =  mrf_ifft2(kdata, cmap).data.cpu().numpy()
        masked_im = mrf_ifft2(masked, cmap).data.cpu().numpy()

        save = h5py.File(f"{self.args.save_path}/imMRF.h5", 'w')
        save.create_dataset('fps', data=im)
        save.close()

        savem = h5py.File(f"{self.args.save_path}/imMRF_masked.h5", 'w')
        savem.create_dataset('fps', data=masked_im)
        savem.close()

        
"""
Synthetic data in kspace
"""
class mrfData_full_syn(Dataset):
    def __init__(self, args):

        files = os.listdir(args.folder_path)
        self.samples = []

        self.args = args
        '''       
        fm = h5py.File(os.path.join(args.folder_path, 'kspace_pca_multicoil.h5'), 'r') # (nt, N, N, nc)
        dm_real, dm_imag = np.flip(fm['real'][:], axis=(1,2)), np.flip(fm['imag'][:], axis=(1,2))               
        dmc = dm_real + 1j * dm_imag        
        '''
        fm = h5py.File(os.path.join(args.folder_path, 'mrf_kmask_144pts.h5'), 'r')
        km = fm['mask'][:] # (144, 256, 256)

               

#        f = h5py.File(os.path.join(args.folder_path, 'full_ksp_from_dict_fft2.h5'), 'r') 
#        f = h5py.File(os.path.join('/shenlab/lab_stor/yilinliu/make_data/all', 'full_ksp_from_dict_False.h5'), 'r') # the one worked 
        f = h5py.File(os.path.join('/shenlab/lab_stor/yilinliu/make_data/all_2nd', 'full_ksp_from_dict_False.h5'), 'r') 
    #    f = h5py.File(os.path.join(args.folder_path, 'full_ksp_from_dict_fft2_newfft_with1000.h5'), 'r') # one that produces weird result
#        f = h5py.File(os.path.join(args.folder_path, 'full_ksp_from_dict2.h5'), 'r') # one that produces weird result
 #       f = h5py.File(os.path.join('/shenlab/lab_stor/yilinliu/make_data/generated', 'full_ksp_from_dict_True.h5'), 'r')
        d_real, d_imag = f['r'][:args.mrf_time_pts], f['i'][:args.mrf_time_pts]               
        dc = d_real + 1j * d_imag

        slice_ksp = np.stack((d_real, d_imag), axis=-1).copy() # (nt, 256, 256, nc, 2)
        slice_ksp_torchtensor = torch.from_numpy(slice_ksp)
        print(f"slice_ksp_torchtensor: {slice_ksp_torchtensor.shape}")
     
        '''
        kmask = (1 - (km[:48] == 0)) # (nt, 256, 256)
        print(f"kmask (prev): {kmask.shape}")
        kmask = torch.from_numpy(kmask).type(torch.FloatTensor).unsqueeze(-1) # (nt, 256, 256, 1)

        rep = int(args.mrf_time_pts / 48)
        kmask = torch.tile(kmask, (rep, 1, 1, 1))
        print(f"kmask (after): {kmask.shape}")
        '''

        fm = h5py.File(os.path.join('/mnt/yapdata/data/yilinliu/datasets/MRF-DIP', 'actual_kmask.h5'), 'r')
        kmask = np.flip(fm['mask'][:], axis=(1,2))
        kmask = torch.from_numpy((kmask != 0.).copy()).unsqueeze(-1)

        masked_kspace = slice_ksp_torchtensor * kmask
        print(f"masked: {masked_kspace.sum()}")

        # visualize
        self.visualize_imMRF(slice_ksp_torchtensor, masked_kspace) 

        save = h5py.File('/shenlab/lab_stor/yilinliu/SelfRecon/core/mrf-related/mask_48x.h5', 'w')
        save.create_dataset('mask', data=kmask.data.cpu().numpy())
        save.close()

       
        masked_fps = mrf_ifft2(masked_kspace, None)
        save = h5py.File('/shenlab/lab_stor/yilinliu/SelfRecon/core/mrf-related/masked_dict_fps_2.h5', 'w')
        save.create_dataset('r', data=masked_fps[...,0].data.cpu().numpy())
        save.create_dataset('i', data=masked_fps[...,1].data.cpu().numpy())
        save.close()
        

        mf = h5py.File(os.path.join(args.gt_mask_path, 'mask.mat'), 'r')
        tmask = np.fliplr(np.flipud(mf['mask'][:])).copy() # (256, 256)
        
        tf = h5py.File('/mnt/yapdata/data/yilinliu/datasets/MRF-DIP/equal-spaced/patternmatching.mat', 'r') #h5py.File(os.path.join(args.gt_mask_path, 'patternmatching_2304.mat'), 'r')
        t1map = tf['t1big'][:] 
        t2map = tf['t2big'][:] 

        mf = h5py.File(os.path.join(args.folder_path, 'm0_2304pnts.h5'), 'r')
        mr = np.fliplr(np.flipud(mf['real'][:])) 
        mi = np.fliplr(np.flipud(mf['imag'][:]))
        m0map = np.stack((mr, mi), -1).copy() # (256, 256, 2)
        
        cf = h5py.File(os.path.join(args.folder_path, 'cmap_pca.h5'), 'r')
        cr, ci = np.flip(cf['real'][:], axis=(0,1)), np.flip(cf['imag'][:], axis=(0,1))
        cmap = np.stack((cr, ci), -1).copy() # (256, 256, nc, 2)
        
        
        print(f"Reconstruct: {slice_ksp.shape} | kmask: {kmask.shape} ")
        
         
        self.samples.append({
                'slice_ksp': masked_kspace, # * 1000000000,
                'slice_ksp_torchtensor': slice_ksp_torchtensor,
                'kmask': kmask,
                'tmask': tmask,
                'm0map': m0map, # * 1000000000,
                't1map': t1map,
                't2map': t2map,
                'cmap': cmap,
                'filename': 'Subj53_144pts'
        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def visualize_imMRF(self, kdata, masked, cmap=None):
        if not torch.is_tensor(kdata):
           kdata = torch.from_numpy(kdata)
        im =  mrf_ifft2(kdata, cmap).data.cpu().numpy()
        masked_im = mrf_ifft2(masked, cmap).data.cpu().numpy()

        save = h5py.File(f"{self.args.save_path}/imMRF.h5", 'w')
        save.create_dataset('fps', data=im)
        save.close()

        savem = h5py.File(f"{self.args.save_path}/imMRF_masked.h5", 'w')
        savem.create_dataset('fps', data=masked_im)
        savem.close()
        
"""
Synthetic data in image space
"""
class mrfData_syn_image(Dataset):
    def __init__(self, args):

        files = os.listdir(args.folder_path)
        self.samples = []

        self.args = args


        f = h5py.File(os.path.join(args.folder_path, 'full_ksp_from_dict_fft2.h5'), 'r')
        d_real, d_imag = f['r'][:args.mrf_time_pts], f['i'][:args.mrf_time_pts]

        
        slice_ksp = np.stack((d_real, d_imag), axis=-1).copy() # (nt, 256, 256, nc, 2)
        slice_ksp_torchtensor = torch.from_numpy(slice_ksp)
        
        
        fm = h5py.File(os.path.join(args.folder_path, 'mrf_kmask_144pts.h5'), 'r')
        km = fm['mask'][:] # (144, 256, 256)
        kmask = (1 - (km[:48] == 0)) # (nt, 256, 256)
        kmask = torch.from_numpy(kmask).type(torch.FloatTensor).unsqueeze(-1) # (nt, 256, 256, 1)

        rep = int(args.mrf_time_pts / 48)
        kmask = torch.tile(kmask, (rep, 1, 1, 1))

        masked_kspace = slice_ksp_torchtensor #* kmask
        
        slice_im_torchtensor = mrf_ifft2(masked_kspace, None)
        slice_im = slice_im_torchtensor.data.cpu().numpy()

        # visualize
        self.visualize_imMRF(slice_ksp_torchtensor, masked_kspace)
        

        mf = h5py.File(os.path.join(args.gt_mask_path, 'mask.mat'), 'r')
        tmask = np.fliplr(np.flipud(mf['mask'][:])).copy() # (256, 256)
        
        tf = h5py.File('/mnt/yapdata/data/yilinliu/datasets/MRF-DIP/equal-spaced/patternmatching.mat', 'r') #h5py.File(os.path.join(args.gt_mask_path, 'patternmatching_2304.mat'), 'r')
        t1map = tf['t1big'][:]
        t2map = tf['t2big'][:]

        mf = h5py.File(os.path.join(args.folder_path, 'm0_2304pnts.h5'), 'r')
        mr = np.fliplr(np.flipud(mf['real'][:]))
        mi = np.fliplr(np.flipud(mf['imag'][:]))
        m0map = np.stack((mr, mi), -1).copy() # (256, 256, 2)
        
        cf = h5py.File(os.path.join(args.folder_path, 'cmap_pca.h5'), 'r')
        cr, ci = np.flip(cf['real'][:], axis=(0,1)), np.flip(cf['imag'][:], axis=(0,1))
        cmap = np.stack((cr, ci), -1).copy() # (256, 256, nc, 2)
        
        
        print(f"Reconstruct: {slice_ksp.shape} | kmask: {kmask.shape} ")
        
         
        self.samples.append({
                'slice_ksp': slice_im * 10000000,
                'slice_ksp_torchtensor': slice_im_torchtensor,
                'kmask': kmask,
                'tmask': tmask,
                'm0map': m0map * 10000000,
                't1map': t1map,
                't2map': t2map,
                'cmap': cmap,
                'filename': 'Subj53_144pts'
        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def visualize_imMRF(self, kdata, masked, cmap=None):
        if not torch.is_tensor(kdata):
           kdata = torch.from_numpy(kdata)
        im =  mrf_ifft2(kdata, cmap).data.cpu().numpy()
        masked_im = mrf_ifft2(masked, cmap).data.cpu().numpy()

        save = h5py.File(f"{self.args.save_path}/imMRF.h5", 'w')
        save.create_dataset('fps', data=im)
        save.close()

        savem = h5py.File(f"{self.args.save_path}/imMRF_masked.h5", 'w')
        savem.create_dataset('fps', data=masked_im)
        savem.close()
        
        
"""
Using inputs of SCQ --> T1, T2
"""
import scipy.io as sio
class mrfData_imMRF(Dataset):
    def __init__(self, args):

        files = os.listdir(args.folder_path)
        self.samples = []

        f = h5py.File('/shenlab/lab_stor/zhenghan/data/MRF/DataNewDictionary/20180206data/180131_1/53/imMRF.mat', 'r')
        data = f[:args.mrf_time_pts]
        d_real, d_imag = np.flip(data['real'][:], axis=(1,2)), np.flip(data['imag'][:], axis=(1,2))
        dc = d_real + 1j * d_imag
        
        slice_im = np.stack((d_real, d_imag), axis=-1).copy() # (nt, 256, 256, 2)
        slice_im_torchtensor = torch.from_numpy(slice_im)

        #mf = sio.loadmat('/mnt/yapdata/data/yilinliu/datasets/MRF-DIP/equal-spaced/mask_khoi.mat') #h5py.File(os.path.join(args.gt_mask_path, 'mask_tight.mat'), 'r')
        #tmask = np.fliplr((np.rot90(mf['MRFmask'][:]))).copy()  #np.fliplr(np.flipud(mf['MRFmask'][:])).copy() # (256, 256)

        mf = h5py.File(os.path.join(args.gt_mask_path, 'mask_tight.mat'), 'r')
        tmask = np.fliplr(np.flipud(mf['mask'][:])).copy() # (256, 256)
        
        tf = h5py.File('/mnt/yapdata/data/yilinliu/datasets/MRF-DIP/equal-spaced/patternmatching.mat', 'r') #h5py.File(os.path.join(args.gt_mask_path, 'patternmatching_2304.mat'), 'r')
        t1map = tf['t1big'][:]
        t2map = tf['t2big'][:]

        mf = h5py.File(os.path.join(args.folder_path, 'm0_2304pnts.h5'), 'r')
        mr = np.fliplr(np.flipud(mf['real'][:]))
        mi = np.fliplr(np.flipud(mf['imag'][:]))
        m0map = np.stack((mr, mi), -1).copy() # (256, 256, 2)
        mc = mr + 1j * mi        

        print(f"Reconstruct: {slice_im.shape}")

        save = h5py.File('mask_khoi', 'w')
        save.create_dataset('mask', data=tmask)
        save.close()


        '''
        save = h5py.File('/shenlab/lab_stor/yilinliu/SelfRecon/core/mrf-related/imMRF_53.h5', 'w')
        save.create_dataset('real', data=d_real)
        save.create_dataset('imag', data=d_imag)
        save.close()        

        savem = h5py.File('/shenlab/lab_stor/yilinliu/SelfRecon/core/mrf-related/m0.h5', 'w')
        savem.create_dataset('real', data=mr)
        savem.create_dataset('imag', data=mi)
        savem.close()

        md = mc * dc
        save = h5py.File('/shenlab/lab_stor/yilinliu/SelfRecon/core/mrf-related/imMRF_m0.h5','w')
        save.create_dataset('real', data=np.real(md))
        save.create_dataset('imag', data=np.imag(md))
        save.close()
        
        save = h5py.File('/shenlab/lab_stor/yilinliu/SelfRecon/core/mrf-related/eroded_imMRF_target.h5','w')
        masked = slice_im * tmask[np.newaxis,...,np.newaxis]
        save.create_dataset('real', data=masked[...,0])
        save.create_dataset('imag', data=masked[...,1])
        save.close()
        '''
 
        self.samples.append({
                'slice_ksp': slice_im * tmask[np.newaxis,...,np.newaxis] if args.use_tissue_mask else slice_im,
                'kmask': np.ones((slice_im_torchtensor.shape)),
                'tmask': tmask,
                'm0map': m0map,
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
Using kspace of the inputs of SCQ --> T1, T2
"""
class mrfData_imMRF_fft(Dataset):
    def __init__(self, args):

        files = os.listdir(args.folder_path)
        self.samples = []

        self.args = args

        # imMRF
        f = h5py.File('/shenlab/lab_stor/zhenghan/data/MRF/DataNewDictionary/20180206data/180131_1/53/imMRF.mat', 'r')
        data = f['imMRF'][:args.mrf_time_pts]
        d_real, d_imag = np.flip(data['real'], axis=(1,2)), np.flip(data['imag'], axis=(1,2))
        
        slice_im = np.stack((d_real, d_imag), axis=-1).copy() # (nt, 256, 256, nc, 2)
        slice_im_torchtensor = torch.from_numpy(slice_im)
        
        # convert to kspace
        slice_ksp_torchtensor = mrf_fft2(slice_im_torchtensor, None)
        print(f"{slice_ksp_torchtensor.shape}")
        
        # create and apply kmask
        fm = h5py.File(os.path.join('/mnt/yapdata/data/yilinliu/datasets/MRF-DIP/imMRF_kmask.h5'), 'r') #'actual_kmask.h5'), 'r')
        kmask = fm['mask'][:args.mrf_time_pts]  #np.flip(fm['mask'][:args.mrf_time_pts], axis=(1,2))
        kmask = (kmask != 0.)
        kmask = torch.from_numpy(kmask.copy()).unsqueeze(-1)

        masked_kspace = slice_ksp_torchtensor * kmask

        masked_fps = mrf_ifft2(masked_kspace, None)
        save = h5py.File('/shenlab/lab_stor/yilinliu/SelfRecon/core/mrf-related/masked_imMRF_ksp.h5', 'w')
        save.create_dataset('r', data=slice_ksp_torchtensor[...,0].data.cpu().numpy())
        save.create_dataset('i', data=slice_ksp_torchtensor[...,1].data.cpu().numpy())
        save.close()


        # visualize
        self.visualize_imMRF(slice_ksp_torchtensor, masked_kspace)

        # tissue mask
        mf = h5py.File(os.path.join(args.gt_mask_path, 'mask.mat'), 'r')
        tmask = np.fliplr(np.flipud(mf['mask'][:])).copy() # (256, 256)
        
        tf = h5py.File('/mnt/yapdata/data/yilinliu/datasets/MRF-DIP/equal-spaced/patternmatching.mat', 'r')
        t1map = tf['t1big'][:]
        t2map = tf['t2big'][:]

        mf = h5py.File(os.path.join(args.folder_path, 'm0_2304pnts.h5'), 'r')
        mr = np.fliplr(np.flipud(mf['real'][:]))
        mi = np.fliplr(np.flipud(mf['imag'][:]))
        m0map = np.stack((mr, mi), -1).copy() # (256, 256, 2)

        print(f"Reconstruct: {slice_im.shape}")

 
        self.samples.append({
                'slice_ksp': masked_kspace * 100000000000000,
                'slice_ksp_torchtensor': slice_ksp_torchtensor,
                'kmask': kmask,
                'tmask': tmask,
                'm0map': m0map * 100000000000000,
                't1map': t1map,
                't2map': t2map,
                'cmap': np.ones((slice_im.shape))[...,np.newaxis],
                'filename': 'subj53'
        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def visualize_imMRF(self, kdata, masked, cmap=None):
        if not torch.is_tensor(kdata):
           kdata = torch.from_numpy(kdata)
        im =  mrf_ifft2(kdata, cmap).data.cpu().numpy()
        masked_im = mrf_ifft2(masked, cmap).data.cpu().numpy()

        save = h5py.File(f"{self.args.save_path}/imMRF.h5", 'w')
        save.create_dataset('fps', data=im)
        save.close()

        savem = h5py.File(f"{self.args.save_path}/imMRF_masked.h5", 'w')
        savem.create_dataset('fps', data=masked_im)
        savem.close()
