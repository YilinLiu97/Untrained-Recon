import torch
import torch.nn
from torch.autograd import Variable
import sys

from .mri_utils import *
from .mrf_utils import *
from .common_utils import *
from .sr_utils import *

def mri_forwardm(img, mask, downsampler=None, fully_meas=None, csm=None):
    # img has dimension (2*num_slices, x,y)
    # output has dimension (1, num_slices, x, y, 2)
    mask = mask[0]

    s = img.shape
    ns = int(s[1] / 2)  # number of slices
    fimg = Variable(torch.zeros((s[0], ns, s[2], s[3], 2))).type(dtype)
    for i in range(ns):
        fimg[0, i, :, :, 0] = img[0, 2 * i, :, :]
        fimg[0, i, :, :, 1] = img[0, 2 * i + 1, :, :]

    Fimg = mri_fft2(fimg, csm)  # dim: (1,num_coils,x,y,2)
    Himg = Fimg.detach().clone()
    Lgt = fully_meas.detach().clone()
    Hgt = fully_meas.detach().clone()

    for i in range(ns): # unmasked region (low-frequency)
        Fimg[0, i, :, :, 0] *= mask
        Fimg[0, i, :, :, 1] *= mask

        Lgt[0, i, :, :, 0] *= mask
        Lgt[0, i, :, :, 1] *= mask

        Himg[0, i, :, :, 0] *= (1 - mask)
        Himg[0, i, :, :, 1] *= (1 - mask)

        Hgt[0, i, :, :, 0] *= (1 - mask)
        Hgt[0, i, :, :, 1] *= (1 - mask)

    unmasked_img = ifft2(Fimg)
    masked_img = ifft2(Himg)

    unmasked_gt = ifft2(Lgt)
    masked_gt = ifft2(Hgt)
    return Fimg, unmasked_img, masked_img, unmasked_gt, masked_gt

def mri_ksp_sup(img):
    # img has dimension (2*num_slices, x,y)
    # output has dimension (1, num_slices, x, y, 2)

    s = img.shape
    ns = int(s[1] / 2)  # number of slices
    fimg = Variable(torch.zeros((s[0], ns, s[2], s[3], 2))).type(dtype)
    for i in range(ns):
        fimg[0, i, :, :, 0] = img[0, 2 * i, :, :]
        fimg[0, i, :, :, 1] = img[0, 2 * i + 1, :, :]
    Fimg = mri_fft2(fimg)  # dim: (1,num_coils,x,y,2)
    return Fimg

def mrf_forwardm(tmaps, dict, csm, m0map, kmask, tmask=None, normalize=True, learned_scaling=None, return_images_only=False):
    """
    Inputs:
    tmaps: (1, 2, N, N). already coils combined.
    csm: (1, N, N, nc, 2), or None
    kmask: (1, nt, N, N, nc), or None
    tmask: (1, N, N)
    m0map: (1, N, N, 2)
    learned_scaling: (1, N, N, 2)

    Return: (1, nt, N, N, nc, 2)
    """
   
    if tmask is not None:
       tmaps = tmask * tmaps

    # retreive fingerprints from dict
    fps = inverse_DM(tmaps, dict, tmask, normalize=normalize) # (nt, N, N, 2)
  
    if m0map is not None:
       fps, m0map = torch.view_as_complex(fps), torch.view_as_complex(m0map)
       scaled = fps * m0map 
    else:
       print(f"m0 is Not used!!!!!!!")
       scaled = torch.view_as_complex(fps)

    if learned_scaling is not None:
       factor = torch.view_as_complex(learned_scaling.permute(0,2,3,1).contiguous())
       print(f"Learned factor: {factor.shape}")
       scaled = scaled * factor

    scaled = torch.stack((scaled.real, scaled.imag), -1) # (nt, 256, 256, 2)
    if return_images_only:
       print(f"Return the images only!!!!!!!!! Doing dictionary matching.")
       return scaled.unsqueeze(0)    

    kspace_fps = mrf_fft2(scaled, csm) # (1, nt, N, N, nc, 2) or (1, nt, N, N, 2) if no csm
    Fimg = kspace_fps * kmask if kmask is not None else kspace_fps    

    if kmask is not None:
       print(f"kspace_fps: {kspace_fps.shape}, kmask: {kmask.shape}")
       kmask_count = torch.sum(kmask, 1) # (1, N, N)
       kmask_count[kmask_count==0] = 1
       print(f"kmask_count: {kmask_count.shape}")
       Fimg = torch.div(Fimg, kmask_count)

    return Fimg

def mrf_forwardm_3D(tmaps, dict, csm, m0map, kmask, tmask=None, normalize=True, learned_scaling=None, return_images_only=False):
    """
    Inputs:
    tmaps: (6, N, N). already coils combined.
    csm: (1, N, N, nc, 2), or None
    kmask: (1, nt, N, N, nc), or None
    tmask: (1, N, N)
    m0map: (1, N, N, 2)
    learned_scaling: (1, N, N, 2)

    Return: (1, nt, N, N, nc, 2)
    """
    if tmaps.ndim  == 4:
        if tmask is not None:
           tmaps = tmask * tmaps
        
        _, s, M, N = tmaps.shape
        tms = torch.chunk(tmaps, s//2, dim=1)
        fps = Variable(torch.zeros((dict.shape[1]//2, M, N, s//2, 2))).type(dtype)

    elif tmaps.ndim == 5:
        if tmask is not None:
           tmaps = tmask.unsqueeze(1).unsqueeze(-1) * tmaps
        
        _, _, M, N, s = tmaps.shape
        tms = torch.chunk(tmaps, s, dim=-1)
        fps = Variable(torch.zeros((dict.shape[1]//2, M, N, s, 2))).type(dtype)
        print(f"fps: {fps.shape}, dict: {dict.shape}")
            
    # retreive fingerprints from dict
    for n, tmap in enumerate(tms):
        fps[:,:,:,n,:] = inverse_DM(tmap.squeeze(-1), dict, tmask, normalize=normalize) # (nt, N, N, 2)
            
  
    if m0map is not None:
       fps, m0map = torch.view_as_complex(fps), torch.view_as_complex(m0map)
       scaled = fps * m0map 
    else:
       scaled = torch.view_as_complex(fps)

    if learned_scaling is not None:
       factor = torch.view_as_complex(learned_scaling.permute(0,2,3,1).contiguous())
       print(f"Learned factor: {factor.shape}")
       scaled = scaled * factor

    scaled = torch.stack((scaled.real, scaled.imag), -1) # (nt, 256, 256, 2)
    if return_images_only:
       return scaled.unsqueeze(0)    

    kspace_fps = mrf_fft2(scaled, csm) # (1, nt, N, N, nc, 2) or (1, nt, N, N, 2) if no csm
    Fimg = kspace_fps * kmask if kmask is not None else kspace_fps    

    if kmask is not None:
       print(f"kspace_fps: {kspace_fps.shape}, kmask: {kmask.shape}")
       kmask_count = torch.sum(kmask, 1) # (1, N, N)
       kmask_count[kmask_count==0] = 1
       print(f"kmask_count: {kmask_count.shape}")
       Fimg = torch.div(Fimg, kmask_count)

    return Fimg



'''
def mrf_syn_forwardm(tmaps, dict, csm, m0map, kmask, tmask=None, normalize=True, learned_scaling=None, return_images_only=False):
    """
    Inputs:
    tmaps: (1, 2, N, N). already coils combined.
    csm: (1, N, N, nc, 2), or None
    kmask: (1, nt, N, N, nc), or None
    tmask: (1, N, N)
    m0map: (1, N, N, 2)
    learned_scaling: (1, N, N, 2)

    Return: (1, nt, N, N, nc, 2)
    """
    if tmask is not None:
       tmaps = tmask * tmaps

    # retreive fingerprints from dict
    fps = inverse_DM(tmaps, dict, normalize=normalize) # (nt, N, N, 2)
  
    if m0map is not None:
       fps, m0map = torch.view_as_complex(fps), torch.view_as_complex(m0map)
       scaled = fps * m0map 
       if learned_scaling is not None:
          factor = torch.view_as_complex(learned_scaling)
          print(f"factor: {factor.shape}")
          scaled = scaled * factor
    else:
       print(f"m0 is Not used!!!!!!!")
       scaled = torch.view_as_complex(fps)
    scaled = torch.stack((scaled.real, scaled.imag), -1) # (nt, 256, 256, 2)
    
    if return_images_only:
       print(f"Return the images only!!!!!!!!! Doing dictionary matching.")
       return scaled.unsqueeze(0)

    kspace_fps = mrf_fft2(scaled, csm) # (1, nt, N, N, nc, 2) or (1, nt, N, N, 2) if no csm
    Fimg = kspace_fps * kmask if kmask is not None else kspace_fps    

    return Fimg


def mrf_dict_forwardm(tmaps, dict, csm, m0map, kmask, normalize=True):
    """
    Given the predicted grids (tmaps), do inverse dict macthing to get MRF frames and return them.
    """
    # retreive fingerprints from dict
    fps = inverse_DM(tmaps, dict, normalize=normalize) # (nt, N, N, 2)
    if m0map is not None:
       fps, m0map = torch.view_as_complex(fps), torch.view_as_complex(m0map)
       scaled = fps * m0map   
       return torch.stack((scaled.real, scaled.imag), -1) # (nt, 256, 256, 2)

    return fps
'''

def denoising_forwardm(img, mask=None, downsampler=None):
    return img

def inpainting_forwardm(img, mask, downsampler=None):
    assert mask is not None
    return img * mask

def sr_forwardm(img, mask, downsampler):
    return downsampler(img)



'''
import h5py
import os
from torch.nn.functional import grid_sample
import csv
import pandas as pd

with_1000_or_not = False
time_points = 1152
nth_coil = 4

dtype = torch.cuda.FloatTensor

data_path = '/mnt/yapdata/data/yilinliu/datasets/MRF-DIP/144_8coils'
gt_mask_path = '/mnt/yapdata/data/yilinliu/datasets/MRF-DIP/equal-spaced'
#gt_mask_path = '/mnt/yapdata/data/yilinliu/datasets/MRF-DIP'
folder_path = '/mnt/yapdata/data/yilinliu/datasets/MRF-DIP/144_8coils'

# kspace data
#f = h5py.File(os.path.join(data_path, 'kspace_pca_multicoil.h5'), 'r')
#dr, di = np.flip(f['real'][:], axis=(1,2)), np.flip(f['imag'][:], axis=(1,2))
#slice_ksp = np.stack((dr, di), axis=-1)  # (nt, 256, 256, nc, 2)
#slice_ksp_torchtensor = torch.from_numpy(slice_ksp).unsqueeze(0).type(dtype)

fm = h5py.File(os.path.join('/mnt/yapdata/data/yilinliu/datasets/MRF-DIP', 'actual_kmask.h5'), 'r')
kmask = np.flip(fm['mask'][:time_points], axis=(1,2))
kmask = (kmask != 0.)
kmask = torch.from_numpy(kmask.copy()).unsqueeze(-1).unsqueeze(0).type(dtype)
print(f"kmask: {kmask.shape}, {torch.unique(kmask)}")

#masked_kspace = slice_ksp_torchtensor * kmask
#print(f"masked: {masked_kspace.sum()}")

#print(f"unders target: {masked_kspace.shape}")

mf = h5py.File(os.path.join(gt_mask_path, 'patternmatching.h5'), 'r')
mr = np.fliplr(np.flipud(mf['m0']['real'][:]))
mi = np.fliplr(np.flipud(mf['m0']['imag'][:]))
m0map = torch.from_numpy(np.stack((mr, mi), -1)).unsqueeze(0).type(dtype) # (1, 256, 256, 2)

saved = h5py.File(f'/shenlab/lab_stor/yilinliu/make_data/all/m0map.h5', 'w')
saved.create_dataset('r', data=mr)
saved.create_dataset('i', data=mi)
saved.close()

        
cf = h5py.File(os.path.join(folder_path, 'cmap_pca.h5'), 'r')
cr, ci = np.flip(cf['real'][:], axis=(0,1)), np.flip(cf['imag'][:], axis=(0,1))
cmap = torch.from_numpy(np.stack((cr, ci), -1)).unsqueeze(0).type(dtype) # (1,256, 256, nc, 2)

# visualize the fps from the actual kspace
#pred_t_orig = mrf_ifft2(slice_ksp_torchtensor, None)[0].data.cpu().numpy()


# get T1, T2 to construct the grid
#tf = h5py.File(os.path.join(gt_mask_path, 'patternmatching.mat'), 'r')
tf = h5py.File(os.path.join(gt_mask_path, 'patternmatching.h5'), 'r')
t1, t2 = tf['t1big'][:], tf['t2big'][:]
t1, t2 = torch.from_numpy(t1), torch.from_numpy(t2)
tmaps = torch.stack((t1, t2), 0).unsqueeze(0).type(dtype)
print(f"tmaps:{tmaps.shape} | t1.min: {t1.min()}, t1.max: {t1.max()}, t2.min: {t2.min()}, t2.max: {t2.max()} ")
t1n = 2 * ((t1 - t1.min()) / (t1.max() - t1.min())) -1
t2n = 2 * ((t2 - t2.min()) / (t2.max() - t2.min())) -1

grid = torch.stack((t1n, t2n), -1).unsqueeze(0).type(dtype) # (1,256,256,2)
print("grid", grid.shape)
dict, r = prepare_dictionary(time_points)
print(f"dict: {dict.shape}, r: {r.shape}")

# The predicted Tmaps using the tmaps coordinates
final = grid_sample(r, grid, mode='bilinear', padding_mode='zeros', align_corners=True)  #retrieve_tmaps(grid, r, interp_mode='bilinear')
dict_frames_m0 = mrf_dict_forwardm(grid.permute(0,3,1,2), dict, None, m0map, None) #grid_sample(dict, grid, mode='bilinear', padding_mode='zeros', align_corners=True)  #mrf_dict_forwardm(grid, dict,None, m0map, None)
dict_frames = mrf_dict_forwardm(grid.permute(0,3,1,2), dict, None, None, None)
print(f"dict_frames: {dict_frames.shape}")

# Save the dictionary fingerprints
print(f"dict fingerprints: {dict_frames_m0.shape}")
#saved = h5py.File(f'/mnt/yapdata/data/yilinliu/datasets/MRF-DIP/144_8coils/dict_fps_m0_{time_points}.h5','w')
saved = h5py.File(f'/shenlab/lab_stor/yilinliu/make_data/all/dict_fps_m0_{time_points}.h5', 'w')
saved.create_dataset('r', data=dict_frames_m0[...,0].data.cpu().numpy())
saved.create_dataset('i', data=dict_frames_m0[...,1].data.cpu().numpy())
saved.close()

#saved = h5py.File(f'/mnt/yapdata/data/yilinliu/datasets/MRF-DIP/144_8coils/dict_fps_ri_{time_points}.h5','w')
saved = h5py.File(f'/shenlab/lab_stor/yilinliu/make_data/all/dict_fps_ri_{time_points}.h5', 'w')
saved.create_dataset('r', data=dict_frames[...,0].data.cpu().numpy())
saved.create_dataset('i', data=dict_frames[...,1].data.cpu().numpy())
saved.close()


# The predicted fingerprints --> kspace
pred_k_nocsm = mrf_forwardm(tmaps, dict, None, m0map, None) # under-sampled kspace per-coil data
pred_t_nocsm = mrf_ifft2(pred_k_nocsm, None)

savet=h5py.File(f'/shenlab/lab_stor/yilinliu/make_data/all/full_ksp_from_dict_{with_1000_or_not}.h5', 'w')
savet.create_dataset('r', data=pred_k_nocsm.data.cpu().numpy()[...,0])
savet.create_dataset('i', data=pred_k_nocsm.data.cpu().numpy()[...,1])
savet.close()

savet=h5py.File(f'/shenlab/lab_stor/yilinliu/make_data/all/imMRF_recon_dict_{with_1000_or_not}.h5', 'w')
savet.create_dataset('r', data=pred_t_nocsm.data.cpu().numpy()[...,0])
savet.create_dataset('i', data=pred_t_nocsm.data.cpu().numpy()[...,1])
savet.close()

# The predicted fingerprints * kmask --> kspace
pred_k_nocsm_kmask = mrf_forwardm(tmaps, dict, None, m0map, kmask) # under-sampled kspace per-coil data
pred_t_nocsm_kmask = mrf_ifft2(pred_k_nocsm_kmask, None)

savet=h5py.File(f'/shenlab/lab_stor/yilinliu/make_data/all/unders_ksp_from_dict_{with_1000_or_not}.h5', 'w')
savet.create_dataset('r', data=pred_k_nocsm_kmask.data.cpu().numpy()[...,0])
savet.create_dataset('i', data=pred_k_nocsm_kmask.data.cpu().numpy()[...,1])
savet.close()

savet=h5py.File(f'/shenlab/lab_stor/yilinliu/make_data/all/masked_imMRF_recon_dict_{with_1000_or_not}.h5', 'w')
savet.create_dataset('r', data=pred_t_nocsm_kmask.data.cpu().numpy()[...,0])
savet.create_dataset('i', data=pred_t_nocsm_kmask.data.cpu().numpy()[...,1])
savet.close()

'''
'''
savet=h5py.File('mrf-related/fps_from_full_dict_ksp_new.h5', 'w')
savet.create_dataset('r', data=pred_t_nocsm.data.cpu().numpy()[...,0])
savet.create_dataset('i', data=pred_t_nocsm.data.cpu().numpy()[...,1])
savet.close()
'''
'''
#final2 = retrieve_tmaps(grid, r, interp_mode='bilinear', align_corners=True)

pd.DataFrame(r[0,0,...].data.cpu().numpy()).to_csv('mrf-related/r_t1.csv')
pd.DataFrame(r[0,1,...].data.cpu().numpy()).to_csv('mrf-related/r_t2.csv')

pd.DataFrame(grid[0,...,0].data.cpu().numpy()).to_csv('mrf-related/grid_t1.csv')
pd.DataFrame(grid[0,...,1].data.cpu().numpy()).to_csv('mrf-related/grid_t2.csv')

pd.DataFrame(tf['t1big'][:]).to_csv('mrf-related/gt_t1.csv')
pd.DataFrame(tf['t2big'][:]).to_csv('mrf-related/gt_t2.csv')

pd.DataFrame(final[0,0,...].data.cpu().numpy()).to_csv('mrf-related/pred_t1_aligned.csv')
pd.DataFrame(final[0,1,...].data.cpu().numpy()).to_csv('mrf-related/pred_t2_aligned.csv')

save=h5py.File('mrf-related/retrieved_tmaps.h5', 'w')
save.create_dataset('tmaps', data=final.data.cpu().numpy())
save.close()

save3=h5py.File('/mnt/yapdata/data/yilinliu/datasets/MRF-DIP/144_8coils/full_ksp_from_dict.h5', 'w')
save3.create_dataset('r', data=pred_k.data.cpu().numpy()[...,0])
save3.create_dataset('i', data=pred_k.data.cpu().numpy()[...,1])
save3.close()

save2=h5py.File('mrf-related/fps_from_ksp.h5', 'w')
save2.create_dataset('fps', data=pred_t.data.cpu().numpy())
save2.close()

gf = h5py.File('mrf-related/normalized.h5', 'w')
gf.create_dataset('mrf-related/grid', data=grid.data.cpu().numpy())
gf.close()

rf = h5py.File('mrf-related/r.h5', 'w')
rf.create_dataset('r', data=r.data.cpu().numpy())
rf.close()

tf2 = h5py.File('mrf-related/tmaps.h5', 'w')
tf2.create_dataset('t1', data=tf['t1big'][:])
tf2.create_dataset('t2', data=tf['t2big'][:])
tf.close()

pred_tc = pred_t[...,0] + 1j*pred_t[...,1]

dict_orig, r_orig = prepare_dictionary(time_points, just_dict_r=True)
if dict_frames.shape[0] == 1:
   dict_frames = dict_frames[0]
if dict_frames.shape[-1] == 2:
   dict_frames = dict_frames[...,0] + 1j * dict_frames[...,1]
DM(dict_frames.data.cpu().numpy(), time_points, dict_orig, r_orig, f'mrf-related/fps2tmaps_{time_points}.h5')

'''
'''
cf = h5py.File(os.path.join(data_path, 'cmap_pca.h5'), 'r')
cr = cf['real'][:]
ci = cf['imag'][:]
cmap = np.stack((cr, ci), -1)
cmap = torch.from_numpy(cmap).unsqueeze(0).type(dtype)
print(f"cmap: {cmap.shape}")

mf = h5py.File(os.path.join(data_path, 'm0_2304pnts.h5'), 'r')
mr = mf['real'][:]
mi = mf['imag'][:]
m0map = np.stack((mr, mi), -1) # (256, 256, 2)
m0map = torch.from_numpy(m0map).unsqueeze(0).type(dtype)
print(f"m0map: {m0map.shape}")

dict, r = prepare_dictionary(144)

kspace_out = mrf_forwardm(grid, dict, cmap, m0map, mask_tensor)
print(f"kspace_out: {kspace_out.shape}")

loss_func = nn.MSELoss()
loss = loss_func(kspace_out, masked_kspace)
print(f"loss: {loss}, unders kspace: {masked_kspace.sum()}, kspace_out: {kspace_out.sum()}")
'''
