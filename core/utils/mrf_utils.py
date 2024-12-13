import os

import numpy as np
import torch
from torch.nn.functional import grid_sample
import h5py
import itertools
import scipy.io as sio
import random
import decimal
import time
from .mri_utils import *
from .common_utils import *

dtype = torch.cuda.FloatTensor

dict_2D_PATH = '/home/yunkuipa/stor/dataset/mrf/2D/equal-spaced'
dict_3D_PATH = '/home/yunkuipa/stor/dataset/mrf/3D/dict_3D_Scan43_331.mat'
#meas_PATH = data_root

def np_to_torch(arr):
    return torch.from_numpy(arr)
    
def mrf_fft2(imMRF, csm=None):
    """
    fingerprints: (nt, N, N, 2) 2-channel real-valued
    csm: (1, N, N, nc, 2), if not None
    return (1, nt, N, N, nc, 2) or (1, nt, N, N, 2) if csm is None
    """
    if imMRF.shape[-1] == 2:
       im = torch.view_as_complex(imMRF)
       if csm is not None and csm.shape[-1] == 2:
          print("csm is used!")
          csm = torch.view_as_complex(csm)
          im = im.unsqueeze(-1) * csm.unsqueeze(0)
          kdata = torch.fft.ifftshift(torch.fft.ifftshift(torch.fft.fft(torch.fft.fft(torch.fft.ifftshift(torch.fft.ifftshift(im, -2),-3),axis=-3),axis=-2),-3),-2)
       else:
          print("csm is NOT used!")
          kdata = torch.fft.ifftshift(torch.fft.ifftshift(torch.fft.fft(torch.fft.fft(torch.fft.ifftshift(torch.fft.ifftshift(im, -1),-2),axis=-2),axis=-1),-2),-1)
    #      kdata = torch.fft.ifftshift(torch.fft.fft2(torch.fft.ifftshift(im, dim=(-1,-2))), dim=(-2,-1))
    kdata = torch.stack([torch.real(kdata), torch.imag(kdata)], -1)
    return kdata

def mrf_ifft2(kspace_mrf, csm):
    """
    kdata: (1, nt, N, N, nc, 2) real or (nt, N, N)
    csm: (1, N, N, nc, 2), if not None
    return: (1, nt, N, N, 2)
    """
    if kspace_mrf.shape[-1] == 2:
       kdata = torch.view_as_complex(kspace_mrf)
       if csm is not None and csm.shape[-1] == 2:
          print("csm is used!")
          csm = torch.view_as_complex(csm)
#          MRFimg = torch.fft.fftshift(torch.fft.fftshift(torch.fft.ifft(torch.fft.ifft(torch.fft.fftshift(torch.fft.fftshift(kdata,2),3),axis=2),axis=3), 3), 2)
          MRFimg = torch.fft.fftshift(torch.fft.fftshift(torch.fft.ifft(torch.fft.ifft(torch.fft.fftshift(torch.fft.fftshift(kdata,-2),-3),axis=-2),axis=-3), -3), -2)
          coil_images = MRFimg * torch.conj(csm)
          combined = torch.sum(coil_images, -1)
          return torch.stack((combined.real, combined.imag), -1)
       else:
          print("csm is NOT used!")
          MRFimg = torch.fft.fftshift(torch.fft.fftshift(torch.fft.ifft(torch.fft.ifft(torch.fft.fftshift(torch.fft.fftshift(kdata,-1),-2),axis=-1),axis=-2), -2), -1)   
     #     MRFimg = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(kdata, dim=(-1,-2))), dim=(-2,-1))
    return torch.stack((MRFimg.real, MRFimg.imag), -1)


def normalize_(d, tmask, d_max, d_min, norm=True):
    if norm:
       if tmask is not None:
          mask = tmask.type(torch.cuda.BoolTensor)
          in_mask = torch.masked_select(d, mask)
          in_mask =  2 * ((in_mask - d_min) / (d_max - d_min)) - 1    #2 * ((in_mask - in_mask.min()) / (in_mask.max() - in_mask.min())) - 1
          dd = d.clone() # to enable gradient
          dd[tmask==1] = in_mask
          d = dd
       else:
          print(f"normalizing without mask.....................")
          d = 2 * ((d - d_min) / (d_max - d_min)) - 1
    return d

def retrieve_tmaps(loc_map, r, tmask, interp_mode='bicubic', pad='border', normalize=True): #, t1_max=3500, t1_min=200, t2_max=300, t2_min=10):
   """
   loc_map: normalized to (1, 256, 256, 2), continuous [-1, 1]
   r: (1, 2, 258, 53)
   """
   loc_map = torch.stack((normalize_(loc_map[:,0,...], tmask, d_max=T2_MAX, d_min=T2_MIN, norm=normalize), normalize_(loc_map[:,1,...], tmask, d_max=T1_MAX, d_min=T1_MIN, norm=normalize)), -1)
   final = grid_sample(r, loc_map, mode=interp_mode, padding_mode=pad, align_corners=True)
   print("Retreived tmaps: ", final)
   print(f">>>>>>>>>>>>>>> T1 mean+std: {final[0,0,...].mean()}+{final[0,0,...].std()}")
   print(f">>>>>>>>>>>>>>> T2 mean+std: {final[0,1,...].mean()}+{final[0,1,...].std()}")
   
#   final = torch.stack((loc_map[:,0,...], loc_map[:,1,...]), 1) #final
   return final

def inverse_DM(tissue_out, new_dict, tmask, interp_mode='bicubic', pad='border', normalize=True):
    """
    tissue_out: (1, 2, N, N) used as a grid
    new_dict: (1, nt, 258, 53)
    
    return: (nt, N, N, 2)
    """
    tissue_grid = torch.stack((normalize_(tissue_out[:,0,...], tmask, d_max=T2_MAX, d_min=T2_MIN, norm=normalize), normalize_(tissue_out[:,1,...], tmask, d_max=T1_MAX, d_min=T1_MIN, norm=normalize)), -1)
    print(f"tissue_grid: {tissue_grid}")
    time_pts = new_dict.shape[1]
    fps = grid_sample(new_dict, tissue_grid, mode=interp_mode, padding_mode=pad, align_corners=True)
    print(f"fps: {fps}")
    return torch.stack((fps[0,:time_pts//2,...], fps[0,time_pts//2:,...]), -1)
    
def DM(MRFimg, Nex, dict, r, save_name, use_gpu=False):
    """
    MRFimg: (t, N, N): need to be complex
    dict: (19419, 2*t): real
    """
    if MRFimg.shape[1] != MRFimg.shape[2]:
       MRFimg = MRFimg.transpose(2,0,1) # (t, N, N)
    print(f"MRFimg: {MRFimg.shape}")       
    print(f"Nex: {Nex}")
    dict = dict[...,:Nex] + 1j * dict[...,Nex:]
    xp = np    
    if use_gpu:
        MRFimg = cupy.asarray(MRFimg)
        dict = cupy.asarray(dict)
        
    N = MRFimg.shape[1]
    dict = dict[:,:Nex]
    MRFimg = MRFimg[:Nex,:,:]
    MRFimg = MRFimg.reshape((Nex,N*N), order='F')
    MRFimg = MRFimg.transpose()
    MRFimgnorm = np.zeros((MRFimg.shape[0],1),dtype=np.float32)
    MRFimgnorm[:,0] = xp.sqrt(xp.sum(MRFimg * xp.conj(MRFimg),axis=1))
    print(f"MRFimgnorm: {MRFimgnorm.shape}") # (65536,1)

    # dictnorm = sqrt(sum(dict(trange,:).*conj(dict(trange,:))));
    dictnorm = np.zeros((1,dict.shape[0]),dtype=np.float32)
    dictnorm[0,:] = xp.sqrt(xp.sum(dict * xp.conj(dict),axis=1))
    print(f"dictnorm: {dictnorm.shape}") # (1, 19419)
    normAll = xp.matmul(MRFimgnorm,dictnorm)
    print(f"normAll: {normAll.shape}") # (65536, 19419)

    # perform inner product
    #innerProduct = conj(xx)*dict(trange,:)./normAll; clear normAll
    innerproduct = xp.matmul(xp.conj(MRFimg),dict.transpose())
    print(f"innerproduct: {innerproduct.shape}") # (65536, 19419)
    innerproduct = np.abs(innerproduct) / normAll
    print(f"innerproduct: {innerproduct.shape}") # (65536, 19419)
    indexm = xp.argmax(innerproduct,axis=1)
    print(f"indexm: {indexm.shape}") # (65536, 1)

    # extract T1 and T2 maps
    t1map = r[0,indexm[:]]
    t1map = t1map.reshape((N,N))
    t2map = r[1,indexm[:]]
    t2map = t2map.reshape((N,N))

    # calculate proton density map
    m0map = np.zeros((N*N),dtype=np.float32)

    for i in range(0,indexm.shape[0]):
        dictCol = dict[indexm[i],:]
        tempdictCol = dictCol.conj()/sum(dictCol.conj()*dictCol)
        m0map[i] = abs(sum(tempdictCol*MRFimg[i,:]))
    m0map = m0map.reshape((N,N),order='F')

    f=h5py.File(f"{save_name}",'w')
    f.create_dataset('t1', data=t1map)
    f.create_dataset('t2', data=t2map)
    f.create_dataset('m0', data=m0map)
    f.close()


def prepare_dictionary(time_pts, which_dict='dict_331', ND=True, just_dict_r=False):
    print(f"time pts when prepare: {time_pts}")
    MRFDict_filename = os.path.join(dict_2D_PATH, f'{which_dict}') if ND == '2D' else dict_3D_PATH 
    print(f"{MRFDict_filename} is loaded.")
    #dict2 = os.path.join(dict_PATH, 'dict_SVD_1152_15.h5')
    f = h5py.File(MRFDict_filename, 'r')
    print(f'dict.keys()={list(f.keys())}')
    dict = np.asarray(f['dict']) if "SVD" not in which_dict else f['dict']
    dict_r = np.asarray(dict['real'], dtype=np.float32)[:,:time_pts]
    dict_i = np.asarray(dict['imag'], dtype=np.float32)[:,:time_pts]

    dict_c = dict_r + 1j*dict_i
#    dictnorm = np.zeros((1,dict_c.shape[0]),dtype=np.float32)
#    dictnorm[0,:] = np.sqrt(np.sum(dict_c * np.conj(dict_c),axis=1))
#    dn = dict_c / dictnorm.transpose(1,0)

    dict = np.concatenate((dict_r, dict_i), -1)
    r = np.asarray(f['r']) # (2, 13123)

    global T1_MAX
    T1_MAX = np.max(r[0,...])
    global T1_MIN 
    T1_MIN = np.min(r[0,...])
    global T2_MAX 
    T2_MAX = np.max(r[1,...])
    global T2_MIN 
    T2_MIN = np.min(r[1,...])
        
    print(f'dict.shape:{dict.shape}, tissues:{r.shape}, T1: {T1_MAX},{T1_MIN}, T2: {T2_MAX}, {T2_MIN}')
    if just_dict_r:
       return dict, r    

    t1v = np.unique(r[0,...])
    t2v = np.unique(r[1,...])
    
    c = list(itertools.product(t1v, t2v)) # (13674, 2) => (258 x 53, 2)
  
    r2 = r.transpose() # (2, 13123)
    r3 = [tuple(i) for i in r2] # convert to array of tuples
    invalid = set(c) - set(r3) # 13674 - 13123 = 551
    invalid_list = list(invalid)

    for i in range(len(invalid)):
        idx = c.index(invalid_list[i])
        c[idx] = 'NAN'
        
    c = np.array(c, dtype=object) # for numpy version 1.26.x
    tissue_table = np.reshape(c, [len(t1v), len(t2v)]) # (258, 53, 2)
    print(f"coordinate system: {tissue_table.shape}")

    v1, v2 = np.meshgrid(t1v, t2v, indexing='ij')
    r_final = np.stack((v1, v2), 0) # (2, 258, 53)
    
    pos_table = np.empty((len(t1v), len(t2v)))
    for i in range(len(t1v)):
        for j in range(len(t2v)):
            if tissue_table[i,j] != 'NAN':
                idx = r3.index(tissue_table[i,j])
                pos_table[i,j] = idx
            else:
                pos_table[i,j] = -999999
                
    pos_table = np.asarray(pos_table, dtype='int') # (258, 53, 2)
    
    new_dict = np.zeros((len(t1v), len(t2v), time_pts*2))
    new_dict[pos_table != -999999] = dict
    
    print(f"Inverse DM is ready, with tissue table: {tissue_table.shape} and idx table: {pos_table.shape}")
    
    return np_to_torch(new_dict.transpose(2,0,1)[np.newaxis,:,:,:]).type(dtype), np_to_torch(r_final[np.newaxis,...]).type(dtype) # dict: (1, tps, 258, 53), r: (2, 258, 53)

def prepare_dictionary2(time_pts):
    MRFDict_filename = os.path.join(dict_PATH, 'dict.mat')
    f = h5py.File(MRFDict_filename, 'r')
    print(f'dict.keys()={list(f.keys())}')
    dict = np.asarray(f['dict'])
    dict_r = np.asarray(dict['real'], dtype=np.float32)[:,:time_pts]
    dict_i = np.asarray(dict['imag'], dtype=np.float32)[:,:time_pts]
    dict = np.concatenate((dict_r, dict_i), -1)
    r = np.asarray(f['r']) #/ np.array([5000, 500])[:, np.newaxis] # (2, 13123)
           
    print(f'dict.shape:{dict.shape}, tissues:{r.shape}, r: {r}')
    
    t1v = np.unique(r[0,...])
    t2v = np.unique(r[1,...])

    c = list(itertools.product(t1v, t2v)) # (13674, 2) => (258 x 53, 2)

    r2 = r.transpose() # (2, 13123)
    r3 = [tuple(i) for i in r2] # convert to array of tuples
    invalid = set(c) - set(r3) # 13674 - 13123 = 551
    invalid_list = list(invalid)
    
    for i in range(len(invalid)):
        idx = c.index(invalid_list[i])
        c[idx] = 'NAN'
        
    tissue_table = np.reshape(c, [len(t1v), len(t2v)]) # (258, 53, 2)
    
    pos_table = np.empty((len(t1v), len(t2v)))
    for i in range(len(t1v)):
        for j in range(len(t2v)):
            if tissue_table[i,j] != 'NAN':
                idx = r3.index(tissue_table[i,j])
                pos_table[i,j] = idx
            else:
                pos_table[i,j] = -999999
                
    pos_table = np.asarray(pos_table, dtype='int') # (258, 53, 2)
    print(f"Inverse DM is ready, with tissue table: {tissue_table.shape} and idx table: {pos_table.shape}")
    return np_to_torch(pos_table).type(torch.cuda.LongTensor), np_to_torch(t1v).type(dtype), np_to_torch(t2v).type(dtype), np_to_torch(dict).type(dtype), np_to_torch(r).type(dtype), tissue_table

def scale_real_data(data, constant):
   """
   data: (..., 2)
   constant: complex
   """
   dc = torch.view_as_complex(data)
   scaled = constant * dc
   return torch.stack((scaled.real, scaled.imag), -1)


def save(d, filename='save.h5', complex=False):
    f = h5py.File(os.path.join(filename), 'w')
    pts = d.shape[-1] // 2
    if complex:
        f.create_dataset('imMRF_generated/real', data=d[...,:pts])
        f.create_dataset('imMRF_generated/imag', data=d[...,pts:])
    else:
        mag = np.abs(d[...,:pts] + 1j * d[...,pts:])
        f.create_dataset('mag', data=mag)
    f.close()

'''
# Main
time_pts = 144
save_name_FP = f'save_fingerprints_{time_pts}'
save_name_DM = f'generated_DM_{time_pts}'
tissue_table, pos_table, t1v, t2v, dict, r = prepare_dictionary(time_pts)

t1_pred = np.random.uniform(60,5000,(30,256,256))
t2_pred = np.random.uniform(10,500,(30,256,256))

tf = h5py.File(os.path.join(data_root,'patternmatching_2304.mat'),'r')
#cf = sio.loadmat(os.path.join(data_root,'cmap.mat'))

t1gt = tf['t1big'][:]
t1_pred = np.clip(t1gt + np.random.uniform(0,0.6, t1gt.shape), 60, 5000)
t2gt = tf['t2big'][:]
t2_pred = np.clip(t2gt + np.random.uniform(0,0.6, t2gt.shape), 10, 500)
m0 = tf['m0big'][:]
our_m0 = cal_m0()
#cmap = cf['cmap'][:]

tissue_out = np.stack((t1_pred, t2_pred), -1)

start_time = time.time()
final = inverse_DM(tissue_out, pos_table, t1v, t2v, dict, time_pts)
print(f"Inverse DM with {time_pts} time points  taken --- {time.time() - start_time} --- seconds")

#save(final* m0[...,np.newaxis], os.path.join(dict_PATH, f'{save_name_FP}.h5'), complex=True)
print(f"Finished! Final is: {final.shape} saved in {dict_PATH}")

print(f"Now doing DM using the generated fingerprints...")
DM(final, time_pts, dict, r, save_name_DM, our_m0=None)
'''
