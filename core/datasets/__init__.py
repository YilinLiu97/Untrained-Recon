import os
import h5py
import numpy as np
import torch
def getDataset(args, net=None):
    if args.task == 'mri_knee' or args.task == 'mri_brain':
        from datasets.fastMRI_data import fastMRIData_DIP_multicoil
        print(f"Reading fastMRI data......")
        data = fastMRIData_DIP_multicoil(args)
    elif args.task == 'mri_stanford3D':
        from datasets.stanfordMRI_3D import stanfordMRI3D_multicoil
        print(f"Reading stanford3D data......")
        data = stanfordMRI3D_multicoil(args) 
    elif args.task == 'mri_pretrain_multicoil':
        from datasets.fastMRI_data import fastMRIData_pretrain_multicoil
        data = fastMRIData_pretrain_multicoil(args, net)
    elif args.task == 'mri_pretrain_singlecoil':
        from datasets.fastMRI_data import fastMRIData_pretrain_singlecoil
        data = fastMRIData_pretrain_singlecoil(args, net)
    elif args.task == 'mrf_coil_real':
        from datasets.mrf import mrfData_coil_real
        data = mrfData_coil_real(args)
    elif args.task == 'mrf_coil_syn':
        from datasets.mrf import mrfData_coil_syn
        data = mrfData_coil_syn(args)
    elif args.task == 'mrf_full_syn':
        from datasets.mrf import mrfData_full_syn
        data = mrfData_full_syn(args)
    elif args.task == 'mrf_imMRF':
        from datasets.mrf import mrfData_imMRF
        print(f"Reading coils-combined imMRF data......")
        data = mrfData_imMRF(args)
    elif args.task == 'mrf_imMRF_fft':
        from datasets.mrf import mrfData_imMRF_fft
        data = mrfData_imMRF_fft(args)
    elif args.task == 'mrf_imMRF_3D':
        if args.mrf3D_ori == 'sagittal':
            from datasets.mrf3d import mrfData_imMRF_3D_sagittal
            data = mrfData_imMRF_3D_sagittal(args)
        elif args.mrf3D_ori == 'coronal':
            from datasets.mrf3d import mrfData_imMRF_3D_coronal
            data = mrfData_imMRF_3D_coronal(args)  
        elif args.mrf3D_ori == 'axial':
            from datasets.mrf3d import mrfData_imMRF_3D_axial
            data = mrfData_imMRF_3D_axial(args)
        else:
            raise NotImplementedError("wrong orientation.")
    elif args.task == 'mrf_syn_image':
        from datasets.mrf import mrfData_syn_image
        data = mrfData_syn_image(args)
    elif args.task == 'denoising':
        from datasets.denoising import noisyImages
        data = noisyImages(args)
    elif args.task == 'real_denoising':
        from datasets.denoising import realNoisy
        data = realNoisy(args)
    elif args.task == 'SIDD_denoising':
        from datasets.denoising import SIDD
        data = SIDD(args)
    elif args.task == 'inpainting':
        from datasets.inpainting import inpaintImages
        data = inpaintImages(args)
    elif args.task == 'sr':
        from datasets.sr import blurImages
        data = blurImages(args)
    else:
        raise NotImplementedError("No such dataset.")
    return data

def update_params_from_data(args):
    files = os.listdir(args.folder_path)
 
    # get all the slices from each subject/file
    for i in range(1):
        try:
          f = h5py.File(args.folder_path + files[i], 'r')
        except:
          print(f"Error: {files[i]} cannot be read. Skipped!")
          continue
        
        vol = f["kspace"][:]
        print(f"{files[i]}: {vol.shape}")
        
        for slicenu in range(1):
            slice_ksp = vol[slicenu]
                
            if not args.progressive:
                args.in_size = slice_ksp.shape[-2:]
            args.out_size = slice_ksp.shape[-2:]
            if slice_ksp.ndim == 2:
               args.out_chns = 2
            else:
               args.out_chns = slice_ksp.shape[0] * 2
            
    return args
