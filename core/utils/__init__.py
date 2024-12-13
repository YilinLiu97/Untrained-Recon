def getForwardm(args):
    if 'mri' in args.task:
        from utils.forwardm import mri_forwardm, mri_ksp_sup
        if args.ksp_sup:
           return mri_ksp_sup
        return mri_forwardm
#    elif args.task == 'mrf_real' or args.task == 'mrf_imMRF_fft':
    elif ('real' in args.task or 'fft' in args.task) and 'image' not in args.task and 'denoising' not in args.task:
        from utils.forwardm import mrf_forwardm
        return mrf_forwardm
    elif 'syn' in args.task and 'image' not in args.task:
        from utils.forwardm import mrf_syn_forwardm
        return mrf_syn_forwardm
    elif args.task == 'mrf_imMRF':
        from utils.forwardm import mrf_forwardm
        return mrf_forwardm
    elif args.task == 'mrf_imMRF_3D': 
        from utils.forwardm import mrf_forwardm_3D
        return mrf_forwardm_3D
    elif args.task == 'denoising' or args.task == 'real_denoising':
        from utils.forwardm import denoising_forwardm
        return denoising_forwardm
    elif args.task == 'inpainting':
        from utils.forwardm import inpainting_forwardm
        return inpainting_forwardm
    elif args.task == 'sr':
        from utils.forwardm import sr_forwardm
        return sr_forwardm
