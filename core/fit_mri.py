import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from tqdm import tqdm
from collections import OrderedDict
from utils.mri_utils import *
from utils.common_utils import *
from utils.pruning_utils import *
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from pruning.DAM import *
from pruning.morphnet import *
from losses import *
from deepinv.loss.regularisers import JacobianSpectralNorm
from models.common import *

import copy
import os

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d


def eval_mri(gt, pred):
   gt, pred = normalize(gt, pred)
   p = compare_psnr(
        gt, pred, data_range=gt.max()
    )
   s = compare_ssim(
        gt, pred, multichannel=False, data_range=gt.max()
    )

   return {'ssim': s, 'psnr': p}


def get_Params(net):
   N, C, Sigma = [], [], []
   for layer, weight in net.named_parameters():
       if 'N' in layer:
          N.append(weight)
       if 'C' in layer:
          C.append(weight)
       if 'Sigma' in layer:
          Sigma.append(weight)
   return N, C, Sigma


def self_validation_idx(mask2d, percent=0.05):
    mask2d = mask2d.data.cpu().numpy()
    if mask2d.shape[0] == 1:
       mask2d = mask2d[0]
    where_ones = np.array(np.where(mask2d==1))
    m = len( where_ones[0] ) 
    self_val_idx = where_ones[:,np.random.randint(0,m,int(percent*m))]
    print(f"mask: {mask2d.shape}, m: {m}, idx: {self_val_idx.shape}")
    return torch.from_numpy(self_val_idx).long().cuda()
         
def fit(args,
        net,
        img_noisy_var,
        net_input,
        mask,
        downsampler,
        orig,
        apply_f,
        experiment_name,
        snapshots_save_path,
        layer_importance_path,
        csm = None,
        unders_recon = None,
        fully_meas = None
        ):

    reg_l2 = JacobianSpectralNorm(max_iter=10, tol=1e-3, eval_mode=False, verbose=True)

    p = [x for x in net.parameters()]
  
    num_convs = 0
    for param in list(filter(lambda param: len(param.data.shape)>2, net.parameters())):
        num_convs += 1

    out_grads = np.zeros((num_convs, args.num_iters))
    out_filters = np.zeros((num_convs+1, args.num_iters))

    mse_wrt_noisy = np.zeros(args.num_iters)

    ### optimizer
    print(f"optimize with {args.optimizer}", args.lr)
    
    if args.model_type == 'DIP_ds':
       optimizer = getOptimizer(net.parameters_no_deepspline(), args)
       ds_optimizer = getOptimizer(net.parameters_deepspline(), args)
    else:
       optimizer = getOptimizer(p, args)
       ds_optimizer = None

    if args.decay_lr:  
      print(f"With annealed learning rate ")
      lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=args.step_size,gamma=args.gamma)

    if len(args.param_idx) != 0:
      rows, cols = (args.num_iters, len(args.param_idx))
      lr_list = torch.zeros(rows, cols) #[[0 for i in range(cols)] for j in range(rows)]

    PSNR_list, SSIM_list, Loss_list, val_errors, Masked_PSNR_list, Masked_SSIM_list, Unmasked_PSNR_list, Unmasked_SSIM_list, Lips_c = [], [], [], [], [], [], [], [], []

    os.makedirs(snapshots_save_path, exist_ok=True)

    reg_noise_std = args.reg_noise_std # set to 1./20. for sigma=50
    noise = net_input.clone()
    net_input_saved = net_input.clone()
    print("With noise pertubation? ", (reg_noise_std>0))
    out_avg = None
    do_exit = False

    print("Apply dropout on input? ", (args.input_dropout>0))
    print("Apply gassuain lpf on input? ", (args.gaussian_blur_ks>0))

    ### main optimization loop
    extra_metrics = {}
    nonzero_list = []
    norms_list = []
    singular_list = []
    condition_list = []
    if args.num_power_iterations: 
       norms, w_svd, cond = spectral_norm(net, args.num_power_iterations)
       norms_list.append(norms)
       singular_list.append(w_svd)
       condition_list.append(cond)

    ## separate self-validation and regular training measurements
    if args.self_val_percent:
       self_val_idx = self_validation_idx(mask, args.self_val_percent)
       print(f">>>>>>>> Will use self-validation for Early Stopping <<<<<<<<")
    else:
       self_val_idx = None
       
    stats_gt = [[], [], [], [], []]
    stats_ns = [[], [], [], [], []]

    for i in tqdm(range(args.num_iters)):

        '''
        if (i+1) % 50 == 0:
           nonzero_ratio = plot_layer_importance(net, f"{snapshots_save_path}/layer_importance_iter{i+1}.png", args.pruning_sensitivity)
           nonzero_list.append(nonzero_ratio)
        '''

        optimizer.zero_grad()
        if ds_optimizer is not None:
           ds_optimizer.zero_grad()

        if reg_noise_std > 0:
           net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        net_input = Variable(net_input).cuda()
        
        # forward pass
        if args.use_freq_mask:
           freq_mask = get_freq_reg_mask(net_input.shape[1], i, args.num_iters, max_visible=None)
#           if args.freq_mask_mode == 'descending':
#              freq_mask = 1 - freq_mask

           net_input = net_input * freq_mask.unsqueeze(-1).unsqueeze(-1).tile((net_input.shape[-2],net_input.shape[-1])).type(dtype)

        if args.input_dropout:
           net_input = torch.nn.functional.dropout(net_input, p=args.input_dropout)
        
        if args.gaussian_blur_ks:
           gk = torchvision.transforms.GaussianBlur(args.gaussian_blur_ks) # args.gaussian_blur_sigma)
           net_input = gk(net_input)

        out = net(net_input)

        down_out, unmasked_img, masked_img, unmasked_gt, masked_gt = apply_f(out, mask, csm=csm, fully_meas=fully_meas, downsampler=downsampler)
       

        # If use self-validation
        if self_val_idx is not None:
           down_out_train = down_out.clone()
           down_out_train[:, :, self_val_idx[0], self_val_idx[1], :] = 0
           img_noisy_var_train = img_noisy_var.clone()
           img_noisy_var_train[:, :, self_val_idx[0], self_val_idx[1], :] = 0
  
           total_loss = loss_func(args.loss_type, down_out_train, img_noisy_var_train)
           val_loss = loss_func(args.loss_type, down_out[:, :, self_val_idx[0], self_val_idx[1], :], img_noisy_var[:, :, self_val_idx[0], self_val_idx[1], :])
        else:         
           # main loss
           total_loss = loss_func(args.loss_type, down_out, img_noisy_var)

        #if args.verbose:
        #   print('task loss: ', total_loss)

        # smoothing
        if args.exp_weight:
           if out_avg is None:
              out_avg = out.detach()
           else:
              out_avg = out_avg * args.exp_weight + out.detach() * (1 - args.exp_weight)
        else:
           out_avg = out.detach()
        
        reg_elem = reg_struc = 0.0
        if args.reg_type and args.decay:
           reg_elem = reg_struc  = 0.0
           for name, param in net.named_parameters():
               if param.requires_grad and torch.sum(torch.abs(param))>0:
                  if args.reg_type == 1:  # L1
                     reg_elem += torch.sum(torch.abs(param))
                  elif args.reg_type == 2:  # Hoyer
                     reg_elem += torch.sum(torch.abs(param)) / torch.sqrt(torch.sum(param ** 2))
                  elif args.reg_type == 3:  # Hoyer Square
                     reg_elem += (torch.sum(torch.abs(param)) ** 2) / torch.sum(param ** 2)
                  elif args.reg_type == 4:  # Transformed L1
                     reg_elem += torch.sum(2 * torch.abs(param) / (1 + torch.abs(param)))           
                  elif args.reg_type == 5: # L2
                     reg_elem += torch.sum(param ** 2)
                  else:
                     reg_elem = 0.0

               if args.decay[1] != 0:
                  if param.requires_grad and 'weight' in name:
                    if args.reg_type == 3:
                        if len(param.shape)==4:
                            reg_struc += ( (torch.sum(torch.sqrt(torch.sum(param**2,(0,2,3))))**2) + (torch.sum(torch.sqrt(torch.sum(param**2,(1,2,3))))**2) )/torch.sum(param**2)
                        elif len(param.shape)==2:
                            reg_struc += ( (torch.sum(torch.sqrt(torch.sum(param**2,0)))**2) + (torch.sum(torch.sqrt(torch.sum(param**2,1)))**2) )/torch.sum(param**2)
                    else:
                        raise NotImplementedError('Regularizer [%s] is not implemented for structural sparsity' % reg_type)

           if args.verbose:
              print("reg_loss: ", args.decay[0] * reg_elem + args.decay[1] * reg_struc)    
           total_loss += args.decay[0] * reg_elem + args.decay[1] * reg_struc 

        if args.singular_weight or args.ortho_weight:
           N, C, Sigmas = get_Params(net)
           sigma_reg = ortho_reg = 0
           for t in range(len(N)):
               sigma_reg += singular_reg_loss(Sigmas[t], args.singular_reg_type)
               ortho_reg += orthogology_loss(N[t]) + orthogology_loss(C[t])
           total_loss += sigma_reg * args.singular_weight + ortho_reg * args.ortho_weight
           if args.verbose:
              print(f"sigma reg: {sigma_reg * args.singular_weight}")
              print(f"ortho reg: {ortho_reg * args.ortho_weight}")

        if args.tv_weight:
           tv_reg = tv_loss(out, args.tv_weight)
           if args.verbose:
              print("tv reg: ", tv_reg)
           total_loss += tv_reg

        if args.Lipschitz_reg:
           lip_loss = torch.tensor(1)
           for name, p in net.named_parameters():
               if 'weight_c' in name:
                  lip_loss = lip_loss + torch.max(nn.functional.softplus(p))**2
                  if (i+1) % 10 == 0:
                     Lips_c.append(nn.functional.softplus(p).data.cpu().numpy())
           if args.verbose:
              print("lip_loss: ", lip_loss)
           total_loss += args.Lipschitz_reg * lip_loss

        total_loss.backward()
   
        optimizer.step()
        
        
        ## stats mri
        #img_it = unmasked_img[0]
        #img_ns = img_noisy_var[0]
        
        #stats2 = get_circular_statastic_mri(img_it, img_ns, size=0.2)
        
        #for ks in range(len(stats2)):
        #    stats_ns[ks].append(stats2[ks])


        # deplay the fitting loss every 10 iterations
        if (i+1) % args.iters_cal_acc == 0:
            # Evaluation
            img_out = out.squeeze(0)
            ncoils = img_out.shape[0] // 2
            img_out = torch.stack((img_out[:ncoils, :, :], img_out[ncoils:,:,:]), -1)
            imgout = combine_coil_rss(img_out, orig.shape)
            metrics = eval_mri(orig, imgout)
            
            unmaskedout = combine_coil_rss(unmasked_img[0], orig.shape)
            unmaskedgt = combine_coil_rss(unmasked_gt[0], orig.shape)
            unmasked_metrics = eval_mri(unmaskedgt, unmaskedout)
           
            maskedout = combine_coil_rss(masked_img[0], orig.shape)
            maskedgt = combine_coil_rss(masked_gt[0], orig.shape)
            masked_metrics = eval_mri(maskedgt, maskedout)

            Masked_PSNR_list.append(masked_metrics['psnr'])
            Masked_SSIM_list.append(masked_metrics['ssim'])
            Unmasked_PSNR_list.append(unmasked_metrics['psnr'])
            Unmasked_SSIM_list.append(unmasked_metrics['ssim'])

            if (i+1) % args.iters_print_acc == 0:
                metrics_baseline = eval_mri(orig, unders_recon)
                if self_val_idx is not None:
                   print('Iteration %05d  Loss: %f  Val loss: %f  baseline_psnr: %f  baseline_ssim: %f  PSRN_gt: %f SSIM_gt: %f  (low-freq)PSNR_gt: %f (low-freq)SSIM_gt: %f (high-freq)PSNR_gt: %f (high-freq)SSIM_gt: %f' % (
                 i,  total_loss.data.cpu(), val_loss.data.cpu(), metrics_baseline['psnr'], metrics_baseline['ssim'], metrics['psnr'], metrics['ssim'], unmasked_metrics['psnr'], unmasked_metrics['ssim'], masked_metrics['psnr'], masked_metrics['ssim']))
                else:
                   print('Iteration %05d  Loss: %f  baseline_psnr: %f  baseline_ssim: %f  PSRN_gt: %f SSIM_gt: %f  (low-freq)PSNR_gt: %f (low-freq)SSIM_gt: %f (high-freq)PSNR_gt: %f (high-freq)SSIM_gt: %f' % (
                 i,  total_loss.data.cpu(), metrics_baseline['psnr'], metrics_baseline['ssim'], metrics['psnr'], metrics['ssim'], unmasked_metrics['psnr'], unmasked_metrics['ssim'], masked_metrics['psnr'], masked_metrics['ssim']))

               # store the image snapshots
                #save_image(f"{snapshots_save_path}/{i+1}_snapshot.png", imgout[np.newaxis,...])
                #save_image(f"{snapshots_save_path}/{i+1}_lowfreq_snapshot.png", unmaskedout[np.newaxis,...])
                #save_image(f"{snapshots_save_path}/{i+1}_highfreq_snapshot.png", maskedout[np.newaxis,...])

            PSNR_list.append(metrics['psnr'])
            SSIM_list.append(metrics['ssim'])
            Loss_list.append(total_loss.data.cpu())
            if self_val_idx is not None:
               val_errors.append(val_loss.data.cpu())             

        if self_val_idx is not None:
           if len(val_errors) > 3*args.window_size:
              if np.mean(val_errors[-args.window_size:]) > np.mean(val_errors[-2*args.window_size:-args.window_size]):
                 print('\nAutomatic early stopping activated.')
                 do_exit = True
                 break

    if (i + 1) % args.num_iters == 0 or do_exit:
        torch.save({'ni': net_input,
                    'net': net.state_dict(),
                    'PSNR_list': PSNR_list,
                    'SSIM_list': SSIM_list,
                    'Loss_list': Loss_list,
                    'Val_Loss_list': val_errors,
                    'Masked_PSNR_list': Masked_PSNR_list,
                    'Masked_SSIM_list': Masked_SSIM_list,
                    'Unmasked_PSNR_list': Unmasked_PSNR_list,
                    'Unmasked_SSIM_list': Unmasked_SSIM_list,
                    'Lips_c_list': Lips_c,  
                    'grads_norm': out_grads,
                    'weights_norm': out_filters,
                    'out_avg': out_avg.data.cpu().numpy()
                    }, f"{experiment_name}/{i+1}th_epoch.pt")
                        
    extra_metrics = {'unmasked_psnr': Unmasked_PSNR_list, 'unmasked_ssim': Unmasked_SSIM_list,
                      'masked_psnr': Masked_PSNR_list, 'masked_ssim': Masked_SSIM_list, 'spectral_norms': norms_list, 'sigmas': singular_list, 'condition_nums': condition_list}

    best_net = net

    bins = 3000
    iter_nums = np.linspace(1, bins, bins)
    sns.set(style="whitegrid")
    plt.grid(axis='x')
    sigma = 6
    sigma2 = 20
    width = 1.5
    a = np.array(stats_ns)
    np.savetxt(f'{experiment_name}/ns.txt', a, fmt='%f')
    plt.plot(gaussian_filter1d(stats_ns[0], sigma=sigma), color='red', linewidth=width)
    plt.plot(gaussian_filter1d(stats_ns[1], sigma=sigma), color='orangered', linewidth=width)
    plt.plot(gaussian_filter1d(stats_ns[2], sigma=sigma), color='darkorange', linewidth=width)
    plt.plot(gaussian_filter1d(stats_ns[3], sigma=sigma), color='orange', linewidth=width)
    plt.plot(gaussian_filter1d(stats_ns[4], sigma=sigma), color='gold', linewidth=width)
    plt.savefig(f"{experiment_name}/plot_ns.png", bbox_inches='tight')
    plt.close()

    return net_input, best_net, extra_metrics, i+1



