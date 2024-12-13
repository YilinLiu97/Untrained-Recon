import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.variable as Variable

from utils.common_utils import *

# version adaptation for PyTorch > 1.7.1
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft
dtype = torch.cuda.FloatTensor

def loss_func(loss_type, A, B):
    if loss_type == 'l1':
       loss = F.l1_loss(A,B)
    elif loss_type == 'complex_l1':
       assert A.shape[-1] == B.shape[-1] == 2
       A, B = torch.view_as_complex(A), torch.view_as_complex(B)
       loss = F.l1_loss(A,B)
       # estimate m0
       complex_cosine = (A*B).sum(1)/((torch.linalg.norm(A,dim=1) + 1e-12) * (torch.linalg.norm(B,dim=1)) + 1e-12)
       dictnorm, signalnorm = torch.linalg.norm(A,dim=1), torch.linalg.norm(B,dim=1)
       m0_est = torch.conj((complex_cosine*signalnorm) / (dictnorm + 1e-12))
       m0_est = m0_est.view(-1,256,256)
       m0_est = torch.stack((m0_est.real, m0_est.imag), -1)
       return loss, m0_est
    elif loss_type == 'mse':
       loss = F.mse_loss(A,B)
    elif loss_type == 'cosine_abs':
       loss = -F.cosine_similarity(A,B).abs().mean()
    elif loss_type == 'arc_cosine':
       assert A.shape[-1] == B.shape[-1] == 2
       M, N = A.shape[-3:-1]
       A, B = torch.view_as_complex(A), torch.view_as_complex(B) # (1,tps,256,256)
       A, B = A.view(A.shape[0], A.shape[1], -1), B.view(B.shape[0], B.shape[1], -1)

#       print(f"A: {A.abs().sum()}, B: {B.abs().sum()}")
       complex_cosine = (A.conj()*B).sum(1)/(torch.max(torch.linalg.norm(A,dim=1) * torch.linalg.norm(B,dim=1), torch.tensor(1e-12)))
       loss = (torch.arccos(complex_cosine.abs()) / torch.pi).mean()
       
       dictnorm, signalnorm = torch.linalg.norm(A,dim=1), torch.linalg.norm(B,dim=1)

       m0_est = torch.conj((complex_cosine*signalnorm) / (dictnorm + 1e-12))
       m0_est = m0_est.view(-1,M,N)
       m0_est = torch.stack((m0_est.real, m0_est.imag), -1)

       return loss, m0_est
       '''
       cos_sim = F.cosine_similarity(A,B)
       print(f"cos_sim: {cos_sim.shape}")
       loss = -cos_sim.mean()
       '''
    elif loss_type == 'vec_cosine':
       assert A.shape[-1] == B.shape[-1] == 2
       M, N = A.shape[-3:-1]
       A, B = torch.view_as_complex(A), torch.view_as_complex(B) # (1,tps,256,256)
       A, B = A.view(A.shape[0], A.shape[1], -1), B.view(B.shape[0], B.shape[1], -1)

#       print(f"A: {A.abs().sum()}, B: {B.abs().sum()}")
       complex_cosine = (A.conj()*B).sum(1)/(torch.max(torch.linalg.norm(A,dim=1) * torch.linalg.norm(B,dim=1), torch.tensor(1e-12)))
       loss = complex_cosine.abs().mean()
       
       dictnorm, signalnorm = torch.linalg.norm(A,dim=1), torch.linalg.norm(B,dim=1)

       m0_est = torch.conj((complex_cosine*signalnorm) / (dictnorm + 1e-12))
       m0_est = m0_est.view(-1,M,N)
       m0_est = torch.stack((m0_est.real, m0_est.imag), -1)

       return loss, m0_est
    elif loss_type == 'complex_l2':
       real_out, real_gt = A[...,0], B[...,0]
       imag_out, imag_gt = A[...,1], B[...,1]
       diff = real_out*imag_gt - real_gt*imag_out
       complex_gt = torch.view_as_complex(B)
       loss = (diff.abs()/(complex_gt.abs() + 1e-12)).mean()
    elif loss_type == 'bce':
       loss = F.binary_cross_entropy(A,B)
    elif loss_type == 'cosine_1':
       # assert A.shape[-1] == B.shape[-1] == 2
       # A, B = torch.view_as_complex(A), torch.view_as_complex(B) # (1,tps,256,256,2)
       A_r, A_i = A[..., 0].view(A.shape[0], A.shape[1], -1), A[..., 1].view(A.shape[0], A.shape[1], -1)
       B_r, B_i = B[..., 0].view(B.shape[0], B.shape[1], -1), B[..., 1].view(B.shape[0], B.shape[1], -1)
       
       r_cosine = (A_r * 1e10 * B_r * 1e10).sum(1)/((torch.linalg.norm(A_r * 1e10,dim=1) + 1e-30) * (torch.linalg.norm(B_r * 1e10,dim=1)) + 1e-30)
       i_cosine = (A_i * B_i).sum(1)/((torch.linalg.norm(A_i,dim=1) + 1e-30) * (torch.linalg.norm(B_i,dim=1)) + 1e-30)
       # i_cosine = (i_cosine + 1 + 1e-100) / 2
       loss = torch.mean(2.0 - r_cosine - i_cosine)
       M, N = A.shape[-3:-1]
       m0_est = torch.zeros(1, M, N, 2).to(A.device)
       
       return loss, m0_est
    else:
       raise NotImplementedError("No such loss type.")
    return loss

'''
def get_distribution_target(mode='gaussian', target_mean=0.3, length=258):
    if mode == 'gaussian':
       from scipy.stats import norm
       data = np.arange(length)
'''       

def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
    tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
    return tv_weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

def tv_loss2(img, tv_weight):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss

def orthogology_loss(mat):
    loss = 0.0
    if mat.requires_grad:
        if mat.size(0)<=mat.size(1):
            mulmat = mat.matmul(mat.transpose(0,1)) #AxA'
        else:
            mulmat = mat.transpose(0,1).matmul(mat) #A'xA
        loss = torch.sum((mulmat-torch.eye(mulmat.size(0)).type(dtype))**2)/(mulmat.size(0)*mulmat.size(1))
    return loss

def singular_reg_loss(param, reg_type=3):
    """
    Regularization for single parameter
    """
    reg = 0.0
    if param.requires_grad and torch.sum(torch.abs(param))>0:
        if reg_type == 2:
            reg = torch.sum(torch.abs(param))/torch.sqrt(torch.sum(param**2))-1 #Hoyer
        elif reg_type == 3:
            reg = (torch.sum(torch.abs(param))**2)/torch.sum(param**2)-1 #Hoyer-Square
        elif reg_type == 1:
            reg = torch.sum(torch.abs(param)) #L1
        else:
            reg = 0.0
    return reg


class EHMLoss(nn.Module):
    def __init__(self):
       super(EHMLoss, self).__init__()
       self.mse = nn.MSELoss()

    def forward(self, psdA, psdB):
       transferred_A = self.exact_feature_distribution_matching(psdA, psdB)
       loss = self.mse(psdA, transferred_A)
       return loss

    def exact_feature_distribution_matching(self, psdA, psdB):
       """
       psdA: output psd 1D
       psdB: target psd 1D
       """
       assert (psdA.size() == psdB.size()) ## content and style features should share the same shape
       B, C = psdA.size(0), psdA.size(1)
       _, index_psdA = torch.sort(psdA)  ## sort content feature
       value_psdB, _ = torch.sort(psdB)      ## sort style feature
       inverse_index = index_psdA.argsort(-1)
       transferred_psdA = psdA + value_psdB.gather(-1, inverse_index) - psdA.detach()
       return Variable(transferred_psdA, requires_grad=True)

class FocalFrequencyLoss(nn.Module):
    def __init__(self, alpha=1.0, ave_spectrum=False, log_matrix=False):
       super(FocalFrequencyLoss, self).__init__()
       self.alpha = alpha
       self.ave_spectrum = ave_spectrum
       self.log_matrix = log_matrix

    def tensor2freq(self, x):
        # perform 2D DFT (real-to-complex, orthonormalization)
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(x, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(x, 2, onesided=False, normalized=True)
        return freq

    def loss_formulation(self, freq_rec, freq_target, matrix=None):
        if matrix is not None:
            weight_matrix = matrix.detach()
        else:
            # the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (freq_rec - freq_target) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha # RMSE * alpha

            if self.log_matrix:
                matrix_tmp = matrix_tmp / torch.log(matrix_tmp + 1.0)

            matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :,  None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
                'The values of spectrum weight matrix should be in the range [0, 1], '
                'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (freq_rec - freq_target) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None):
        """Forward function to calculate focal frequency loss.
                Args:
                    pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
                    target (torch.Tensor): of shape (N, C, H, W). Target tensor.
                    matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                        Default: None (If set to None: calculated online, dynamic).
                """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) 

class Freq_Statistics_Matching(nn.Module):
    def __init__(self):
        super(Freq_Statistics_Matching, self).__init__()
        self.mse = nn.MSELoss()

    def cal_stats(self, psd1d):
        mu = torch.mean(psd1d)
        diffs = psd1d - mu
        var = torch.mean(torch.pow(diffs, 2.0))
        std = torch.pow(var, 0.5)
        zscores = diffs / std
        skews = torch.mean(torch.pow(zscores, 3.0))
        kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0
        return mu, std, skews, kurtoses

    def cal_mag(self, image):
        return fft_mag(image, log=True)

    def forward(self, output, target):
        assert output.size() == target.size()
        mag2d_rec, mag2d_target = self.cal_mag(output), self.cal_mag(target)
        mu_rec, mu_target = mag2d_rec.mean(), mag2d_target.mean()
        std_rec, std_target = mag2d_rec.std(), mag2d_target.std()
#        psd2d_rec, psd1d_rec = get_psd(output, log=True)
#        psd2d_target, psd1d_target = get_psd(target, log=True)
        #output_mu, output_std, output_skew, output_kurt = self.cal_stats(psd1d_rec)
        #target_mu, target_std, target_skew, target_kurt = self.cal_stats(psd1d_target)
#        print(f"mse(mu): {self.mse(output_mu, target_mu)} mse(std): {self.mse(output_std, target_std)} skew: {output_skew**2} kurt: {output_kurt**2}")
#        print(self.mse(psd1d_rec, psd1d_target))
        return self.mse(mu_rec, mu_target) + \
            self.mse(std_rec, std_target) #+ \
            #output_skew ** 2 + \
            #output_kurt ** 2


