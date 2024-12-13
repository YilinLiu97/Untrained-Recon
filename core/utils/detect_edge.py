# Code  borrowed from https://github.com/DCurro/CannyEdgePytorch

import torch
import torch.nn as nn
import numpy as np
from scipy.signal.windows import gaussian


class CannyNet(nn.Module):
    def __init__(self, device, threshold=400.0, use_cuda=True, requires_grad=False):
        super(CannyNet, self).__init__()

        self.threshold = threshold
        self.use_cuda = use_cuda
        self.device = device

        filter_size = 5
        generated_filters = gaussian(filter_size, std=1.0).reshape([1, filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, filter_size),
                                                    padding=(0, filter_size // 2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size, 1),
                                                  padding=(filter_size // 2, 0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape,
                                                 padding=sobel_filter.shape[0] // 2)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape,
                                               padding=sobel_filter.shape[0] // 2)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        # filters were flipped manually
        filter_0 = np.array([[0, 0, 0],
                             [0, 1, -1],
                             [0, 0, 0]])

        filter_45 = np.array([[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, -1]])

        filter_90 = np.array([[0, 0, 0],
                              [0, 1, 0],
                              [0, -1, 0]])

        filter_135 = np.array([[0, 0, 0],
                               [0, 1, 0],
                               [-1, 0, 0]])

        filter_180 = np.array([[0, 0, 0],
                               [-1, 1, 0],
                               [0, 0, 0]])

        filter_225 = np.array([[-1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 0]])

        filter_270 = np.array([[0, -1, 0],
                               [0, 1, 0],
                               [0, 0, 0]])

        filter_315 = np.array([[0, 0, -1],
                               [0, 1, 0],
                               [0, 0, 0]])

        all_filters = np.stack(
            [filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape,
                                            padding=filter_0.shape[-1] // 2)
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
        self.directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))))

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, img, threshold=400.0):
        thresh_list = []
        for i in range(1, img.shape[1] + 1):
            img_t = (img[:, i - 1:i, :, :] + 1) * 127.5

            blur_horizontal = self.gaussian_filter_horizontal(img_t)
            blurred_img_t = self.gaussian_filter_vertical(blur_horizontal)

            grad_x_t = self.sobel_filter_horizontal(blurred_img_t)
            grad_y_t = self.sobel_filter_vertical(blurred_img_t)

            # COMPUTE THICK EDGES
            grad_mag = torch.sqrt(grad_x_t ** 2 + grad_y_t ** 2)
            grad_ori_y = grad_y_t
            grad_ori_x = grad_x_t
            grad_orientation = (torch.atan2(grad_ori_y, grad_ori_x) * (180.0 / 3.14159))
            grad_orientation += 180.0
            grad_orientation = torch.round(grad_orientation / 45.0) * 45.0

            # THIN EDGES (NON-MAX SUPPRESSION)
            all_filtered = self.directional_filter(grad_mag)

            inidices_positive = (grad_orientation / 45) % 8
            inidices_negative = ((grad_orientation / 45) + 4) % 8

            batch = inidices_positive.size()[0]
            height = inidices_positive.size()[2]
            width = inidices_positive.size()[3]
            pixel_count = height * width * batch
            if self.use_cuda:
                pixel_range = torch.cuda.FloatTensor([range(pixel_count)])
                pixel_range = pixel_range.to(self.device)
            else:
                pixel_range = torch.FloatTensor([range(pixel_count)])

            indices = (inidices_positive.view(-1).data * pixel_count + pixel_range).squeeze()
            channel_select_filtered_positive = all_filtered.view(-1)[indices.long()].view(batch, 1, height, width)

            indices = (inidices_negative.view(-1).data * pixel_count + pixel_range).squeeze()
            channel_select_filtered_negative = all_filtered.view(-1)[indices.long()].view(batch, 1, height, width)

            channel_select_filtered = torch.stack([channel_select_filtered_positive, channel_select_filtered_negative])

            is_max = channel_select_filtered.min(dim=0)[0] > 0.0

            thin_edges = grad_mag.clone()
            thin_edges[is_max == 0] = 0.0

            # THRESHOLD
            thresholded = thin_edges.clone()
            thresholded[thin_edges < threshold] = 0.0

            thresholded = (thresholded > 0.0).float()
            thresh_list.append(thresholded)

        thresholded_list = torch.cat(thresh_list, dim=1)
        thresholded_list = (thresholded_list - thresholded_list.min()) / \
                           (thresholded_list.max() - thresholded_list.min())

        return thresholded_list


