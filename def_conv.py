import torch
import torch.nn as nn
import torch.nn.functional as F



class DefConv(nn.Module):
    def __init__(self, inc, outc, kernel_size = 3):
        super(DefConv, self).__init__()
        self.kernel_size = kernel_size
        self.conv_kernel = nn.Conv2d(
            inc, outc, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x, offset):
        
        assert offset.size(1)//2 == self.kernel_size * self.kernel_size 
        x = F.interpolate(x,(x.shape[2]*self.kernel_size,x.shape[3]*self.kernel_size))
        offset = self.reshape_flow(offset)
        x = self.flow_warp(x,offset)
        out = self.conv_kernel(x)
        
        return out


    def flow_warp(self, x, flow, padding_mode='border'):
        """Warp an image or feature map with optical flow
        Args:
            x (Tensor): size (n, c, h, w)
            flow (Tensor): size (n, 2, h, w), values range from -1 to 1
            padding_mode (str): 'zeros' or 'border'

        Returns:
            Tensor: warped image or feature map
        """
        assert x.size()[-2:] == flow.size()[-2:]
        n, _, h, w = x.size()
        x_ = torch.arange(w).view(1, -1).expand(h, -1)
        y_ = torch.arange(h).view(-1, 1).expand(-1, w)
        grid = torch.stack([x_, y_], dim=0).float().cuda()
        grid = grid.unsqueeze(0).expand(n, -1, -1, -1)
        grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
        grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
        grid += 2 * flow
        grid = grid.permute(0, 2, 3, 1)
        return F.grid_sample(x, grid, padding_mode=padding_mode)

    def reshape_flow(self, flow):
        # scale flow field for grid_sample to -1 1
        max_min = flow.max() - flow.min()
        flow = 2.*(flow - flow.min())/max_min-1
        # reshape offset for flow
        flow = flow.reshape(-1,2,flow.shape[2]*self.kernel_size, flow.shape[3]*self.kernel_size)
        return flow
        
