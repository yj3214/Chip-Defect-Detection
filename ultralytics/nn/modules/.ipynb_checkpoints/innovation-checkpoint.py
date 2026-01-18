import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import conv2d

def build_norm_layer(norm_layer, num_features):
    if norm_layer == 'BN':
        return None, nn.BatchNorm2d(num_features)
    elif norm_layer == 'LN':
        return None, nn.LayerNorm(num_features)
    else:
        raise NotImplementedError(f"Normalization {norm_layer} not supported")

class LearnableDoGFilter(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=7, init_sigma=1.0, init_k=1.6,
                 norm_layer='BN', act_layer=nn.ReLU, min_sigma=0.5, max_sigma=8.0):
        """
        Learnable Difference of Gaussians (DoG) filter.
        
        Args:
            in_c (int): input channels
            out_c (int): output channels
            kernel_size (int): fixed kernel size (must be odd)
            init_sigma (float): initial value for sigma1
            init_k (float): initial ratio k = sigma2 / sigma1 (>1)
            norm_layer (str): normalization type
            act_layer (nn.Module): activation
            min_sigma, max_sigma (float): bounds for sigma values
        """
        super(LearnableDoGFilter, self).__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.kernel_size = kernel_size
        self.out_c = out_c
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

        # Initial feature projection
        self.conv_init = nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3)

        # Learnable parameters: log(sigma1) and log(k) to ensure positivity
        self.log_sigma1 = nn.Parameter(torch.tensor(math.log(init_sigma), dtype=torch.float32))
        self.log_k = nn.Parameter(torch.tensor(math.log(init_k), dtype=torch.float32))

        # Coordinate grid (fixed, register as buffer)
        ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        self.register_buffer('xx', xx)
        self.register_buffer('yy', yy)

        # Post-processing layers
        self.act = act_layer()
        self.norm1 = build_norm_layer(norm_layer, out_c)[1]
        self.norm2 = build_norm_layer(norm_layer, out_c)[1]
        
        # downsample
        self.down = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=False),
        )

    def _get_dog_kernel(self):
        # Enforce bounds via sigmoid or clamp (we use exp + clamp for stability)
        sigma1 = torch.exp(self.log_sigma1).clamp(self.min_sigma, self.max_sigma)
        k = torch.exp(self.log_k).clamp(1.01, 5.0)  # ensure k > 1
        sigma2 = (sigma1 * k).clamp(self.min_sigma, self.max_sigma)

        # Compute Gaussian kernels
        gauss1 = torch.exp(-(self.xx**2 + self.yy**2) / (2 * sigma1**2 + 1e-8))
        gauss1 = gauss1 / (gauss1.sum() + 1e-8)

        gauss2 = torch.exp(-(self.xx**2 + self.yy**2) / (2 * sigma2**2 + 1e-8))
        gauss2 = gauss2 / (gauss2.sum() + 1e-8)

        # DoG = G1 - G2
        dog = gauss1 - gauss2
        dog = dog - dog.mean()  # zero-center (optional but recommended)

        # Normalize by L1 norm for stable magnitude
        dog = dog / (dog.abs().sum() + 1e-8)

        # Expand to [out_c, 1, K, K]
        dog = dog.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
        dog = dog.repeat(self.out_c, 1, 1, 1)  # [out_c, 1, K, K]
        return dog

    def forward(self, x):
        x = self.conv_init(x)  # [B, out_c, H, W]

        # Dynamically generate DoG kernel
        dog_kernel = self._get_dog_kernel()

        # Apply depth-wise convolution with current kernel
        # We use functional conv2d to avoid storing weight
        dog_response = conv2d(
            input=x, 
            weight=dog_kernel, 
            bias=None, 
            stride=1, 
            padding=self.kernel_size // 2,
            groups=self.out_c
        )

        dog_edge = self.act(self.norm1(dog_response))
        x = self.norm2(x + dog_edge)
        x = self.down(x)
        return x

    def get_sigmas(self):
        """Utility to inspect learned sigmas"""
        sigma1 = torch.exp(self.log_sigma1).item()
        k = torch.exp(self.log_k).item()
        sigma2 = sigma1 * k
        return sigma1, sigma2

class Gaussian(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=7, sigma=0.5, norm_type='BN', act_type='ReLU'):
        """
        Pure torch implementation of Gaussian Filter (no external dependencies).
        
        Args:
            dim (int): number of input/output channels
            kernel_size (int): size of Gaussian kernel (must be odd)
            sigma (float): standard deviation of Gaussian
            norm_type (str): 'BN' for BatchNorm2d, 'GN' for GroupNorm, 'None' for no norm
            act_type (str): 'ReLU', 'GELU', 'SiLU', 'None'
        """
        super(Gaussian, self).__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"

        # Generate fixed Gaussian kernel
        gaussian_kernel = self._create_gaussian_kernel(kernel_size, sigma)
        # Expand to [dim, 1, K, K] for depth-wise convolution
        gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
        gaussian_kernel = gaussian_kernel.repeat(dim, 1, 1, 1)      # [dim, 1, K, K]

        # Create depth-wise conv with fixed weight
        self.gaussian_conv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=dim,
            bias=False
        )
        self.gaussian_conv.weight.data = gaussian_kernel
        self.gaussian_conv.weight.requires_grad = True

        # Build normalization layer
        if norm_type == 'BN':
            self.norm = nn.BatchNorm2d(dim)
        elif norm_type == 'GN':
            self.norm = nn.GroupNorm(num_groups=2, num_channels=dim)  # 2 groups
        elif norm_type == 'None':
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        # Build activation layer
        if act_type == 'ReLU':
            self.act = nn.ReLU(inplace=False)
        elif act_type == 'GELU':
            self.act = nn.GELU()
        elif act_type == 'SiLU':
            self.act = nn.SiLU(inplace=False)
        elif act_type == 'None':
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unknown act_type: {act_type}")
        
    def _create_gaussian_kernel(self, size: int, sigma: float):
        """Create a normalized 2D Gaussian kernel."""
        ax = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()  # Normalize
        return kernel

    def forward(self, x):
        """
        Forward pass: Apply Gaussian smoothing → Norm → Activation.
        
        Args:
            x: Tensor of shape [B, C, H, W]
        
        Returns:
            Tensor of shape [B, C, H, W]
        """
        x_smooth = self.gaussian_conv(x)     # Depth-wise Gaussian smoothing
        x_norm = self.norm(x_smooth)         # Normalization
        x_act = self.act(x_norm)             # Activation
        return x_act

class DownSample(nn.Module):
    def __init__(self, dim, dim_out):
        super(DownSample, self).__init__()
        # downsample
        self.down = nn.Sequential(
            nn.Conv2d(dim, dim_out, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        
        return self.down(x)

class GaussianFilter(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=7, sigma=0.5, norm_type='BN', act_type='ReLU'):
        super(GaussianFilter, self).__init__()
        self.gaussian = Gaussian(dim, dim_out, kernel_size=7, sigma=0.5, norm_type='BN', act_type='ReLU')
        self.down = DownSample(dim, dim_out)
    
    def forward(self, x):
        x_gaussian = self.gaussian(x)
        x_out = self.down(x + x_gaussian)
        return x_out