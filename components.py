import torch
import torch.nn as nn

class Squeeze(nn.Module):
    """
    Squeeze operation to increase the number of channels by reducing spatial dimensions.
    """
    def forward(self, x):
        B, C, H, W = x.size()
        assert H % 2 == 0 and W % 2 == 0, "Input width and height must be even"
        x = x.view(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * 4, H // 2, W // 2)
        return x

    def inverse(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C // 4, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(B, C // 4, H * 2, W * 2)
        return x
    
class ActNorm(nn.Module):
    def __init__(self, num_features):
        super(ActNorm, self).__init__()
        self.scale = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.initialized = False

    def forward(self, x):
        if not self.initialized:
            with torch.no_grad():
                # Initialize scale and bias using data statistics
                self.bias.data.copy_(x.mean([0, 2, 3], keepdim=True))
                self.scale.data.copy_(x.std([0, 2, 3], keepdim=True))
            self.initialized = True

        log_det = torch.sum(torch.log(self.scale))
        x = (x - self.bias) / self.scale
        return x, log_det

    def inverse(self, x):
        return x * self.scale + self.bias

class Invertible1x1Conv(nn.Module):
    def __init__(self, num_channels):
        super(Invertible1x1Conv, self).__init__()
        W = torch.linalg.qr(torch.randn(num_channels, num_channels))[0]  # Random orthogonal matrix
        self.W = nn.Parameter(W)

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()

        # Compute the log determinant of the matrix W
        W_log_det = torch.slogdet(self.W)[1] * height * width
        W_inv = torch.inverse(self.W)

        # Apply the linear transformation
        x = x.view(batch_size, num_channels, -1)
        x = torch.matmul(self.W, x)
        x = x.view(batch_size, num_channels, height, width)

        return x, W_log_det

    def inverse(self, x):
        batch_size, num_channels, height, width = x.size()
        W_inv = torch.inverse(self.W)

        x = x.view(batch_size, num_channels, -1)
        x = torch.matmul(W_inv, x)
        x = x.view(batch_size, num_channels, height, width)

        return x

class AffineCoupling(nn.Module):
    def __init__(self, num_channels):
        super(AffineCoupling, self).__init__()
        # The network applies to x1, which has num_channels // 2 channels, and outputs 2 * (num_channels // 2) channels.
        # This is so we can split it into 'log_s' and 't'.
        self.net = nn.Sequential(
            nn.Conv2d(num_channels // 2, num_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        # Split input into two parts along the channel dimension
        x1, x2 = torch.chunk(x, 2, dim=1)  # x1 and x2 both have num_channels // 2 channels
        
        # Apply the network on x1 to get log_s and t
        h = self.net(x1)
        log_s, t = torch.chunk(h, 2, dim=1)  # log_s and t both have num_channels // 2 channels
        
        # Apply the affine transformation to x2
        s = torch.sigmoid(log_s + 2)  # Ensures scaling factor is positive, adding stability
        z2 = s * x2 + t  # Scale and shift x2
        z = torch.cat([x1, z2], dim=1)  # Concatenate x1 and the transformed x2
        
        # Compute the log determinant of the Jacobian (important for the likelihood computation)
        log_det = torch.sum(torch.log(s).view(x.size(0), -1), dim=1)
        
        return z, log_det
    
    def inverse(self, z):
        # Split input z into two parts along the channel dimension
        z1, z2 = torch.chunk(z, 2, dim=1)
        
        # Apply the network to z1 to compute log_s and t
        h = self.net(z1)
        log_s, t = torch.chunk(h, 2, dim=1)
        
        # Perform the inverse of the affine transformation
        s = torch.sigmoid(log_s + 2)
        x2 = (z2 - t) / s  # Invert the scaling and shifting on z2
        x = torch.cat([z1, x2], dim=1)  # Concatenate z1 and the inverted x2
        
        return x
