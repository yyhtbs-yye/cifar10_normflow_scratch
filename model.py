from components import ActNorm, Invertible1x1Conv, AffineCoupling, Squeeze

import torch
import torch.nn as nn

class Glow(nn.Module):
    def __init__(self, num_channels, num_levels, num_steps):
        super(Glow, self).__init__()
        self.num_levels = num_levels
        self.num_steps = num_steps

        self.flows = nn.ModuleList()
        for _ in range(num_levels):
            # After all steps, apply squeeze to double channels and halve spatial dimensions
            if _ < num_levels - 1:  # Only apply squeeze if not the last level
                self.flows.append(Squeeze())
                num_channels *= 4  # After squeeze, num_channels are multiplied by 4

            steps = nn.ModuleList()
            # Add flow steps (ActNorm, Invertible1x1Conv, AffineCoupling) to this level
            for _ in range(num_steps):
                steps.append(ActNorm(num_channels))
                steps.append(Invertible1x1Conv(num_channels))
                steps.append(AffineCoupling(num_channels))

            self.flows.append(steps)


    def forward(self, x):
        batch_size = x.size(0)

        log_det_total = torch.zeros(batch_size, device=x.device)  # Initialize log_det_total as a tensor of zeros

        z = x

        for steps in self.flows:
            if isinstance(steps, Squeeze):
                # Apply squeeze at the end of each level
                z = steps(z)
            else:
                for step in steps:
                    z, log_det = step(z)
                    log_det_total += log_det

        return z, log_det_total

    def inverse(self, z):
        x = z
        for steps in reversed(self.flows):
            if isinstance(steps, Squeeze):
                # Apply inverse squeeze when reversing the flow
                x = steps.inverse(x)
            else:
                for step in reversed(steps):
                    x = step.inverse(x)
        return x
