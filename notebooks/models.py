import torch
from torch import nn

import torch.functional as F
import torchvision.transforms.functional as TF
import torch.utils.checkpoint as checkpoint

class Automata(nn.Module):

    def __init__(self, grid_size, n_channels, hidden_size, device):

        super(Automata, self).__init__()

        self.grid_size = grid_size
        self.n_channels = n_channels
        self.hidden_size = hidden_size

        self.filters = torch.Tensor([[[[-1, 0, 1],
                                       [-2, 0, 2],
                                       [-1, 0, 1]
                                       ]],
                                     [[[-1, -2, -1],
                                       [0, 0, 0],
                                       [1, 2, 1]
                                       ]],
                                     [[[0, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 0]
                                       ]]]).to(device)

        self.mapper = nn.Sequential(
            nn.Linear(3*n_channels, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_channels),
            nn.Tanh()
        )

        self.mapper[2].weight.data.fill_(0)
        self.mapper[2].bias.data.fill_(0)

    def perception(self, x):

        # reshape for same convolution across channels
        x = x.reshape(-1, 1, self.grid_size[0], self.grid_size[1])

        # toroidal padding
        conved = nn.functional.pad(x, (1, 1, 1, 1), mode="circular")

        conved = nn.functional.conv2d(conved, self.filters)

        # reshape for perception computations
        conved = conved.view(self.batch_size, 3*self.n_channels, -1)
        conved = conved.transpose(1, 2)

        conved = self.mapper(conved)

        conved = conved.transpose(1, 2)
        # (batch_size, channels, total_cells)

        conved = conved.view(self.batch_size, self.n_channels, *self.grid_size)

        can_update = torch.rand_like(conved) < 0.5

        return conved*can_update

    def forward(self, x, iterations, keep_history=False):

        if keep_history:

            self.history = torch.zeros(iterations, *x.shape)

        self.batch_size = x.shape[0]
        
        x.requires_grad = True

        for i in range(iterations):

            next_to_alive = nn.functional.max_pool2d(
                x[:, 3], (3, 3), stride=1, padding=1) > 0.1

            next_to_alive = next_to_alive.unsqueeze(1)
            
            x = x + checkpoint.checkpoint(self.perception,x)
            x = x*next_to_alive

            x[:, :4].clamp_(0, 1)

            if keep_history:

                self.history[i] = x.detach()

        return x