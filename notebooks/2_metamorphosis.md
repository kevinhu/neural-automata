---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import torch
from torch import nn

import torch.functional as F
import torchvision.transforms.functional as TF
import torch.utils.checkpoint as checkpoint

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import numpy as np
import random

import sys
sys.path.append("../includes")

import models
import utils

from PIL import Image

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
```

# Load images

```python
image_1 = utils.load_emoji("🐞", 32, 16).to(device)
image_2 = utils.load_emoji("🦋", 32, 16).to(device)

plt.imshow(image_1.transpose(0, 2).cpu())
plt.show()
plt.imshow(image_2.transpose(0, 2).cpu())
plt.show()

img_size = 64
```

# Set up model and pool


## Hyperparameters

```python
n_channels = 16
n_epochs = 10000
lr = 0.001
pool_size = 1024
batch_size = 16
hidden_size = 64

images = torch.stack([image_1, image_2])

model = models.Automata((64, 64), n_channels, hidden_size, device).to(device)
```

## Initialize pool

```python
# initialize pool with seeds
seed = torch.zeros(n_channels, img_size, img_size).to(device)
seed[3:, 32, 32] = 1

pool_initials = seed[None, :].repeat(pool_size, 1, 1, 1)
pool_targets = image_1[None, :].repeat(pool_size, 1, 1, 1)

pool_target_ids = torch.zeros(pool_size).long()
# 0 for image_1, 1 for image_2
```

# Train model

```python
losses = []

criterion = nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for i in range(n_epochs):

    iterations = 100

    pool_indices = torch.Tensor(random.sample(
        range(pool_size), batch_size)).long()

    initial_states = pool_initials[pool_indices]
    targets = pool_targets[pool_indices]
    target_ids = pool_target_ids[pool_indices]

    out = model(initial_states, iterations)

    phenotypes = out[:, :4].squeeze()

    optimizer.zero_grad()

    loss = criterion(phenotypes, targets)

    per_sample_loss = loss.mean((1, 2, 3))
    total_loss = per_sample_loss.mean()

    total_loss.backward()
    optimizer.step()

    # argsort the losses per sample
    ranked_loss = per_sample_loss.argsort()

    # get indices of min- and max-loss samples
    min_loss_indices = ranked_loss[:-batch_size//8]
    max_loss_indices = ranked_loss[-batch_size//8:]

    replacements = out.detach()
    replacements[max_loss_indices] = seed.clone()

    # high-loss outputs are re-tasked with
    # mapping the seed to the first image,

    # low-loss outputs are tasked with mapping
    # mapping to the other image
    pool_target_ids[pool_indices[max_loss_indices]] = 0
    pool_target_ids[pool_indices[min_loss_indices]] = 1 - \
        pool_target_ids[pool_indices[min_loss_indices]]

    pool_targets[pool_indices[max_loss_indices]] = images[0]
    pool_targets[pool_indices[min_loss_indices]
                 ] = images[pool_target_ids[pool_indices[min_loss_indices]]]

    pool_initials[pool_indices] = replacements

    if i % 100 == 0:

        print(i, np.log10(float(total_loss.cpu().detach())))

        torch.save(model.state_dict(), "../models/metamorphosis_"+str(i))

    losses.append(float(total_loss))
```

```python
plt.plot(np.log10(losses))
```

# Load model checkpoint

```python
model.load_state_dict(torch.load(
    "../models/metamorphosis_9900", map_location=torch.device('cpu')))
```

```python
video = utils.get_model_history(model, seed, 1000)
utils.channels_to_gif("../videos/metamorphosis_channels.gif", video)
utils.colors_to_gif("../videos/metamorphosis_colors.gif", video)
```
