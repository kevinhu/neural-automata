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

import imageio
import imageio_ffmpeg

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

model = models.Automata((64, 64), n_channels, hidden_size, device).to(device)
```

## Initialize pool

```python
images = torch.stack([image_1, image_2])

# initialize pool with seeds
seed = torch.zeros(n_channels, img_size, img_size).to(device)
seed[3:, 32, 32] = 1

seed_1 = seed.clone()
seed_2 = seed.clone()

seed_1[4, 32, 32] = 0

seeds = torch.stack([seed_1, seed_2])

pool_initials = seeds.repeat(pool_size//2, 1, 1, 1)
pool_targets = images.repeat(pool_size//2, 1, 1, 1)

pool_target_ids = torch.Tensor([0, 1]).repeat(pool_size//2).long()
# 0 for image_1, 1 for image_2
# half image 1, half image 2
```

# Train model

```python
losses = []

criterion = nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for i in range(n_epochs):

    iterations = random.randint(96, 128)

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
    min_loss_indices = ranked_loss[:1]
    max_loss_indices = ranked_loss[1:]

    replacements = out.detach()

    max_loss_targets = target_ids[max_loss_indices]

    # high-loss outputs are re-tasked with
    # mapping the seeds to the respective image

    # low-loss outputs are tasked with mapping
    # the previous output to the same image

    replacements[max_loss_indices] = seeds[max_loss_targets]
    pool_initials[pool_indices] = replacements

    if i % 100 == 0:

        print(i, np.log10(float(total_loss.cpu().detach())))

        torch.save(model.state_dict(), "../models/divergence_"+str(i))

    losses.append(float(total_loss))
```

```python
plt.plot(np.log10(losses))
```

# Load model checkpoint

```python
model.load_state_dict(torch.load(
    "../models/divergence_10000", map_location=torch.device('cpu')))
```

```python
seed_mix = seed.clone()
seed_mix[4, 32, 32] = 0.5
```

```python
video = utils.get_model_history(model, seed_1, 128)
utils.channels_to_gif("../videos/divergence_seed_1_channels.gif", video)
utils.colors_to_gif("../videos/divergence_seed_1_colors.gif", video)

video = utils.get_model_history(model, seed_2, 128)
utils.channels_to_gif("../videos/divergence_seed_2_channels.gif", video)
utils.colors_to_gif("../videos/divergence_seed_2_colors.gif", video)

video = utils.get_model_history(model, seed_mix, 128)
utils.channels_to_gif("../videos/divergence_seed_mix_channels.gif", video)
utils.colors_to_gif("../videos/divergence_seed_mix_colors.gif", video)
```
