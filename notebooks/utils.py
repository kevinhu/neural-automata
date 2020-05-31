import requests
import io

import numpy as np

from PIL import Image

import torch

def load_image(url, size=40, padding=12):
    
    r = requests.get(url)
    img = Image.open(io.BytesIO(r.content))
    
    img.thumbnail((size, size), Image.ANTIALIAS)
    
    img = np.float32(img)/255.0
    img = np.pad(img,((padding,padding),(padding,padding),(0,0)))
    
    img[..., :3] *= img[..., 3:]
    
    img = torch.Tensor(img).float()
    img = img.transpose(0,2)

    return img


def load_emoji(emoji):
    code = hex(ord(emoji))[2:].lower()
    url = 'https://raw.githubusercontent.com/iamcal/emoji-data/master/img-apple-64/%s.png' % code
    return load_image(url)
