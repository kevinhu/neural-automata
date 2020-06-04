import requests
import io

import numpy as np

from PIL import Image

import torch

def load_image(url, size, padding):
    
    r = requests.get(url)
    img = Image.open(io.BytesIO(r.content))
    
    img = img.resize((size, size), Image.ANTIALIAS)
    
    img = np.float32(img)/255.0
    img = np.pad(img,((padding,padding),(padding,padding),(0,0)))
    
    img[..., :3] *= img[..., 3:]
    
    img = torch.Tensor(img).float()
    img = img.transpose(0,2)

    return img


def load_emoji(emoji, size, padding):
    code = hex(ord(emoji))[2:].lower()
    print(code)
    url = 'https://github.com/samuelngs/apple-emoji-linux/raw/master/png/128/emoji_u%s.png' % code
    return load_image(url,size,padding)
