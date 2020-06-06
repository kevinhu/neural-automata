import requests
import io

import numpy as np

from PIL import Image
import imageio
import imageio_ffmpeg

import torch


def load_image(url, size, padding):

    r = requests.get(url)
    img = Image.open(io.BytesIO(r.content))

    img = img.resize((size, size), Image.ANTIALIAS)

    img = np.float32(img)/255.0
    img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)))

    img[..., :3] *= img[..., 3:]

    img = torch.Tensor(img).float()
    img = img.transpose(0, 2)

    return img


def load_emoji(emoji, size, padding):
    code = hex(ord(emoji))[2:].lower()
    print(code)
    url = 'https://github.com/samuelngs/apple-emoji-linux/raw/master/png/128/emoji_u%s.png' % code
    return load_image(url, size, padding)


def normalize(a):

    a = a-a.min()
    a = a/a.max()

    return a


def get_model_history(model, seed_state, iterations):

    with torch.no_grad():
        out = model(seed_state[None, :], iterations, keep_history=True)
        video = model.history.cpu().detach()
        video = video[:, 0]
        video = video.transpose(1, 3).numpy()

    return video


def channels_to_gif(output_path,
                    video,
                    fps=60,
                    grid_size=(64, 64),
                    row_channels=4,
                    col_channels=4):

    iterations = video.shape[0]
    channel_count = video.shape[-1]

    assert row_channels * \
        col_channels == channel_count, "Row-column channel product must equal total channel count"

    tiled = np.zeros(
        (iterations, grid_size[0]*4, grid_size[1]*4), dtype=np.uint8)

    k = 0

    for i in range(row_channels):
        for j in range(col_channels):

            patch = normalize(video[:, :, :, k])

            patch[video[:, :, :, 3] == 0] = 0
            patch = 1-patch
            patch = patch*255
            patch = patch.astype(np.uint8)

            tiled[:, grid_size[0]*i:grid_size[0] *
                  (i+1), grid_size[1]*j:grid_size[1]*(j+1)] = patch

            k += 1

    tiled = tiled[:, :, :, None].repeat(3, axis=3)

    tiled = [Image.fromarray(x, 'RGB') for x in tiled]

    imageio.mimwrite(output_path, tiled, fps=fps)


def colors_to_gif(output_path,
                  video,
                  fps=60,
                  grid_size=(64, 64)):

    video = video[:, :, :, :4]
    video = video*255
    video = video.astype(np.uint8)

    video = video.repeat(4, axis=1).repeat(4, axis=2)

    background = Image.new(
        'RGBA', (grid_size[0]*4, grid_size[1]*4), (255, 255, 255))

    video = [Image.fromarray(x, 'RGBA') for x in video]
    video = [Image.alpha_composite(background, x) for x in video]

    imageio.mimwrite(output_path, video, fps=60)
