import requests
import io

import numpy as np

from PIL import Image
import imageio
import imageio_ffmpeg

import torch


def load_image(url, size, padding):
    """
    Load an image in RGBA format from a URL, and
    apply transparent padding, returning as a
    PyTorch tensor

    Parameters
    ----------
    url: string
        source URL of the image
    size: (int, int)
        (rows, columns) size of the image to resize
    padding: int
        padding to apply equally around image

    Returns
    -------
    img: image in (channels, rows, columns) format
    """

    r = requests.get(url)
    img = Image.open(io.BytesIO(r.content))

    img = img.resize((size, size), Image.ANTIALIAS)

    # normalize image intensities to (0,1) range
    img = np.float32(img) / 255.0
    img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)))

    # multiply intensities by alphas
    img[..., :3] *= img[..., 3:]

    img = torch.Tensor(img).float()

    # move channels to first axis
    img = img.transpose(0, 2)

    return img


def load_emoji(emoji, size, padding):
    """
    Loads the image tensor of an emoji
    (in Apple emoji format)

    Parameters
    ----------
    emoji: string
        the emoji to load
    size: (int, int)
        (rows, columns) size of the emoji to resize
    padding: int
        padding to apply equally around emoji

    Returns
    -------
    img: emoji in (channels, rows, columns) format
    """

    code = hex(ord(emoji))[2:].lower()
    print(code)
    url = (
        "https://github.com/samuelngs/apple-emoji-linux/raw/master/png/128/emoji_u%s.png"
        % code
    )
    return load_image(url, size, padding)


def normalize(a):
    """
    Normalize an array to the [0,1] range

    Parameters
    ----------
    a: array
        array to normalize

    Returns
    -------
    a: normalized array
    """

    a = a - a.min()
    a = a / a.max()

    return a


def get_model_history(model, seed_state, iterations):
    """
    Get the iterations of an automata model from a seed state

    Parameters
    ----------
    model: Automata
        model to run
    seed_state: tensor
                initial state, of shape (channels, rows, columns)
        iterations: int
                number of iterations to execute

    Returns
    -------
    video: the iterations, in shape (iterations, rows, columns, channels)
    """

    with torch.no_grad():
        out = model(seed_state[None, :], iterations, keep_history=True)
        video = model.history.cpu().detach()
        video = video[:, 0]
        video = video.transpose(1, 3).numpy()

    return video


def channels_to_gif(output_path, video, fps=60, row_channels=4, col_channels=4):
    """
    Save a GIF of the channels over time, given the output
    of `get_model_history`

    Parameters
    ----------
    output_path: string
        path to save the GIF to
    video: array
        output of `get_model_history`
    fps: int
        frames per second of output GIF
    row_channels: int
        number of channels to place on each row in the GIF
    col_channels: int
        number of channels to place on each column in the GIF
    """

    iterations = video.shape[0]
    channel_count = video.shape[-1]
    grid_size = (video.shape[1], video.shape[2])

    if row_channels * col_channels != channel_count:
        raise ValueError("Row-column channel product must equal total channel count")

    tiled = np.zeros(
        (iterations, grid_size[0] * col_channels, grid_size[1] * row_channels),
        dtype=np.uint8,
    )

    k = 0

    for i in range(col_channels):
        for j in range(row_channels):

            patch = normalize(video[:, :, :, k])

            patch[video[:, :, :, 3] == 0] = 0
            patch = 1 - patch
            patch = patch * 255
            patch = patch.astype(np.uint8)

            tiled[
                :,
                grid_size[0] * i : grid_size[0] * (i + 1),
                grid_size[1] * j : grid_size[1] * (j + 1),
            ] = patch

            k += 1

    tiled = tiled[:, :, :, None].repeat(3, axis=3)

    tiled = [Image.fromarray(x, "RGB") for x in tiled]

    imageio.mimwrite(output_path, tiled, fps=fps)


def colors_to_gif(output_path, video, fps=60):
    """
    Save a GIF of the first 4 channels interpreted as RGBA
    given the output of `get_model_history`

    Parameters
    ----------
    output_path: string
        path to save the GIF to
    video: array
        output of `get_model_history`
    fps: int
        frames per second of output GIF
    """

    video = video[:, :, :, :4]
    video = video * 255
    video = video.astype(np.uint8)

    video = video.repeat(4, axis=1).repeat(4, axis=2)

    background = Image.new("RGBA", (video.shape[1], video.shape[2]), (255, 255, 255))

    video = [Image.fromarray(x, "RGBA") for x in video]
    video = [Image.alpha_composite(background, x) for x in video]

    imageio.mimwrite(output_path, video, fps=60)
