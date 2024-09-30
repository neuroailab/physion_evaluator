import io
import numpy as np
from PIL import Image

def get_image(raw_img, pil=False):
    '''
    raw_img: binary image to be read by PIL
    returns: numpy array of shape [H, W, C]
    '''
    img = Image.open(io.BytesIO(raw_img))

    if pil:
        return img
    else:
        return np.array(img)

def get_num_frames(h5_file):
    return len(h5_file['frames'].keys())

def index_imgs(h5_file, indices, suffix='', pil=False):
    all_imgs = [index_img(h5_file, index, suffix=suffix, pil=pil) for index in indices]
    if not pil:
        all_imgs = np.stack(all_imgs, 0)
    return all_imgs

def index_img(h5_file, index, suffix='', pil=False):
    if index > len(h5_file['frames']) - 1:
        index = len(h5_file['frames']) - 1

    img0 = h5_file['frames'][str(index).zfill(4)]['images']

    rgb_img = get_image(img0['_img' + suffix][:], pil=pil)

    return rgb_img
