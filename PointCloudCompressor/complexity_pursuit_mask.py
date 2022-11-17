import numpy as np


def return_mask(half_life, half_lives_per_mask, max_mask_length):
    mask_length = min(half_lives_per_mask*half_life, max_mask_length)
    print('Returning mask for Complexity Pursuit with size: ', mask_length)
    mask = (2**(-1/half_life))**np.arange(0, mask_length).T
    mask[0] = 0
    mask = mask/(np.sum(np.abs(mask)))
    mask[0] = -1
    return mask
