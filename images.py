import numpy as np


def pad_image(img, x_final, y_final):
    """
    Pad `img` (array with shape [Y, X, 3], 
    representing an RGB image) with zeros (black) 
    on the Y and X dimensions so the final `img` 
    has shape [`y_final`, `x_final`, 3].
    
    If the specified final size is smaller than
    the original size, nothing is done on that
    dimension.    
    """
    
    # Get shapes:
    final_shape = np.array([y_final, x_final, 3])
    img_shape   = np.array(img.shape)
    assert img_shape[2] == 3, 'Expecting RGB image, but found last axis with {} dims.'.format(img_shape[2])
    #assert (final_shape >= img_shape).all(), 'Final size should be equal or larger than the image.'
    
    # Compute padding:
    total_padding = np.max([np.zeros_like(img_shape), final_shape - img_shape], axis=0)
    left_pad  = (total_padding / 2).astype(int)
    right_pad = total_padding - left_pad

    # Pad image:
    padded_img = np.pad(img, tuple(zip(left_pad, right_pad)))

    return padded_img


def crop_to_center(img, x_final, y_final):
    """
    Crop `img` (array with shape [Y, X, 3], 
    representing an RGB image) to the center 
    portion so the final image has shape 
    [`y_final`, `x_final`, 3].
    
    If the specified final size is larger than
    the original size, nothing is done on that
    dimension.
    """
    
    y, x, c = img.shape
    start_x = x // 2 - (x_final // 2)
    start_y = y // 2 - (y_final // 2)    
    return img[start_y:start_y + y_final, start_x:start_x + x_final, :]


def pad_crop_image(img, x_final, y_final):
    """
    Pad and crop `img` (array with shape 
    [Y, X, 3], representing an RGB image)
    so to return an array with shape 
    [`y_final`, `x_final`, 3].
    """
    
    final_img = pad_image(img, x_final, y_final)
    final_img = crop_to_center(final_img, x_final, y_final)
    
    return final_img
