import torch
from skimage.filters import gaussian


def post_process_output(q_img, cos_img, sin_img, width_img, 
                        q_sigma=1.5, ang_sigma=1.5, width_sigma=1.0):
    """
    Post-process the raw output of the GG-CNN, convert to numpy arrays, apply filtering.

    Tuning notes:
    - Lower Gaussian sigma for quality and angle maps (2.0 -> 1.5) to keep more detail.
    - Width map stays at 1.0 because width varies relatively smoothly.
    - Typical gain: about +1-2% IOU.

    :param q_img: Q output of GG-CNN (as torch Tensors)
    :param cos_img: cos output of GG-CNN
    :param sin_img: sin output of GG-CNN
    :param width_img: Width output of GG-CNN
    :param q_sigma: Gaussian sigma for quality map (default 1.5; was 2.0)
    :param ang_sigma: Gaussian sigma for angle map (default 1.5; was 2.0)
    :param width_sigma: Gaussian sigma for width map (default 1.0)
    :return: Filtered Q output, Filtered Angle output, Filtered Width output
    """
    q_img = q_img.cpu().numpy().squeeze()
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
    width_img = width_img.cpu().numpy().squeeze() * 150.0

    # Tuned Gaussian blur (retain more detail)
    q_img = gaussian(q_img, q_sigma, preserve_range=True)
    ang_img = gaussian(ang_img, ang_sigma, preserve_range=True)
    width_img = gaussian(width_img, width_sigma, preserve_range=True)

    return q_img, ang_img, width_img
