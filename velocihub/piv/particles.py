"""Particles of the interrogation windows: are there enough of them for the PIV (at least about 10 per window)?

A particle is a local maximum of the image filtered with a Gaussian particle image (matched filter: the correlation of
the image with a particle image, of standard deviation a quarter of the particle diameter) whose height above the
background is at least snr times the noise. The background is the median of the filtered largest window and the
noise its robust standard deviation (1.4826 median absolute deviation, at least half a grey level for the images
of integers). The windows are nested, centred on the same point: the particles are found once in the largest one
and counted in each.

    from velocihub.piv import count_particles

    found = count_particles(image, center=(960, 600), sizes=(128, 64, 32), snr=4, diameter=3)
    for w in found["windows"]:
        print(w["size"], w["count"])
"""

import numpy as np
from scipy import ndimage

from ..greyscale import to_grey

# Particles needed in an interrogation window (rule of thumb of the PIV)
MIN_PARTICLES = 10
# Signal-to-noise ratio of a particle and diameter of the particle images (px), by default
DEFAULT_SNR = 4.0
DEFAULT_DIAMETER = 3.0


def count_particles(image, center, sizes, snr=DEFAULT_SNR, diameter=DEFAULT_DIAMETER):
    """Particles of the interrogation windows of the sizes given (px), centred on center = (x, y) in pixels from the
    top-left corner of the image (the pixel of row r and column c spans [c, c + 1] x [r, r + 1]); the windows are
    moved inside the image if needed.

    Returns a dictionary: windows, a list of {size, x0, y0 (top-left corner), count} from the largest window to the
    smallest; x, y and snr of the particles of the largest window (their centres, in pixels from the top-left
    corner of the image); box = (x0, y0, size) of the largest window; background and noise (grey levels)."""
    if not snr > 0 or not diameter > 0:
        raise ValueError("The signal-to-noise ratio and the particle diameter must be positive")
    A = to_grey(image)
    integers = np.issubdtype(A.dtype, np.integer)
    A = A.astype(np.float64)
    H, W = A.shape
    sizes = sorted({int(s) for s in sizes if int(s) > 0}, reverse=True)
    if not sizes:
        raise ValueError("No interrogation window size")
    big = sizes[0]
    x0 = int(np.clip(round(center[0] - big / 2), 0, max(W - big, 0)))
    y0 = int(np.clip(round(center[1] - big / 2), 0, max(H - big, 0)))
    # Filtered with a margin around the largest window, so that its edges are filtered as its inside
    sigma = max(diameter / 4, 0.5)
    margin = int(np.ceil(3 * sigma)) + 1
    ax0, ay0 = max(x0 - margin, 0), max(y0 - margin, 0)
    F = ndimage.gaussian_filter(A[ay0 : min(y0 + big + margin, H), ax0 : min(x0 + big + margin, W)], sigma,
                                mode="nearest")
    inner = F[y0 - ay0 : y0 - ay0 + big, x0 - ax0 : x0 - ax0 + big]
    background = float(np.median(inner))
    noise = max(1.4826 * float(np.median(np.abs(inner - background))), 0.5 if integers else 0.0)
    noise = noise or float(np.finfo(float).eps)
    # Local maxima above the background: a plateau of equal values is one particle
    neighbourhood = max(3, int(round(diameter)) | 1)
    peaks = ((F == ndimage.maximum_filter(F, size=neighbourhood, mode="nearest")) & (F > background)
             & (F - background >= snr * noise))
    labels, n = ndimage.label(peaks)
    if n:
        index = np.arange(1, n + 1)
        rows, cols = np.array(ndimage.center_of_mass(peaks, labels, index)).T
        heights = np.asarray(ndimage.maximum(F, labels, index), float)
    else:
        rows = cols = heights = np.zeros(0)
    x, y = cols + ax0 + 0.5, rows + ay0 + 0.5
    inside = (x >= x0) & (x < x0 + big) & (y >= y0) & (y < y0 + big)
    x, y, ratio = x[inside], y[inside], (heights[inside] - background) / noise
    windows = []
    for s in sizes:
        wx0, wy0 = x0 + (big - s) // 2, y0 + (big - s) // 2
        count = int(((x >= wx0) & (x < wx0 + s) & (y >= wy0) & (y < wy0 + s)).sum())
        windows.append({"size": s, "x0": wx0, "y0": wy0, "count": count})
    return {"windows": windows, "x": x, "y": y, "snr": ratio, "box": (x0, y0, big), "background": background,
            "noise": noise}
