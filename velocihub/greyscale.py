"""The single reader of the images: every step (pre-processing, PIV, PTV, particle diagnostic, figures) reads its
images with read_grey, so that they all see the same intensities.

The images may be in any colour space: they are brought to one channel, the brightness (value V of the HSV colour
space: hue, saturation, value). V = max(R, G, B) does not depend on the hue and the saturation, which are not
computed: the conversion is a maximum over the colour channels, as fast as their mean. The channels are taken as
they are (no gamma correction): a camera gives intensities proportional to the light. A grey image is unchanged; a
colour image keeps, at each pixel, its brightest channel (the one of the light of the laser, e.g. green at 532 nm).

    palette (P, PA)            the colours of the palette (not the indices), then V
    RGB, RGBA, RGBX, BGR(A)    V = max of the three colour channels (alpha left out)
    CMYK, YCbCr, LAB           converted to RGB by Pillow, then V
    HSV                        its channel V
    L, LA, 1                   the luminance (alpha left out; 1 bit: 0 or 255)
    I;16, I, F                 as they are: 16 bit unsigned; 32 bit integers as 16 bit unsigned when they fit,
                               else 32 bit floats; 32 bit floats

The depth is kept: uint8, uint16 or float32. Pillow reads the images; OpenCV, if installed, those it can't read
(e.g. 3 x 16 bit TIFF).

This module only needs numpy and Pillow (OpenCV optionally) and imports nothing of VelociHub: the PIV and PTV
workers, run by another Python interpreter that may not have VelociHub, load it from its file.
"""
import time

import numpy as np
from PIL import Image

# Read tries of a file that a network or USB drive refuses for a moment, and seconds between them
READ_TRIES = 5
READ_WAIT = 3.0


def to_grey(A):
    """One channel of an image array: unchanged if it has one (H x W), else the brightness V = max of its colour
    channels (H x W x C: the first three; one or two channels: the first, the second being alpha). The type is kept,
    except bool (0 or 255, uint8) and the 32 bit integers (uint16 when they fit, else float32)."""
    A = np.asarray(A)
    if A.ndim == 3:
        A = A[..., 0] if A.shape[2] < 3 else A[..., :3].max(axis=2)
    if A.dtype == bool:
        return A.astype(np.uint8) * np.uint8(255)
    if A.dtype.byteorder not in ("=", "|"):
        A = A.astype(A.dtype.newbyteorder("="))
    if np.issubdtype(A.dtype, np.integer) and A.dtype not in (np.uint8, np.uint16):
        if A.size and A.min() >= 0 and A.max() <= 65535:
            return A.astype(np.uint16)
        return A.astype(np.float32)
    if np.issubdtype(A.dtype, np.floating) and A.dtype != np.float32:
        return A.astype(np.float32)
    return A


def pil_array(im):
    """The array of a Pillow image, converted to a colour space whose channels are intensities (palette: its
    colours; CMYK, YCbCr, LAB: RGB; HSV: its channel V)."""
    mode = im.mode
    if mode in ("P", "PA"):
        im = im.convert("RGBA" if mode == "PA" or "transparency" in im.info else "RGB")
    elif mode in ("CMYK", "YCbCr", "LAB"):
        im = im.convert("RGB")
    elif mode == "HSV":
        return np.asarray(im)[..., 2]
    elif mode == "1":
        im = im.convert("L")
    return np.asarray(im)


def read_array(path):
    """The image of a file as an array, its channels as stored (Pillow, else OpenCV), from a path with any
    characters."""
    try:
        with Image.open(path) as im:
            return pil_array(im)
    except (OSError, ValueError, TypeError) as error:
        try:
            import cv2
        except ImportError:
            raise error from None
        A = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
        if A is None:
            raise error from None
        return A  # BGR(A): the maximum of the colour channels doesn't depend on their order


def read_grey(path, tries=READ_TRIES, wait=READ_WAIT):
    """The image of a file in one channel (to_grey: the brightness of a colour image), its depth kept. A read that
    fails is tried again (tries in all, wait seconds between them): a network or USB drive sometimes refuses it for
    a moment."""
    for attempt in range(max(int(tries), 1)):
        try:
            return to_grey(read_array(path))
        except OSError:
            if attempt >= tries - 1:
                raise
            time.sleep(wait)

