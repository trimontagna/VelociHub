TracTrac, Python version 3.0 (22/05/2021), by Joris Heyman: massive object tracking (PTV).
https://github.com/jorishey1234/tractrac, commit ef64bdbf5f16266526ad82fc67cabeaaf2646977 (MIT licence: LICENSE).
Reference: Heyman J. (2019), TracTrac: a fast multi-object tracking algorithm for motion estimation,
Computers & Geosciences 128, 11-18.

tractrac.py is the file Python/tractrac.py of that commit, with three lines changed for numpy 2, each marked
"# VelociHub:" (lines 100, 993 and 994):
- Propagate_MotionModel_KdTree: np.int16(motion_av) overflowed when compared with more than 32767 points;
- tractrac(): np.where(ISgood == 1) failed when no particle was matched (ISgood is then an empty list).

VelociHub tracks the images with its own copy of the loop of tractrac() (velocihub/piv/ptv_worker.py), which calls the
functions of this file (detection, motion model) as they are: tested to give the same tracks as tractrac().
Needs opencv-python, scipy, matplotlib, imutils, parse and h5py (imported by tractrac.py).
