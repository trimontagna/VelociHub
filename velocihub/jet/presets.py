"""The jet measurements of the PIV laboratory of the VKI: three fields of view along the jet, filmed at 150 Hz, the jet
coming from the left, centred in the height of the images. Reminders shown next to the camera settings, and the
cameras of the archiving add-on (archiving/post_archive.py); the values are entered by the user in the interface.
"""

from ..piv.base import Camera, num2str

# Fields of view: size, and axial offset of their left edge from the one of FOV 1 (12 cm between the fields of view)
FOV_PRESETS = {1: dict(width_mm=196.1, height_mm=122.6, x_offset_mm=0.0),
               2: dict(width_mm=196.5, height_mm=122.8, x_offset_mm=120.0),
               3: dict(width_mm=196.3, height_mm=122.7, x_offset_mm=240.0)}
PRESET_FRAME_RATE = 150.0
# The origin of the axes of the jet measurements (Camera.x_shift, y_shift): the left edge, the middle of the height
# (the jet axis)
JET_X_SHIFT = 0.0
JET_Y_SHIFT = 0.5
# Diameter of the nozzle (mm): the reference length L_ref of their figures
NOZZLE_DIAMETER_MM = 50.0


def fov_help():
    """Reminder of the fields of view of the jet measurements, shown next to the camera settings."""
    lines = [f"FOV {k}: {v['width_mm']} x {v['height_mm']} mm, horiz. offset {num2str(v['x_offset_mm'])} mm"
             for k, v in FOV_PRESETS.items()]
    return f"Jet measurements ({num2str(PRESET_FRAME_RATE)} Hz):\n" + "\n".join(lines)


def fov_offsets():
    """The horizontal offsets of the fields of view of the jet measurements, one per line."""
    return "\n".join(f"FOV {k}: {num2str(v['x_offset_mm'])} mm" for k, v in FOV_PRESETS.items())


def jet_camera(fov, x_offset_mm=None, y_offset_mm=0.0, frame_rate_hz=PRESET_FRAME_RATE):
    """The camera of the field of view fov (1, 2, 3) of the jet measurements: its width (square pixels), its offset
    (x_offset_mm: the preset one if None), the axes centred on the jet (JET_X_SHIFT, JET_Y_SHIFT)."""
    preset = FOV_PRESETS[int(fov)]
    return Camera(fov_width_mm=preset["width_mm"],
                  x_offset_mm=preset["x_offset_mm"] if x_offset_mm is None else x_offset_mm,
                  frame_rate_hz=frame_rate_hz, y_offset_mm=y_offset_mm, x_shift=JET_X_SHIFT, y_shift=JET_Y_SHIFT)
