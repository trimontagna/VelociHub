"""Jet theory add-on: what VelociHub knows of jets, kept apart from the rest of the software, which makes no
assumption on the flow.

theory.py   the potential core of a jet (JetSettings, part of the recipe PostSettings, on with jet_theory; JetAnalysis,
            a mixin of PostProcessing: potential_core, its figures core_map, core_profiles, core_centreline and its
            numbers in metrics.json)
presets.py  the jet measurements of the PIV laboratory: fields of view, frame rate, axes centred on the jet
            (jet_camera), used by the archiving add-on

    import velocihub as vh

    camera = vh.jet.jet_camera(1)  # FOV 1, 150 Hz, y = 0 on the jet axis
    post = vh.PostProcessing(vh.load_vectors("Run_30/PIV", camera=camera), vh.PostSettings(jet_theory=True))
    post.plot("core_profiles")
"""

from .presets import (FOV_PRESETS, JET_X_SHIFT, JET_Y_SHIFT, NOZZLE_DIAMETER_MM, PRESET_FRAME_RATE, fov_help,
                      fov_offsets, jet_camera)
from .theory import CORE_MODES, JET_FIGURES, JET_GROUP, JetAnalysis, JetSettings, peak_skewness

__all__ = ["CORE_MODES", "FOV_PRESETS", "JET_FIGURES", "JET_GROUP", "JET_X_SHIFT", "JET_Y_SHIFT", "JetAnalysis",
           "JetSettings", "NOZZLE_DIAMETER_MM", "PRESET_FRAME_RATE", "fov_help", "fov_offsets", "jet_camera",
           "peak_skewness"]
