"""The processing pipelines of the velocimetry: one per engine of the PIV/PTV (piv.ENGINES: PaIRS, TracTrac...), named
by its label. The results of a pipeline are in its folder of the measurement (its results_folder: PIV for PaIRS, PTV
for TracTrac), their post-processing exported to its subfolder of PostProcessing (PostProcessing/PaIRS,
PostProcessing/TracTrac). A new engine registered is a new pipeline: nothing else to change.

    import velocihub as vh

    vh.pipelines.results_folders("D:/Jet/Run_30")    # {"PaIRS": "D:/Jet/Run_30/PIV", "TracTrac": "D:/Jet/Run_30/PTV"}
    vh.pipelines.pipeline_of("D:/Jet/Run_30/PTV")    # "TracTrac"
"""

import json
import os
import zipfile

import numpy as np

from . import piv
from .fields import VECTORS_FILE, ptv_results


def engines():
    """The engines of the pipelines (classes, in the order of their registration)."""
    return list(piv.ENGINES.values())


def pipelines():
    """The pipelines: {label: results folder (relative to the measurement)}, in the order of the engines."""
    return {cls.label: cls.results_folder for cls in engines()}


DEFAULT_PIPELINE = piv.ENGINES[piv.DEFAULT_ENGINE].label


def pipeline_label(name):
    """The label of a pipeline from its label, the key or the name of its engine, or a method (PIV, PTV: the first
    pipeline of this method; the results of the previous versions); None if unknown."""
    if not isinstance(name, str) or not name:
        return None
    low = name.lower()
    for cls in engines():
        if low in (cls.label.lower(), cls.key.lower(), cls.name.lower()):
            return cls.label
    return next((cls.label for cls in engines() if cls.method.lower() == low), None)


def recorded_engine(maps):
    """The engine recorded with the vector fields of a folder (its settings.json, else the settings of its
    vectors.npz), None if none."""
    try:
        with open(os.path.join(maps, "settings.json"), encoding="utf-8") as f:
            return json.load(f).get("engine")
    except (OSError, ValueError, AttributeError):
        pass
    try:
        with zipfile.ZipFile(os.path.join(maps, VECTORS_FILE)) as z, z.open("settings.npy") as f:
            return json.loads(str(np.lib.format.read_array(f, allow_pickle=False))).get("engine")
    except (OSError, KeyError, ValueError, AttributeError, zipfile.BadZipFile):
        return None


def pipeline_of(maps):
    """The pipeline of the vector fields of a folder: that of the engine recorded with them, else the one whose
    results folder it is named after, else that of their method (PTV: the particles of their cells known); None
    without vectors.npz (DynamicStudio CSV exports, nothing)."""
    if not maps or not os.path.isfile(os.path.join(maps, VECTORS_FILE)):
        return None
    label = pipeline_label(recorded_engine(maps))
    if label is not None:
        return label
    name = os.path.basename(os.path.normpath(maps))
    by_folder = [cls for cls in engines() if cls.results_folder == name]
    if len(by_folder) == 1:
        return by_folder[0].label
    return pipeline_label("PTV" if ptv_results(maps) else "PIV")


def results_folders(folder):
    """The results of a measurement by pipeline: {label: folder}, those it has, in the order of the pipelines (the
    vector fields in the folder itself, as the previous versions could keep them: those of their pipeline)."""
    out = {}
    for sub in dict.fromkeys(pipelines().values()):
        path = os.path.join(folder, sub)
        label = pipeline_of(path)
        if label is not None and label not in out:
            out[label] = path
    label = pipeline_of(folder)
    if label is not None and label not in out:
        out[label] = folder
    order = list(pipelines())
    return dict(sorted(out.items(), key=lambda item: order.index(item[0]) if item[0] in order else len(order)))


def chosen_results(folders, results=None):
    """The folder of the results chosen (results: a pipeline, pipeline_label, or results sets: the first of them that
    the measurement has; None: the default one) among the results of a measurement ({label: folder}): those of the
    default pipeline are those of the first pipeline found if it has none; None if it doesn't have those chosen."""
    if isinstance(results, (list, tuple)):
        return next((folders[label] for label in results_sets(results) if label in folders), None)
    label = DEFAULT_PIPELINE if results is None else pipeline_label(results)
    if label in folders:
        return folders[label]
    if label == DEFAULT_PIPELINE:
        return next(iter(folders.values()), None)
    return None


def results_sets(value):
    """The results sets chosen (the batch post-processing: the pipelines whose results are exported, in the order of
    the pipelines): a list of pipelines (pipeline_label), a pipeline alone (the previous versions), None: all of them.
    ValueError for an unknown one."""
    if value is None:
        return list(pipelines())
    chosen = []
    for name in [value] if isinstance(value, str) else value:
        label = pipeline_label(name)
        if label is None:
            raise ValueError(f'Unknown results "{name}": {", ".join(pipelines())}')
        chosen.append(label)
    return [label for label in pipelines() if label in chosen]


def selected_results(folders, value):
    """The results of a measurement ({label: folder}) of the results sets chosen (results_sets)."""
    chosen = results_sets(value)
    return {label: folder for label, folder in folders.items() if label in chosen}


def pipeline_of_folder(rel):
    """The pipeline whose results folder is a folder relative to a measurement (as a project file keeps its maps
    folder), None if none (or several)."""
    if not isinstance(rel, str) or not rel or os.path.isabs(rel):
        return None
    labels = [label for label, sub in pipelines().items() if os.path.normpath(rel) == os.path.normpath(sub)]
    return labels[0] if len(labels) == 1 else None


def results_folder_names():
    """The folders of the results of the pipelines (relative to a measurement)."""
    return set(pipelines().values())
