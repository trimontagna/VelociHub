"""Preferences of VelociHub (the MATLAB getpref/setpref of the original tool): a JSON file in the home folder,
~/.velocihub.json (those of the previous names of the tool read until it is written). Used by the interface (the
folders, windows and options it remembers) and by the PIV engines (the Python interpreter of PaIRS-UniNa)."""

import json
import os


# Replaces the MATLAB preferences (getpref/setpref)
PREFS_FILE = os.path.join(os.path.expanduser("~"), ".velocihub.json")


# The preferences of the previous names of the tool (Pre_PIV_Post, POD_PAIRS): read until PREFS_FILE is written
LEGACY_PREFS_FILES = tuple(os.path.join(os.path.expanduser("~"), name) for name in (".ppp.json", ".pod_pairs.json"))


def read_prefs():
    """All the preferences (PREFS_FILE, else LEGACY_PREFS_FILES; none if neither can be read)."""
    for path in (PREFS_FILE, *LEGACY_PREFS_FILES):
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, ValueError):
            continue
    return {}


def getpref(name, default=None):
    """MATLAB getpref('VelociHub', name)."""
    prefs = read_prefs()
    return prefs.get(name, default) if isinstance(prefs, dict) else default


def setpref(name, value):
    """MATLAB setpref('VelociHub', name, value)."""
    prefs = read_prefs()
    prefs = prefs if isinstance(prefs, dict) else {}
    prefs[name] = value
    try:
        with open(PREFS_FILE, "w") as f:
            json.dump(prefs, f, indent=2)
    except OSError:
        pass
