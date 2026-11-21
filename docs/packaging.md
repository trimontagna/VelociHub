# Packaging: pip and conda

VelociHub is a pure-Python package (`noarch`), so one build works on every platform. The repository already contains
everything needed:

| File | Purpose |
|---|---|
| `pyproject.toml` | pip/PyPI metadata: dependencies, optional extras, the five commands, version from `velocihub/credits.py` |
| `conda/meta.yaml` | conda-build recipe (noarch: python) |
| `environment.yml` | Conda environment for users of the repository (dependencies + VelociHub in editable mode) |
| `environment-pairs.yml` | Conda environment of the PIV engine PaIRS-UniNa (Python 3.13) |

## What was checked

- `pip wheel .` builds `velocihub-1.3.0-py3-none-any.whl` with every subpackage, the vendored TracTrac and its
  licence, and the three licence files (`LICENSE`, `NOTICE.md`, TracTrac's `LICENSE`).
- The wheel installed into an empty folder (outside the repository) imports, and its commands work:
  `velocihub-batch` ran the pre-processing and the PIV (PaIRS-UniNa found in its own conda environment) end to end,
  the post-processing exported its figures, TracTrac tracked a sequence, and `velocihub-archive --list` listed the
  project.
- Tested with Python 3.12, 3.13 and 3.14 (the minimum, `requires-python = ">=3.12"`).

## pip

### Install from GitHub (now)

```bash
pip install "velocihub[all] @ git+https://github.com/trimontagna/VelociHub.git"
pip install "velocihub[all] @ git+https://github.com/trimontagna/VelociHub.git@v1.3.0"   # a tagged version
```

A private repository asks for your GitHub credentials (or use an SSH URL, `git+ssh://git@github.com/...`).

### Build a wheel

```bash
python -m pip install build
python -m build                    # dist/velocihub-1.3.0-py3-none-any.whl and dist/velocihub-1.3.0.tar.gz
pip install dist/velocihub-1.3.0-py3-none-any.whl
```

### Publish on PyPI (`pip install velocihub`)

1. Create accounts on https://test.pypi.org and https://pypi.org, and an API token.
2. `python -m pip install twine`, then `python -m build`.
3. Try TestPyPI first: `twine upload --repository testpypi dist/*`, then
   `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple velocihub`.
4. `twine upload dist/*`. Each release needs a new version in `velocihub/credits.py` (`__version__`).

A GitHub Actions workflow with PyPI *trusted publishing* can do steps 2-4 on every tag. Note that a package on PyPI
is public, so the repository's licence terms (CC BY-NC-SA 4.0) then apply to everyone who installs it.

### Extras

| Extra | Installs | For |
|---|---|---|
| `velocihub[ptv]` | imutils, parse, h5py | TracTrac |
| `velocihub[regression]` | spicy_vki | SPICY regression |
| `velocihub[video]` | imageio-ffmpeg | MP4 videos |
| `velocihub[pairs]` | PaIRS-UniNa (Python ≤ 3.13 only) | PIV in the same environment |
| `velocihub[all]` | ptv + regression + video | Everything that works with any Python |

PaIRS-UniNa only has wheels for Python 3.10-3.13 (its compiled module links `python313.dll` for 3.13). With Python
3.14+, keep it in a separate environment (`environment-pairs.yml`); VelociHub runs it as a worker process with that
interpreter, so the two Python versions never mix.

## conda

### Build the package locally

```bash
conda install -n base conda-build          # once
conda build conda -c conda-forge           # builds velocihub-1.3.0-py_0.conda in the local channel
conda create -n vh -c local -c conda-forge velocihub
conda activate vh
velocihub
```

The recipe takes its dependencies from conda-forge (`py-opencv`, `matplotlib-base`, `tk`...). The optional PTV and
video packages exist on conda-forge too (`conda install -c conda-forge imutils parse h5py imageio-ffmpeg`);
`spicy_vki` and `PaIRS-UniNa` are only on PyPI, installed with pip inside the environment.

### Share it

- **Your own channel on anaconda.org**: `anaconda login`, then `anaconda upload <path printed by conda build>`; users
  run `conda install -c <your-channel> -c conda-forge velocihub`. This is the simplest way to offer
  `conda install`.
- **conda-forge** (the community channel) requires an open licence that allows any use; CC BY-NC-SA 4.0 forbids
  commercial use, so a conda-forge submission (a pull request to `conda-forge/staged-recipes`) would most likely be
  refused. Your own channel, or pip, is the way for this licence.

### Without a package

`conda env create -f environment.yml` gives a working environment from a clone, with VelociHub installed in editable
mode (changes to the code apply immediately).

## Recommendation

1. Keep installing from GitHub with pip (`pip install "velocihub[all] @ git+..."`), tagging releases (`v1.3.0`).
2. When it should be public: publish the wheel on PyPI, and a conda package on your own anaconda.org channel.
3. Each release: bump `__version__` in `velocihub/credits.py` and the version in `conda/meta.yaml`, tag, build, upload.
