# Custom OpenCV Setup in a Conda Environment

This document explains how to set up the custom OpenCV build (with `BallTracker`)
in a new conda environment on Windows.

The custom build is a *dynamic* build (~6 MB `cv2.pyd` + sibling `opencv_*.dll`s),
not the bundled stock version (~90 MB single `cv2.pyd`). The stock pip-installed
OpenCV ships its own loader (`cv2/__init__.py`, `cv2/config.py`) that we reuse,
just with a small config patch so it finds our DLLs.

---

## 1. Create the conda env (Python 3.11 to match the build)

```bat
conda create -n my_new_env python=3.11 numpy
conda activate my_new_env
```

Python 3.11 is required because the prebuilt `cv2.cp311-win_amd64.pyd` is built
against the Python 3.11 ABI.

## 2. Install stock OpenCV first

```bat
pip install opencv-contrib-python
```

This installs the official cv2 package with its loader (`__init__.py`, `config.py`,
etc.) into `site-packages/cv2/`. We will overwrite the binaries below; we just
want the Python infrastructure.

## 3. Replace the stock binaries with the custom build

```bat
cd C:\_development\ai\opencv-custom\opencv\build

REM Replace the Python extension
copy /Y lib\python3\Release\cv2.cp311-win_amd64.pyd "%CONDA_PREFIX%\Lib\site-packages\cv2\cv2.pyd"

REM Copy all OpenCV DLLs alongside it
copy /Y bin\Release\opencv_*.dll "%CONDA_PREFIX%\Lib\site-packages\cv2\"
```

## 4. Patch `cv2/config.py` so the loader finds the DLLs

Edit `%CONDA_PREFIX%\Lib\site-packages\cv2\config.py` and add `LOADER_DIR` to
`BINARIES_PATHS`:

```python
import os

BINARIES_PATHS = [
    LOADER_DIR,
    os.path.join(os.path.join(LOADER_DIR, '../../'), 'x64/vc14/bin')
] + BINARIES_PATHS
```

The `LOADER_DIR` entry tells Python to look for OpenCV DLLs in the `cv2/` folder
itself (where we just copied them). Without this, the loader only checks
`x64/vc14/bin` (which the stock pip wheel uses) and the imports fail with
`DLL load failed while importing cv2`.

## 5. Remove any rogue `cv2.pyd` from the DLLs folder

If a stale custom build is sitting in `%CONDA_PREFIX%\DLLs\`, Python finds it
*before* the site-packages loader runs and the DLL search paths never get added.
Delete it:

```bat
del "%CONDA_PREFIX%\DLLs\cv2.cp311-win_amd64.pyd" 2>nul
```

## 6. Verify

```bat
python -c "import cv2; print(cv2.__version__); p = cv2.BallTrackerParams(); print(p.driftThreshold)"
```

Expected output:

```
4.13.0
0.3
```

`4.13.0` confirms the custom build (stock pip ships 4.11.0). The
`BallTrackerParams` instance confirms our custom class is exposed.

---

## Updating after a rebuild

When you rebuild OpenCV (e.g. after editing `ball_tracker.cpp`), repeat steps
3 and 5. Step 4 (the `config.py` patch) is one-time per env.

```bat
cd C:\_development\ai\opencv-custom\opencv\build
copy /Y lib\python3\Release\cv2.cp311-win_amd64.pyd "%CONDA_PREFIX%\Lib\site-packages\cv2\cv2.pyd"
copy /Y bin\Release\opencv_*.dll "%CONDA_PREFIX%\Lib\site-packages\cv2\"
del "%CONDA_PREFIX%\DLLs\cv2.cp311-win_amd64.pyd" 2>nul
```

## Troubleshooting

**`ImportError: DLL load failed while importing cv2`**
Run this to find which `cv2.pyd` Python is loading:

```python
import importlib.util
print(importlib.util.find_spec('cv2').origin)
```

If it points to `<env>\DLLs\cv2.cp311-win_amd64.pyd` instead of
`<env>\Lib\site-packages\cv2\cv2.pyd`, delete the rogue file (step 5).

**`cv2.__version__` reports `4.11.0`**
You're loading the stock pip-installed binary. Repeat step 3 — the copy may
have failed silently (file was open / permission denied).

**`AttributeError: module 'cv2' has no attribute 'BallTracker'`**
The `cv2.pyd` you deployed is the stock build, not the custom one. Verify the
file is ~6 MB (custom) and not ~90 MB (stock).
