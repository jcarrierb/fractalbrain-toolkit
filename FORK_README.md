# fractalbrain-toolkit — GPU Fork (CuPy)

## Overview
This fork of the original [fractalbrain-toolkit](https://github.com/chiaramarzi/fractalbrain-toolkit) adds optional GPU acceleration through [CuPy](https://cupy.dev/).  
It preserves the original fractal dimension calculation methods (3D box-counting), while significantly reducing computation time when using a CUDA-capable GPU.

**Benchmark**  
- 3 subjects × 360 ROIs  
  - Original CPU version: ~17 minutes  
  - GPU fork (CUDA): ~1 minute 


## Requirements

### Hardware
- NVIDIA GPU with CUDA support (e.g., RTX 30xx / 40xx)
- Up-to-date NVIDIA drivers
- CUDA Toolkit installed or provided by the driver

### Python Dependencies
Install CuPy for your CUDA version:
pip install cupy-cuda12x

All other dependencies remain the same as the original toolkit.


## Technical Notes
If FRACTALBRAIN_DEVICE is not provided, the default backend is CPU.
Alternatively, you can set the environment variable: set FRACTALBRAIN_DEVICE=cuda

Two options are supported:
  A) Explicitly specify the fork path using --repo-root (recommended if you keep the original and fork side by side).
  B) Install only the fork in your Python environment. In this case, running fractalbrain will automatically use the fork.

--device (or FRACTALBRAIN_DEVICE) controls the computation backend:
cpu → NumPy
cuda → CuPy

Acceleration happens mainly in asofi.py during the box-counting step (histogramdd).
The scientific method is unchanged (automatic scale fitting + box-counting).
Slight numerical differences may occur between CPU and GPU results due to binning precision, float32 vs float64, and minor rounding variations.


## Attribution
This is a GPU optimization fork of the original fractalbrain-toolkit.
All original code and methodology remain credited to:

Created on Tue Nov 5 11:55:49 2019
Author: Chiara Marzi, Ph.D. student in Biomedical, Electrical and System Engineering,
at Dept. of Electrical, Electronic and Information Engineering – DEI "Guglielmo Marconi",
University of Bologna, Bologna, Italy.
E-mail: chiara.marzi3@unibo.it
fractalbrain toolkit e-mail: fractalbraintoolkit@gmail.com
