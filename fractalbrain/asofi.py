#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 12:08:16 2019

@author: Chiara Marzi, Ph.D. student in Biomedical, Electrical and System Engineering,
at Dept. of Electrical, Electronic and Information Engineering – DEI "Guglielmo Marconi",
University of Bologna, Bologna, Italy. 
E-mail address: chiara.marzi3@unibo.it

fractalbrain toolkit e-mail address: fractalbraintoolkit@gmail.com
"""

# The asofi name was chosen as the acronym of Automated Selection of Fractal Indices
def asofi(subjid, image):
    from fpdf import FPDF
    import logging
    import math
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.max_open_warning': 0})
    import nibabel as nib
    import numpy as np
    import os
    import sklearn.metrics as skl
    import sys

    # --- GPU/CPU backend (numpy or cupy) ---
    # xp(): returns array module (numpy or cupy) depending on FRACTALBRAIN_DEVICE / --device
    try:
        from .backends import xp, asnumpy
        X = xp()  # numpy or cupy
        backend_name = X.__name__  # 'numpy' or 'cupy'
    except Exception:
        X = np
        backend_name = "numpy"
        def asnumpy(a):  # fallback
            return a

    # ---------- GPU-accelerated non-empty box counting (hashing + unique) ----------
    def _make_offsets(M: int, s: int):
        """
        Build M random offsets in [-s, 0] for (y,x,z), like original logic.
        Generated with NumPy for reproducibility, then moved to backend.
        """
        rng = np.random.default_rng(seed=1)  # deterministic like random.seed(1)
        offs = np.stack([
            -rng.integers(0, s + 1, size=M),  # y
            -rng.integers(0, s + 1, size=M),  # x
            -rng.integers(0, s + 1, size=M),  # z
        ], axis=1)  # (M,3)
        return X.asarray(offs)

    def _count_boxes_one_scale(voxels_X, s: int, offsets_xyz_X):
        """
        voxels_X: (N,3) array on backend X with [y,x,z] indices of non-zero voxels
        s: int box size (scale)
        offsets_xyz_X: (M,3) offsets on backend X
        return: mean number of non-empty boxes across M offsets
        """
        # Broadcast offsets vs voxels: (M,1,3) + (1,N,3) -> (M,N,3)
        big = 1_000_000  # keep indices positive before // s (no effect on uniqueness)
        shifted = (voxels_X[None, :, :] + offsets_xyz_X[:, None, :] + big) // s  # (M,N,3)

        # Hash 3D -> 1D to count unique boxes fast
        # Large co-prime-ish strides to minimize collisions.
        stride_y = 73856093
        stride_x = 19349663
        stride_z = 83492791
        hashed = (shifted[:, :, 0] * stride_y +
                  shifted[:, :, 1] * stride_x +
                  shifted[:, :, 2] * stride_z)  # (M,N)

        counts = []
        for m in range(hashed.shape[0]):
            uniq = X.unique(hashed[m])
            counts.append(int(uniq.shape[0]))
        return float(np.mean(counts))

    ### MANAGEMENT OF INPUT FILES: PATH, FILE NAME, EXTENSION, ETC. ###
    print("Loading", image, "image...")
    imagepath = os.path.dirname(image)
    if not imagepath or imagepath == '.':
        imagepath = os.getcwd()
    os.makedirs(imagepath, exist_ok=True)  # ensure out dir exists

    imagefile = os.path.basename(image)
    imagename, image_extension = os.path.splitext(imagefile)
    imagename, image_extension = os.path.splitext(imagename)

    # prefix basename to avoid duplicated folders in output filenames
    prefix_base = os.path.basename(subjid)

    ### LOG FILE SETTING ###
    log_file_name = os.path.join(imagepath, f"{prefix_base}_fractal")
    log = logging.getLogger(log_file_name + '.asofi')
    log.info('Started: image %s with prefix name %s', image, subjid)
    log.info('Backend in use: %s', backend_name)

    ### NIFTI IMAGE LOADING ###
    img = nib.load(image)
    nii_header = img.header
    imageloaded = img.get_fdata()

    ### CHECK THE IMAGE ISOTROPY ###
    voxels_size = nii_header['pixdim'][1:4]
    log.info('The voxel size is %s x %s x %s mm^3', voxels_size[0], voxels_size[1], voxels_size[2])
    if voxels_size[0] != voxels_size[1] or voxels_size[0] != voxels_size[2] or voxels_size[1] != voxels_size[2]:
        sys.exit('The voxel is not isotropic! Exit.')

    ### COMPUTING THE MINIMUM AND MAXIMUM SIZES OF THE IMAGE ###
    L_min = float(voxels_size[0])
    log.info('The minimum size of the image is %s mm', L_min)
    Ly = int(imageloaded.shape[0])
    Lx = int(imageloaded.shape[1])
    Lz = int(imageloaded.shape[2])
    L_Max = max(Lx, Ly, Lz)
    log.info('The maximum size of the image is %s voxels', L_Max)

    ### NON-ZERO VOXELS OF THE IMAGE: NUMBER AND Y, X, Z COORDINATES ###
    # Faster than triple nested loops: argwhere on CPU, then send to backend
    nz_idx = np.argwhere(imageloaded > 0)  # (N, 3) order [y, x, z]
    voxels = X.asarray(nz_idx)  # to GPU if cupy selected, else stays numpy
    log.info('The non-zero voxels in the image are (the image volume) %s', nz_idx.shape[0])

    ##### FRACTAL ANALYSIS #####
    ### LOGARITHM SCALES VECTOR AND COUNTS VECTOR CREATION ###
    Ns = []
    scales = []
    stop = math.ceil(math.log2(L_Max)) if L_Max > 0 else 0
    for exp in range(stop + 1):
        scales.append(2 ** exp)
    scales_np = np.asarray(scales)  # keep numpy copy for plotting/polyfit

    ### THE 3D BOX-COUNTING ALGORITHM WITH 20 OFFSETS (GPU-optimized)
    for scale in scales_np:
        scale = int(scale)
        log.info('Computing scale %s...', scale)
        offsets_X = _make_offsets(M=20, s=scale)           # (20,3) on backend
        mean_count = _count_boxes_one_scale(voxels, scale, offsets_X)
        Ns.append(mean_count)

    ### AUTOMATED SELECTION OF THE FRACTAL SCALING WINDOW ###
    minWindowSize = 5  # in log scale (>= ~1.2 decades)
    scales_indices = []
    for step in range(scales_np.size, minWindowSize - 1, -1):
        for start_index in range(0, scales_np.size - step + 1):
            scales_indices.append((start_index, start_index + step - 1))
    scales_indices = np.asarray(scales_indices)

    k_ind = 1  # number of independent variables in the regression model
    R2_adj = -1
    FD = None
    mfs = None
    Mfs = None
    fsw_index = None
    coeffs_selected = None

    log2_scales = np.log2(scales_np)
    log2_Ns = np.log2(np.asarray(Ns) + 1e-12)  # stabilize if Ns has zeros

    for k in range(scales_indices.shape[0]):
        s0, s1 = scales_indices[k, 0], scales_indices[k, 1]
        coeffs = np.polyfit(log2_scales[s0:s1 + 1], log2_Ns[s0:s1 + 1], 1)
        n = (s1 - s0 + 1)
        y_true = log2_Ns[s0:s1 + 1]
        y_pred = np.polyval(coeffs, log2_scales[s0:s1 + 1])
        R2 = skl.r2_score(y_true, y_pred)
        R2_adj_tmp = 1 - (1 - R2) * ((n - 1) / (n - (k_ind + 1)))

        log.info('In the interval [%s, %s] voxels, the FD is %s and the adjusted R2 is %s',
                 scales_np[s0], scales_np[s1], -coeffs[0], R2_adj_tmp)

        # round for fair comparison as original code did (order matters)
        R2_adj = round(R2_adj, 3)
        R2_adj_tmp = round(R2_adj_tmp, 3)

        if R2_adj_tmp > R2_adj:
            R2_adj = R2_adj_tmp
            FD = -coeffs[0]
            mfs = scales_np[s0]
            Mfs = scales_np[s1]
            fsw_index = k
            coeffs_selected = coeffs
        FD = round(FD, 4)

    ### FRACTAL ANALYSIS RESULTS ###
    mfs_mm = float(mfs) * L_min
    Mfs_mm = float(Mfs) * L_min
    log.info('The mfs automatically selected is %s', mfs_mm)
    log.info('The Mfs automatically selected is %s', Mfs_mm)
    log.info('The FD automatically selected is %s', FD)
    log.info('The R2_adj is %s', R2_adj)
    print("mfs automatically selected:", mfs_mm)
    print("Mfs automatically selected:", Mfs_mm)
    print("FD automatically selected:", FD)
    print(f"(backend used for counting: {backend_name})")

    ### SAVING THE PLOT WITH THE AUTOMATED SELECTED FRACTAL SCALING WINDOW ###
    plt.figure()
    plt.plot(np.log2(scales_np), np.log2(np.asarray(Ns)), 'o', mfc='none')
    s0, s1 = scales_indices[fsw_index, 0], scales_indices[fsw_index, 1]
    plt.plot(
        np.log2(scales_np[s0:s1 + 1]),
        np.polyval(coeffs_selected, np.log2(scales_np[s0:s1 + 1]))
    )
    plt.xlabel('log $\epsilon$ (mm)')
    plt.ylabel('log N (-)')

    out_png = os.path.join(imagepath, f"{prefix_base}_{imagename}_FD_plot.png")
    plt.savefig(out_png)
    plt.clf()

    ### CREATION OF A TXT FILE WITH FRACTAL ANALYSIS RESULTS ###
    out_txt = os.path.join(imagepath, f"{prefix_base}_{imagename}_FractalIndices.txt")
    with open(out_txt, 'w') as f:
        f.write("mfs (mm), %f\n" % mfs_mm)
        f.write("Mfs (mm), %f\n" % Mfs_mm)
        f.write("FD (-), %f\n" % FD)
        f.write("backend, %s\n" % backend_name)

    ### CREATION OF A PDF FILE WITH FRACTAL ANALYSIS RESULTS ###
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Subject    " + subjid, ln=1, align="C")
    pdf.cell(200, 10, txt="Image    " + imagefile, ln=1, align="C")
    pdf.cell(200, 10, txt="mfs    " + str(mfs_mm) + " mm", ln=1, align="C")
    pdf.cell(200, 10, txt="Mfs    " + str(Mfs_mm) + " mm", ln=1, align="C")
    pdf.cell(200, 10, txt="FD    " + str(FD), ln=1, align="C")
    pdf.cell(200, 10, txt="R2adj    " + str(R2_adj), ln=1, align="C")
    try:
        pdf.image(out_png, x=60, w=100)
    except Exception as e:
        log.info("Failed to embed PNG into PDF: %s", e)
    out_pdf = os.path.join(imagepath, f"{prefix_base}_{imagename}_FD_summary.pdf")
    pdf.output(out_pdf)

    return
