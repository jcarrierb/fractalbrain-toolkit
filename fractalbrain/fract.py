#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:55:49 2019

@author: Chiara Marzi, Ph.D. student in Biomedical, Electrical and System Engineering,
at Dept. of Electrical, Electronic and Information Engineering – DEI "Guglielmo Marconi",
University of Bologna, Bologna, Italy. 
E-mail address: chiara.marzi3@unibo.it

fractalbrain toolkit e-mail address: fractalbraintoolkit@gmail.com
"""

# --- GPU/CPU backend + box-counting imports (optional extension) ---
try:
    from .boxcount_gpu import boxcount_3d_counts
    from .backends import xp, asnumpy  # xp() returns numpy or cupy
except Exception:
    xp = None
    asnumpy = None
    boxcount_3d_counts = None


def compute_fd_from_binary(binary_vol, scales):
    if boxcount_3d_counts is None or xp is None or asnumpy is None:
        raise RuntimeError(
            "boxcount_3d_counts/xp/asnumpy not available. "
            "Ensure fractalbrain/backends.py and fractalbrain/boxcount_gpu.py are present."
        )

    counts = boxcount_3d_counts(binary_vol, scales)

    # build arrays for regression
    X = xp()  # <--- get array module (numpy or cupy)
    s_arr = X.array(sorted(counts.keys()), dtype=X.float32)
    n_arr = X.array([counts[int(s)] for s in s_arr.tolist()], dtype=X.float32)

    # log-log
    log_inv_s = X.log(1.0 / s_arr)
    log_n     = X.log(n_arr + 1e-8)

    # linear fit (1D): slope = FD (closed-form)
    x = log_inv_s - log_inv_s.mean()
    y = log_n     - log_n.mean()
    fd = (x * y).sum() / (x * x).sum()

    # bring to host if needed
    return float(fd), asnumpy(s_arr), asnumpy(n_arr)


def fract(subjid, image):
    from fractalbrain.asofi import asofi
    import logging
    import os
    import time
    import datetime

    # START TIME
    start_time = time.process_time()
    start_time_to_log = time.asctime(time.localtime(time.time()))
    NOW = datetime.datetime.now()
    DATE = NOW.strftime("%Y-%m-%d")
    TIME = NOW.strftime("%H-%M-%S")

    # LOG FILE SETTING
    imagepath = os.path.dirname(image)
    if not imagepath or imagepath == '.':
        imagepath = os.getcwd()
    os.makedirs(imagepath, exist_ok=True)
    prefix_base = os.path.basename(subjid)
    log_file_name = os.path.join(imagepath, f"{prefix_base}_fractal_{DATE}_{TIME}")

    log = logging.getLogger(log_file_name)
    hdlr = logging.FileHandler(log_file_name + '.log', mode="w")
    formatter = logging.Formatter(fmt='%(asctime)s - %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    hdlr.setFormatter(formatter)
    log.addHandler(hdlr)
    log.setLevel(logging.INFO)

    log.info('Started at %s', start_time_to_log)

    # FRACTAL ANALYSIS (original pipeline)
    asofi(subjid, image)

    # END TIME
    end_time = time.process_time()
    end_time_to_log = time.asctime(time.localtime(time.time()))
    elapsed_time = end_time - start_time
    log.info('#----------------------------------------')
    log.info('Started at %s', start_time_to_log)
    log.info('Ended at %s', end_time_to_log)
    log.info('fract-run-time-seconds %s', elapsed_time)
    return


if __name__ == "__main__":
    import argparse
    import os
    import textwrap
    import sys

    # MANAGEMENT OF ARGUMENTS, USAGE AND HELP
    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)

    parser = MyParser(
        prog='fractalbrain.fract',
        usage='%(prog)s [-h] [--device {cpu,cuda}] prefix image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
            Examples: 
            python -m fractalbrain.fract subjid image.nii.gz
            python -m fractalbrain.fract sub001 cerebralGM.nii.gz
            python -m fractalbrain.fract prefixes_list.txt NIfTI_list.txt
        ''')
    )

    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu",
        help="Backend for box counting (cpu=NumPy, cuda=CuPy)"
    )

    parser.add_argument('prefix', metavar='prefix',
                        help='the prefix name of the NIfTI image that will be processed or a file containing a list of prefixes')
    parser.add_argument('image', metavar='image',
                        help='the NIfTI image that will be processed or a file containing a list of NIfTI images. In the latter case, the fractal analysis will be performed on each NIfTI image sequentially')
    args = parser.parse_args()

    os.environ["FRACTALBRAIN_DEVICE"] = args.device

    # CHECK IF THE USER PASSED DIRECTLY THE NIFTI IMAGE OR A LIST
    imagefile = os.path.basename(args.image)
    imagename, image_extension1 = os.path.splitext(imagefile)
    imagename, image_extension2 = os.path.splitext(imagename)
    image_extension = image_extension2 + image_extension1
    if image_extension == '.nii' or image_extension == '.nii.gz':
        print("The prefix is: ", args.prefix)
        subjid = args.prefix
        print("The NIfTI image is: ", args.image)
        image = args.image
        fract(subjid, image)
        # fract(**vars(args))
    elif os.path.isfile(args.image) and os.path.isfile(args.prefix):
        print(args.image, "is a file containing a list of NIfTI images and", args.prefix, "is a file containing a list of prefixes")
        with open(args.prefix, 'r') as fid_subj_list, open(args.image, 'r') as fid_imgs_list:
            for x, y in zip(fid_subj_list, fid_imgs_list):
                subjid = x.strip()
                image = y.strip()
                print("subjid:", subjid)
                print("image", image)
                fract(subjid, image)
