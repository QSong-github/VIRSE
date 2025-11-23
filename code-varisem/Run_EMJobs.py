#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# The MIT License (MIT)
# Copyright (c) <2019> <The Whitehead Institute for Biomedical Research>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
Created on Tue Jul 30 2019

@author: harish
"""
import os
import EM_Files
import EM_Plots
from EM_Algorithm import Run_EM
import numpy as np


def Run_EMJob(X, bvfile_basename, MIN_ITS, INFO_THRESH, CONV_CUTOFF,
              SIG_THRESH, outplot_dir, K, CPUS, run):
    """
    Executes one VI/EM run for a given K.
    Automatically re-runs until π is not collapsed (i.e. 0.01 ≤ π ≤ 0.99).
    Keeps K=1 baseline always.
    """

    if K == 1:
        # Record number of reads for K=1
        EM_Plots.NumReads_File(bvfile_basename, X, outplot_dir)

    attempt = 1
    while True:
        print(f"[VARISEM] ▶ Starting attempt {attempt} for K={K}, run={run}")

        # ---- Run VI / EM ----
        EM_res = Run_EM(X, K, MIN_ITS, CONV_CUTOFF, CPUS)

        # ---- Handle failed or empty result ----
        if EM_res is None:
            print(f"[WARN] Run returned None for K={K}, attempt={attempt}. Retrying...\n")
            attempt += 1
            continue

        # ---- Unpack results ----
        log_like_list, final_mu, final_obs_pi, final_real_pi, resps, BIC = EM_res
        imbalance = (max(final_obs_pi) > 0.90 or min(final_obs_pi) < 0.10)

        # ---- If imbalanced, re-run with new seed ----
        if K > 1 and imbalance:
            print(f"[WARN] Extremely imbalanced mixture weights detected: π = {final_obs_pi}")
            print(f"[RETRY] Re-running K={K}, run={run} with new random seed...\n")

            # 更换随机种子以避免陷入死循环
            new_seed = np.random.randint(1, 10**6)
            np.random.seed(new_seed)
            attempt += 1
            continue

        # ---- Break only if successful ----
        print(f"[SUCCESS] Valid mixture obtained: π = {final_obs_pi}\n")
        break

    # ---- Generate plots ----
    EM_Plots.Run_Plots(
        bvfile_basename, X, K, log_like_list, final_mu,
        final_obs_pi, final_real_pi, resps, BIC, outplot_dir, run
    )

    print(f"[DONE] Successfully completed K={K}, run={run} after {attempt} attempt(s)\n")
