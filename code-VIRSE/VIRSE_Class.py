#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# VIRSE — Variational Inference for RNA Structure Ensembles
"""
Bit-vector data container (BV_Object).
"""
import numpy as np
import random


class BV_Object():
    """
    """
    def __init__(self, bit_vectors, mut_popavg, n_discard, ref_file,
                 ref, seq, infiles_dir, indices):
        BV_Matrix, BV_Abundance, n_occur = [], [], {}
        for bit_vector in bit_vectors:
            bit_vector = tuple(bit_vector)  # Change to a tuple
            if bit_vector in n_occur:
                n_occur[bit_vector] += 1
            else:
                n_occur[bit_vector] = 1
        for bit_vector in n_occur:
            bv = np.array(list(map(float, bit_vector)))  # Convert to float
            BV_Matrix.append(bv)
            BV_Abundance.append(n_occur[bit_vector])

        BV_Matrix = np.array(BV_Matrix)
        BV_Abundance = np.array(BV_Abundance)
        self.BV_Matrix = BV_Matrix  # Only unique bit vectors
        self.BV_Abundance = BV_Abundance  # Abundance of each bit vector
        self.n_occur = n_occur
        self.n_bitvectors = len(bit_vectors)
        self.n_unique_bitvectors = len(n_occur.keys())
        self.n_discard = n_discard
        self.mut_popavg = mut_popavg
        self.ref = ref
        self.ref_file = ref_file
        self.seq = seq
        self.infiles_dir = infiles_dir
        self.indices = indices
