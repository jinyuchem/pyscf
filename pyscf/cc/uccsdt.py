#!/usr/bin/env python
# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Yu Jin <yjin@flatironinstitute.org>
#

'''
UHF-CCSDT with T3 amplitudes stored only for the significant part
'''

import numpy as np
import numpy
from functools import reduce
import ctypes
from pyscf import ao2mo, lib
from pyscf.lib import logger
from pyscf.mp.mp2 import get_e_hf
from pyscf.mp.ump2 import get_nocc, get_nmo, get_frozen_mask
from pyscf.cc import ccsd, _ccsd
from pyscf.cc.rccsdt import einsum_, run_diis, _finalize


def call_unpack_6fold_antisymm_c(t3, t3_blk, map_o, mask_o, map_v, mask_v,
                                 i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, c0, c1,
                                 nocc, nvir, blk_i, blk_j, blk_k, blk_a, blk_b, blk_c):
    assert t3.dtype == np.float64 and t3_blk.dtype == np.float64
    assert map_o.dtype == np.int64 and mask_o.dtype == np.bool_
    assert map_v.dtype == np.int64 and mask_v.dtype == np.bool_

    # Ensure arrays are contiguous
    t3_c = np.ascontiguousarray(t3)
    t3_blk_c = np.ascontiguousarray(t3_blk)
    map_o_c = np.ascontiguousarray(map_o)
    mask_o_c = np.ascontiguousarray(mask_o)
    map_v_c = np.ascontiguousarray(map_v)
    mask_v_c = np.ascontiguousarray(mask_v)

    drv = _ccsd.libcc.unpack_6fold_antisymm_c
    drv(
        t3_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        t3_blk_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        map_o_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        mask_o_c.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
        map_v_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        mask_v_c.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
        ctypes.c_int64(i0), ctypes.c_int64(i1),
        ctypes.c_int64(j0), ctypes.c_int64(j1),
        ctypes.c_int64(k0), ctypes.c_int64(k1),
        ctypes.c_int64(a0), ctypes.c_int64(a1),
        ctypes.c_int64(b0), ctypes.c_int64(b1),
        ctypes.c_int64(c0), ctypes.c_int64(c1),
        ctypes.c_int64(nocc), ctypes.c_int64(nvir),
        ctypes.c_int64(blk_i), ctypes.c_int64(blk_j), ctypes.c_int64(blk_k),
        ctypes.c_int64(blk_a), ctypes.c_int64(blk_b), ctypes.c_int64(blk_c)
    )
    return t3_blk

def call_update_packed_6fold_antisymm_c(t3, t3_blk, map_o, map_v,
                                       i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, c0, c1,
                                       nocc, nvir, blk_i, blk_j, blk_k, blk_a, blk_b, blk_c, alpha, beta):
    assert t3.dtype == np.float64 and t3_blk.dtype == np.float64
    assert map_o.dtype == np.int64 and map_v.dtype == np.int64

    # Ensure arrays are contiguous
    t3_c = np.ascontiguousarray(t3)
    t3_blk_c = np.ascontiguousarray(t3_blk)
    map_o_c = np.ascontiguousarray(map_o)
    map_v_c = np.ascontiguousarray(map_v)

    drv = _ccsd.libcc.update_packed_6fold_antisymm_c
    drv(
        t3_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        t3_blk_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        map_o_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        map_v_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(i0), ctypes.c_int64(i1),
        ctypes.c_int64(j0), ctypes.c_int64(j1),
        ctypes.c_int64(k0), ctypes.c_int64(k1),
        ctypes.c_int64(a0), ctypes.c_int64(a1),
        ctypes.c_int64(b0), ctypes.c_int64(b1),
        ctypes.c_int64(c0), ctypes.c_int64(c1),
        ctypes.c_int64(nocc), ctypes.c_int64(nvir),
        ctypes.c_int64(blk_i), ctypes.c_int64(blk_j), ctypes.c_int64(blk_k),
        ctypes.c_int64(blk_a), ctypes.c_int64(blk_b), ctypes.c_int64(blk_c),
        ctypes.c_double(alpha), ctypes.c_double(beta)
    )
    return t3

def call_unpack_2fold_antisymm_c(t3, t3_blk, map_o, mask_o, map_v, mask_v,
                                 i0, i1, j0, j1, a0, a1, b0, b1, k0, k1, c0, c1,
                                 nocc, nvir, dim4, dim5, blk_i, blk_j, blk_a, blk_b):
    assert t3.dtype == np.float64 and t3_blk.dtype == np.float64
    assert map_o.dtype == np.int64 and mask_o.dtype == np.bool_
    assert map_v.dtype == np.int64 and mask_v.dtype == np.bool_

    # Ensure arrays are contiguous
    t3_c = np.ascontiguousarray(t3)
    t3_blk_c = np.ascontiguousarray(t3_blk)
    map_o_c = np.ascontiguousarray(map_o)
    mask_o_c = np.ascontiguousarray(mask_o)
    map_v_c = np.ascontiguousarray(map_v)
    mask_v_c = np.ascontiguousarray(mask_v)

    drv = _ccsd.libcc.unpack_2fold_antisymm_c
    drv(
        t3_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        t3_blk_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        map_o_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        mask_o_c.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
        map_v_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        mask_v_c.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
        ctypes.c_int64(i0), ctypes.c_int64(i1),
        ctypes.c_int64(j0), ctypes.c_int64(j1),
        ctypes.c_int64(a0), ctypes.c_int64(a1),
        ctypes.c_int64(b0), ctypes.c_int64(b1),
        ctypes.c_int64(k0), ctypes.c_int64(k1),
        ctypes.c_int64(c0), ctypes.c_int64(c1),
        ctypes.c_int64(nocc), ctypes.c_int64(nvir),
        ctypes.c_int64(dim4), ctypes.c_int64(dim5),
        ctypes.c_int64(blk_i), ctypes.c_int64(blk_j),
        ctypes.c_int64(blk_a), ctypes.c_int64(blk_b),
    )
    return t3_blk

def call_update_packed_2fold_antisymm_c(t3, t3_blk, map_o, map_v,
                                        i0, i1, j0, j1, a0, a1, b0, b1, k0, k1, c0, c1,
                                        nocc, nvir, dim4, dim5, blk_i, blk_j, blk_a, blk_b, alpha, beta):
    assert t3.dtype == np.float64 and t3_blk.dtype == np.float64
    assert map_o.dtype == np.int64 and map_v.dtype == np.int64

    # Ensure arrays are contiguous
    t3_c = np.ascontiguousarray(t3)
    t3_blk_c = np.ascontiguousarray(t3_blk)
    map_o_c = np.ascontiguousarray(map_o)
    map_v_c = np.ascontiguousarray(map_v)

    drv = _ccsd.libcc.update_packed_2fold_antisymm_c
    drv(
        t3_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        t3_blk_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        map_o_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        map_v_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(i0), ctypes.c_int64(i1),
        ctypes.c_int64(j0), ctypes.c_int64(j1),
        ctypes.c_int64(a0), ctypes.c_int64(a1),
        ctypes.c_int64(b0), ctypes.c_int64(b1),
        ctypes.c_int64(k0), ctypes.c_int64(k1),
        ctypes.c_int64(c0), ctypes.c_int64(c1),
        ctypes.c_int64(nocc), ctypes.c_int64(nvir),
        ctypes.c_int64(dim4), ctypes.c_int64(dim5),
        ctypes.c_int64(blk_i), ctypes.c_int64(blk_j),
        ctypes.c_int64(blk_a), ctypes.c_int64(blk_b),
        ctypes.c_double(alpha), ctypes.c_double(beta)
    )
    return t3

def _tril2cube_6fold(no):
    no3 = no * (no - 1) * (no - 2) // 6
    tril2cube_map = np.zeros((6, no, no, no), dtype=np.int64)
    tril2cube_mask = np.zeros((6, no, no, no), dtype=np.bool_)
    i, j, k = np.meshgrid(np.arange(no), np.arange(no), np.arange(no), indexing='ij')
    t3_map = np.where((i < j) & (j < k))
    tril2cube_map[0, t3_map[0], t3_map[1], t3_map[2]] = np.arange(no3)
    tril2cube_map[1, t3_map[0], t3_map[2], t3_map[1]] = np.arange(no3)
    tril2cube_map[2, t3_map[1], t3_map[0], t3_map[2]] = np.arange(no3)
    tril2cube_map[3, t3_map[1], t3_map[2], t3_map[0]] = np.arange(no3)
    tril2cube_map[4, t3_map[2], t3_map[0], t3_map[1]] = np.arange(no3)
    tril2cube_map[5, t3_map[2], t3_map[1], t3_map[0]] = np.arange(no3)
    tril2cube_mask[0] = (i < j) & (j < k)
    tril2cube_mask[1] = (i < k) & (k < j)
    tril2cube_mask[2] = (j < i) & (i < k)
    tril2cube_mask[3] = (k < i) & (i < j)
    tril2cube_mask[4] = (j < k) & (k < i)
    tril2cube_mask[5] = (k < j) & (j < i)
    # tril2cube_sign = np.array([1.0, -1.0, -1.0, 1.0, 1.0, -1.0])
    return tril2cube_map, tril2cube_mask

def _tril2cube_2fold(no):
    no2 = no * (no - 1) // 2
    tril2cube_map = np.zeros((2, no, no), dtype=np.int64)
    tril2cube_mask = np.zeros((2, no, no), dtype=np.bool_)
    i, j = np.meshgrid(np.arange(no), np.arange(no), indexing='ij')
    t3_map = np.where(i < j)
    tril2cube_map[0, t3_map[0], t3_map[1]] = np.arange(no2)
    tril2cube_map[1, t3_map[1], t3_map[0]] = np.arange(no2)
    tril2cube_mask[0] = i < j
    tril2cube_mask[1] = i > j
    # tril2cube_sign = np.array([1.0, -1.0])
    return tril2cube_map, tril2cube_mask

def setup_tril2cube_t3_uhf(mycc):
    mycc.t2c_map_6f_oa, mycc.t2c_mask_6f_oa = _tril2cube_6fold(mycc.nocca)
    mycc.t2c_map_2f_oa, mycc.t2c_mask_2f_oa = _tril2cube_2fold(mycc.nocca)
    mycc.t2c_map_6f_va, mycc.t2c_mask_6f_va = _tril2cube_6fold(mycc.nvira)
    mycc.t2c_map_2f_va, mycc.t2c_mask_2f_va = _tril2cube_2fold(mycc.nvira)
    mycc.t2c_map_6f_ob, mycc.t2c_mask_6f_ob = _tril2cube_6fold(mycc.noccb)
    mycc.t2c_map_2f_ob, mycc.t2c_mask_2f_ob = _tril2cube_2fold(mycc.noccb)
    mycc.t2c_map_6f_vb, mycc.t2c_mask_6f_vb = _tril2cube_6fold(mycc.nvirb)
    mycc.t2c_map_2f_vb, mycc.t2c_mask_2f_vb = _tril2cube_2fold(mycc.nvirb)
    return mycc

def build_t2_indices_uhf(mycc):
    # t2aa
    i_idx, j_idx = np.triu_indices(mycc.nocca, k=1)
    a_idx, b_idx = np.triu_indices(mycc.nvira, k=1)
    I = np.repeat(i_idx, len(a_idx))
    J = np.repeat(j_idx, len(a_idx))
    A = np.tile(a_idx, len(i_idx))
    B = np.tile(b_idx, len(i_idx))
    t2aa_idx = (I, J, A, B)
    # t2bb
    i_idx, j_idx = np.triu_indices(mycc.noccb, k=1)
    a_idx, b_idx = np.triu_indices(mycc.nvirb, k=1)
    I = np.repeat(i_idx, len(a_idx))
    J = np.repeat(j_idx, len(a_idx))
    A = np.tile(a_idx, len(i_idx))
    B = np.tile(b_idx, len(i_idx))
    t2bb_idx = (I, J, A, B)

    mycc.t2aa_idx = t2aa_idx
    mycc.t2bb_idx = t2bb_idx
    return mycc

def _unp_aaa(mycc, t3, t3_blk, i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, c0, c1,
            blk_i=None, blk_j=None, blk_k=None, blk_a=None, blk_b=None, blk_c=None):
    if not blk_i: blk_i=mycc.blksize_o_aaa
    if not blk_j: blk_j=mycc.blksize_o_aaa
    if not blk_k: blk_k=mycc.blksize_o_aaa
    if not blk_a: blk_a=mycc.blksize_v_aaa
    if not blk_b: blk_b=mycc.blksize_v_aaa
    if not blk_c: blk_c=mycc.blksize_v_aaa
    call_unpack_6fold_antisymm_c(t3, t3_blk, mycc.t2c_map_6f_oa, mycc.t2c_mask_6f_oa,
                                mycc.t2c_map_6f_va, mycc.t2c_mask_6f_va,
                                i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, c0, c1, mycc.nocca, mycc.nvira,
                                blk_i, blk_j, blk_k, blk_a, blk_b, blk_c)

def _unp_bbb(mycc, t3, t3_blk, i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, c0, c1,
            blk_i=None, blk_j=None, blk_k=None, blk_a=None, blk_b=None, blk_c=None):
    if not blk_i: blk_i=mycc.blksize_o_aaa
    if not blk_j: blk_j=mycc.blksize_o_aaa
    if not blk_k: blk_k=mycc.blksize_o_aaa
    if not blk_a: blk_a=mycc.blksize_v_aaa
    if not blk_b: blk_b=mycc.blksize_v_aaa
    if not blk_c: blk_c=mycc.blksize_v_aaa
    call_unpack_6fold_antisymm_c(t3, t3_blk, mycc.t2c_map_6f_ob, mycc.t2c_mask_6f_ob,
                                mycc.t2c_map_6f_vb, mycc.t2c_mask_6f_vb,
                                i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, c0, c1, mycc.noccb, mycc.nvirb,
                                blk_i, blk_j, blk_k, blk_a, blk_b, blk_c)

def _update_packed_aaa(mycc, t3, t3_blk, i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, c0, c1,
                        blk_i=None, blk_j=None, blk_k=None, blk_a=None, blk_b=None, blk_c=None,
                        alpha=1.0, beta=0.0):
    if not blk_i: blk_i=mycc.blksize_o_aaa
    if not blk_j: blk_j=mycc.blksize_o_aaa
    if not blk_k: blk_k=mycc.blksize_o_aaa
    if not blk_a: blk_a=mycc.blksize_v_aaa
    if not blk_b: blk_b=mycc.blksize_v_aaa
    if not blk_c: blk_c=mycc.blksize_v_aaa

    call_update_packed_6fold_antisymm_c(t3, t3_blk, mycc.t2c_map_6f_oa, mycc.t2c_map_6f_va,
        i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, c0, c1, mycc.nocca, mycc.nvira,
        blk_i, blk_j, blk_k, blk_a, blk_b, blk_c, alpha=alpha, beta=beta)

def _update_packed_bbb(mycc, t3, t3_blk, i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, c0, c1,
                        blk_i=None, blk_j=None, blk_k=None, blk_a=None, blk_b=None, blk_c=None,
                        alpha=1.0, beta=0.0):
    if not blk_i: blk_i=mycc.blksize_o_aaa
    if not blk_j: blk_j=mycc.blksize_o_aaa
    if not blk_k: blk_k=mycc.blksize_o_aaa
    if not blk_a: blk_a=mycc.blksize_v_aaa
    if not blk_b: blk_b=mycc.blksize_v_aaa
    if not blk_c: blk_c=mycc.blksize_v_aaa

    call_update_packed_6fold_antisymm_c(t3, t3_blk, mycc.t2c_map_6f_ob, mycc.t2c_map_6f_vb,
        i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, c0, c1, mycc.noccb, mycc.nvirb,
        blk_i, blk_j, blk_k, blk_a, blk_b, blk_c, alpha=alpha, beta=beta)

def _unp_aab(mycc, t3, t3_blk, i0, i1, j0, j1, a0, a1, b0, b1, k0=None, k1=None, c0=None, c1=None,
            blk_i=None, blk_j=None, blk_a=None, blk_b=None, dim4=None, dim5=None):
    if not k0: k0 = 0
    if not k1: k1 = mycc.noccb
    if not c0: c0 = 0
    if not c1: c1 = mycc.nvirb
    if not blk_i: blk_i = mycc.blksize_o_aab
    if not blk_j: blk_j = mycc.blksize_o_aab
    if not blk_a: blk_a = mycc.blksize_v_aab
    if not blk_b: blk_b = mycc.blksize_v_aab
    if not dim4: dim4 = mycc.noccb
    if not dim5: dim5 = mycc.nvirb
    call_unpack_2fold_antisymm_c(t3, t3_blk, mycc.t2c_map_2f_oa, mycc.t2c_mask_2f_oa,
                                mycc.t2c_map_2f_va, mycc.t2c_mask_2f_va,
                                i0, i1, j0, j1, a0, a1, b0, b1, k0, k1, c0, c1,
                                mycc.nocca, mycc.nvira, dim4, dim5, blk_i, blk_j, blk_a, blk_b)

def _unp_bba(mycc, t3, t3_blk, i0, i1, j0, j1, a0, a1, b0, b1, k0=None, k1=None, c0=None, c1=None,
            blk_i=None, blk_j=None, blk_a=None, blk_b=None, dim4=None, dim5=None):
    if not k0: k0 = 0
    if not k1: k1 = mycc.nocca
    if not c0: c0 = 0
    if not c1: c1 = mycc.nvira
    if not blk_i: blk_i = mycc.blksize_o_aab
    if not blk_j: blk_j = mycc.blksize_o_aab
    if not blk_a: blk_a = mycc.blksize_v_aab
    if not blk_b: blk_b = mycc.blksize_v_aab
    if not dim4: dim4 = mycc.nocca
    if not dim5: dim5 = mycc.nvira
    call_unpack_2fold_antisymm_c(t3, t3_blk, mycc.t2c_map_2f_ob, mycc.t2c_mask_2f_ob,
                                mycc.t2c_map_2f_vb, mycc.t2c_mask_2f_vb,
                                i0, i1, j0, j1, a0, a1, b0, b1, k0, k1, c0, c1,
                                mycc.noccb, mycc.nvirb, dim4, dim5, blk_i, blk_j, blk_a, blk_b)

def _update_packed_aab(mycc, t3, t3_blk, i0, i1, j0, j1, a0, a1, b0, b1,
                        k0=None, k1=None, c0=None, c1=None, blk_i=None, blk_j=None, blk_a=None, blk_b=None,
                        dim4=None, dim5=None, alpha=1.0, beta=0.0):
    if not k0: k0 = 0
    if not k1: k1 = mycc.noccb
    if not c0: c0 = 0
    if not c1: c1 = mycc.nvirb
    if not blk_i: blk_i = mycc.blksize_o_aab
    if not blk_j: blk_j = mycc.blksize_o_aab
    if not blk_a: blk_a = mycc.blksize_v_aab
    if not blk_b: blk_b = mycc.blksize_v_aab
    if not dim4: dim4 = mycc.noccb
    if not dim5: dim5 = mycc.nvirb

    call_update_packed_2fold_antisymm_c(t3, t3_blk, mycc.t2c_map_2f_oa, mycc.t2c_map_2f_va,
        i0, i1, j0, j1, a0, a1, b0, b1, k0, k1, c0, c1, mycc.nocca, mycc.nvira, dim4, dim5,
        blk_i, blk_j, blk_a, blk_b, alpha=alpha, beta=beta)

def _update_packed_bba(mycc, t3, t3_blk, i0, i1, j0, j1, a0, a1, b0, b1,
                        k0=None, k1=None, c0=None, c1=None, blk_i=None, blk_j=None, blk_a=None, blk_b=None,
                        dim4=None, dim5=None, alpha=1.0, beta=0.0):
    if not k0: k0 = 0
    if not k1: k1 = mycc.nocca
    if not c0: c0 = 0
    if not c1: c1 = mycc.nvira
    if not blk_i: blk_i = mycc.blksize_o_aab
    if not blk_j: blk_j = mycc.blksize_o_aab
    if not blk_a: blk_a = mycc.blksize_v_aab
    if not blk_b: blk_b = mycc.blksize_v_aab
    if not dim4: dim4 = mycc.nocca
    if not dim5: dim5 = mycc.nvira

    call_update_packed_2fold_antisymm_c(t3, t3_blk, mycc.t2c_map_2f_ob, mycc.t2c_map_2f_vb,
        i0, i1, j0, j1, a0, a1, b0, b1, k0, k1, c0, c1, mycc.noccb, mycc.nvirb, dim4, dim5,
        blk_i, blk_j, blk_a, blk_b, alpha=alpha, beta=beta)

def init_amps(mycc, eris=None):
    time0 = logger.process_clock(), logger.perf_counter()
    if eris is not None:
        mo_energy = eris.mo_energy
        focka, fockb = eris.focka, eris.fockb
        nocca, noccb = eris.nocca, eris.noccb
        eris_oovv = eris.ovov.transpose(0, 2, 1, 3)
        eris_OOVV = eris.OVOV.transpose(0, 2, 1, 3)
        eris_oOvV = eris.ovOV.transpose(0, 2, 1, 3)
    else:
        mo_energy = mycc.mo_energy
        focka, fockb = mycc.focka, mycc.fockb
        nocca, noccb = mycc.nocca, mycc.noccb
        eris_oovv = mycc.erisaa[:nocca, :nocca, nocca:, nocca:]
        eris_OOVV = mycc.erisbb[:noccb, :noccb, noccb:, noccb:]
        eris_oOvV = mycc.erisab[:nocca, :noccb, nocca:, noccb:]

    fova = focka[:nocca, nocca:]
    fovb = fockb[:noccb, noccb:]
    mo_ea_o = mo_energy[0][:nocca]
    mo_ea_v = mo_energy[0][nocca:]
    mo_eb_o = mo_energy[1][:noccb]
    mo_eb_v = mo_energy[1][noccb:]
    eia_a = lib.direct_sum('i-a->ia', mo_ea_o, mo_ea_v)
    eia_b = lib.direct_sum('i-a->ia', mo_eb_o, mo_eb_v)

    t1a = fova.conj() / eia_a
    t1b = fovb.conj() / eia_b

    t2aa = eris_oovv.conj() / lib.direct_sum('ia+jb->ijab', eia_a, eia_a)
    t2ab = eris_oOvV.conj() / lib.direct_sum('ia+jb->ijab', eia_a, eia_b)
    t2bb = eris_OOVV.conj() / lib.direct_sum('ia+jb->ijab', eia_b, eia_b)
    # NOTE: The definition of t2ab here is different from that in the uccsd.py code
    t2ab = t2ab.transpose(0, 2, 1, 3)
    t2aa = t2aa - t2aa.transpose(0, 1, 3, 2)
    t2bb = t2bb - t2bb.transpose(0, 1, 3, 2)
    e  =        np.einsum('iaJB,iJaB', t2ab, eris_oOvV)
    e += 0.25 * np.einsum('ijab,ijab', t2aa, eris_oovv)
    e -= 0.25 * np.einsum('ijab,ijba', t2aa, eris_oovv)
    e += 0.25 * np.einsum('ijab,ijab', t2bb, eris_OOVV)
    e -= 0.25 * np.einsum('ijab,ijba', t2bb, eris_OOVV)
    mycc.emp2 = e.real
    logger.info(mycc, 'Init t2, MP2 energy = %.15g', mycc.emp2)
    logger.timer(mycc, 'init mp2', *time0)
    return mycc.emp2, (t1a, t1b), (t2aa, t2ab, t2bb)

def energy(mycc, t1=None, t2=None, eris=None):
    '''UCC correlation energy'''
    if t1 is None: t1 = (mycc.t1a, mycc.t1b)
    if t2 is None: t2 = (mycc.t2aa, mycc.t2ab, mycc.t2bb)

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, nvira, noccb, nvirb = t2ab.shape

    if eris is not None:
        focka, fockb = eris.fock[0], eris.fock[1]
        eris_oovv = eris.ovov.transpose(0, 2, 1, 3)
        eris_OOVV = eris.OVOV.transpose(0, 2, 1, 3)
        eris_oOvV = eris.ovOV.transpose(0, 2, 1, 3)
    else:
        focka, fockb = mycc.focka, mycc.fockb
        eris_oovv = mycc.erisaa[:nocca, :nocca, nocca:, nocca:]
        eris_OOVV = mycc.erisbb[:noccb, :noccb, noccb:, noccb:]
        eris_oOvV = mycc.erisab[:nocca, :noccb, nocca:, noccb:]

    fova = focka[:nocca, nocca:]
    fovb = fockb[:noccb, noccb:]
    # NOTE: need double check
    ess  = np.einsum('ia,ia', fova, t1a)
    ess += np.einsum('ia,ia', fovb, t1b)
    ess += 0.25 * np.einsum('ijab,ijab', t2aa, eris_oovv)
    ess -= 0.25 * np.einsum('ijab,ijba', t2aa, eris_oovv)
    ess += 0.25 * np.einsum('ijab,ijab', t2bb, eris_OOVV)
    ess -= 0.25 * np.einsum('ijab,ijba', t2bb, eris_OOVV)
    eos  =        np.einsum('iaJB,iJaB', t2ab, eris_oOvV)
    ess += 0.5 * lib.einsum('ia,jb,ijab', t1a, t1a, eris_oovv)
    ess -= 0.5 * lib.einsum('ia,jb,ijba', t1a, t1a, eris_oovv)
    ess += 0.5 * lib.einsum('ia,jb,ijab', t1b, t1b, eris_OOVV)
    ess -= 0.5 * lib.einsum('ia,jb,ijba', t1b, t1b, eris_OOVV)
    eos +=       lib.einsum('ia,JB,iJaB', t1a, t1b, eris_oOvV)

    if abs((ess + eos).imag) > 1e-4:
        logger.warn(mycc, 'Non-zero imaginary part found in %s energy %s', mycc.__class__.name, ess + eos)

    mycc.e_corr = lib.tag_array((ess + eos).real, e_corr_ss=ess.real, e_corr_os=eos.real)

    return mycc.e_corr.real

def update_xy_uhf(mycc, t1):
    nocca, noccb = mycc.nocca, mycc.noccb
    nmoa, nmob = mycc.nmoa, mycc.nmob
    t1a, t1b = t1
    xa = np.eye(nmoa, dtype=t1a.dtype)
    xb = np.eye(nmob, dtype=t1b.dtype)
    xa[nocca:, :nocca] -= t1a.T
    xb[noccb:, :noccb] -= t1b.T
    ya = np.eye(nmoa, dtype=t1a.dtype)
    yb = np.eye(nmob, dtype=t1b.dtype)
    ya[:nocca, nocca:] += t1a
    yb[:noccb, noccb:] += t1b
    return xa, xb, ya, yb

def update_fock_uhf(mycc, xa, xb, ya, yb, t1):
    nocca, noccb = mycc.nocca, mycc.noccb
    t1a, t1b = t1
    focka, fockb = mycc.focka, mycc.fockb
    erisaa, erisab, erisbb = mycc.erisaa, mycc.erisab, mycc.erisbb
    t1_focka = focka + einsum_('risa,ia->rs', erisaa[:, :nocca, :, nocca:], t1a)
    t1_focka += einsum_('risa,ia->rs', erisab[:, :noccb, :, noccb:], t1b)
    t1_focka -= einsum_('rias,ia->rs', erisaa[:, :nocca, nocca:, :], t1a)
    t1_focka = xa @ t1_focka @ ya.T
    t1_fockb = fockb + einsum_('risa,ia->rs', erisbb[:, :noccb, :, noccb:], t1b)
    t1_fockb += einsum_('iras,ia->rs', erisab[:nocca, :, nocca:, :], t1a)
    t1_fockb -= einsum_('rias,ia->rs', erisbb[:, :noccb, noccb:, :], t1b)
    t1_fockb = xb @ t1_fockb @ yb.T
    return t1_focka, t1_fockb

def update_eris_uhf(mycc, xa, xb, ya, yb):
    '''t1_erisaa and t1_erisbb are anti-symmetrized'''
    erisaa, erisab, erisbb = mycc.erisaa, mycc.erisab, mycc.erisbb
    t1_erisaa = einsum_('tvuw,pt->pvuw', erisaa, xa)
    t1_erisaa = einsum_('pvuw,rv->pruw', t1_erisaa, xa)
    t1_erisaa = t1_erisaa.transpose(2, 3, 0, 1)
    if not t1_erisaa.flags['C_CONTIGUOUS']:
        t1_erisaa = np.ascontiguousarray(t1_erisaa)
    t1_erisaa = einsum_('uwpr,qu->qwpr', t1_erisaa, ya)
    t1_erisaa = einsum_('qwpr,sw->qspr', t1_erisaa, ya)
    t1_erisaa = t1_erisaa.transpose(2, 3, 0, 1)
    # anti-symmetrization
    t1_erisaa -= t1_erisaa.transpose(0, 1, 3, 2)

    t1_erisbb = einsum_('tvuw,pt->pvuw', erisbb, xb)
    t1_erisbb = einsum_('pvuw,rv->pruw', t1_erisbb, xb)
    t1_erisbb = t1_erisbb.transpose(2, 3, 0, 1)
    if not t1_erisbb.flags['C_CONTIGUOUS']:
        t1_erisbb = np.ascontiguousarray(t1_erisbb)
    t1_erisbb = einsum_('uwpr,qu->qwpr', t1_erisbb, yb)
    t1_erisbb = einsum_('qwpr,sw->qspr', t1_erisbb, yb)
    t1_erisbb = t1_erisbb.transpose(2, 3, 0, 1)
    # anti-symmetrization
    t1_erisbb -= t1_erisbb.transpose(0, 1, 3, 2)

    t1_erisab = einsum_('tvuw,pt->pvuw', erisab, xa)
    t1_erisab = einsum_('pvuw,rv->pruw', t1_erisab, xb)
    t1_erisab = t1_erisab.transpose(2, 3, 0, 1)
    if not t1_erisab.flags['C_CONTIGUOUS']:
        t1_erisab = np.ascontiguousarray(t1_erisab)
    t1_erisab = einsum_('uwpr,qu->qwpr', t1_erisab, ya)
    t1_erisab = einsum_('qwpr,sw->qspr', t1_erisab, yb)
    t1_erisab = t1_erisab.transpose(2, 3, 0, 1)
    return t1_erisaa, t1_erisab, t1_erisbb

def update_t1_fock_eris_uhf(mycc, t1):
    xa, xb, ya, yb = update_xy_uhf(mycc, t1)
    mycc.t1_focka, mycc.t1_fockb = update_fock_uhf(mycc, xa, xb, ya, yb, t1)
    mycc.t1_erisaa, mycc.t1_erisab, mycc.t1_erisbb = update_eris_uhf(mycc, xa, xb, ya, yb)
    return mycc

def intermediates_t1t2_uhf(mycc, t2):
    nocca, noccb = mycc.nocca, mycc.noccb
    t1_focka, t1_fockb = mycc.t1_focka, mycc.t1_fockb
    t1_erisaa, t1_erisab, t1_erisbb = mycc.t1_erisaa, mycc.t1_erisab, mycc.t1_erisbb
    t2aa, t2ab, t2bb = t2
    # aa
    tf_vv = t1_focka[nocca:, nocca:].copy()
    einsum_('klcd,klbd->bc', t1_erisaa[:nocca, :nocca, nocca:, nocca:], t2aa, out=tf_vv, alpha=-0.5, beta=1.0)
    einsum_('lkcd,lbkd->bc', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=tf_vv, alpha=-1.0, beta=1.0)
    tf_oo = t1_focka[:nocca, :nocca].copy()
    einsum_('klcd,jlcd->kj', t1_erisaa[:nocca, :nocca, nocca:, nocca:], t2aa, out=tf_oo, alpha=0.5, beta=1.0)
    einsum_('kldc,jdlc->kj', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=tf_oo, alpha=1.0, beta=1.0)
    W_oooo = t1_erisaa[:nocca, :nocca, :nocca, :nocca].copy()
    einsum_('klcd,ijcd->klij', t1_erisaa[:nocca, :nocca, nocca:, nocca:], t2aa, out=W_oooo, alpha=0.5, beta=1.0)
    W_ovvo = t1_erisaa[:nocca, nocca:, nocca:, :nocca].copy()
    einsum_('klcd,jlbd->kbcj', t1_erisaa[:nocca, :nocca, nocca:, nocca:], t2aa, out=W_ovvo, alpha=0.5, beta=1.0)
    einsum_('klcd,jbld->kbcj', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=W_ovvo, alpha=0.5, beta=1.0)
    W_OvVo = t1_erisab[nocca:, :noccb, :nocca, noccb:].transpose(1, 0, 3, 2).copy()
    einsum_('klcd,jbld->kbcj', t1_erisbb[:noccb, :noccb, noccb:, noccb:], t2ab, out=W_OvVo, alpha=0.5, beta=1.0)
    einsum_('lkdc,jlbd->kbcj', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2aa, out=W_OvVo, alpha=0.5, beta=1.0)
    # bb
    tf_VV = t1_fockb[noccb:, noccb:].copy()
    einsum_('klcd,klbd->bc', t1_erisbb[:noccb, :noccb, noccb:, noccb:], t2bb, out=tf_VV, alpha=-0.5, beta=1.0)
    einsum_('kldc,kdlb->bc', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=tf_VV, alpha=-1.0, beta=1.0)
    tf_OO = t1_fockb[:noccb, :noccb].copy()
    einsum_('klcd,jlcd->kj', t1_erisbb[:noccb, :noccb, noccb:, noccb:], t2bb, out=tf_OO, alpha=0.5, beta=1.0)
    einsum_('lkcd,lcjd->kj', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=tf_OO, alpha=1.0, beta=1.0)
    W_OOOO = t1_erisbb[:noccb, :noccb, :noccb, :noccb].copy()
    einsum_('klcd,ijcd->klij', t1_erisbb[:noccb, :noccb, noccb:, noccb:], t2bb, out=W_OOOO, alpha=0.5, beta=1.0)
    W_OVVO = t1_erisbb[:noccb, noccb:, noccb:, :noccb].copy()
    einsum_('klcd,jlbd->kbcj', t1_erisbb[:noccb, :noccb, noccb:, noccb:], t2bb, out=W_OVVO, alpha=0.5, beta=1.0)
    einsum_('lkdc,ldjb->kbcj', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=W_OVVO, alpha=0.5, beta=1.0)
    W_oVvO = t1_erisab[:nocca, noccb:, nocca:, :noccb].copy()
    einsum_('klcd,ldjb->kbcj', t1_erisaa[:nocca, :nocca, nocca:, nocca:], t2ab, out=W_oVvO, alpha=0.5, beta=1.0)
    einsum_('klcd,jlbd->kbcj', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2bb, out=W_oVvO, alpha=0.5, beta=1.0)
    # ab
    W_oOoO = t1_erisab[:nocca, :noccb, :nocca, :noccb].copy()
    einsum_('klcd,icjd->klij', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=W_oOoO, alpha=1.0, beta=1.0)
    W_vOvO = - t1_erisab[nocca:, :noccb, nocca:, :noccb]
    einsum_('lkcd,lajd->akcj', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=W_vOvO, alpha=0.5, beta=1.0)
    W_VoVo = - t1_erisab[:nocca, noccb:, :nocca, noccb:].transpose(1, 0, 3, 2)
    einsum_('kldc,idlb->bkci', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=W_VoVo, alpha=0.5, beta=1.0)
    W_vovo = - t1_erisaa[nocca:, :nocca, nocca:, :nocca]
    einsum_('klcd,lida->akci', t1_erisaa[:nocca, :nocca, nocca:, nocca:], t2aa, out=W_vovo, alpha=0.5, beta=1.0)
    einsum_('klcd,iald->akci', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=W_vovo, alpha=0.5, beta=1.0)
    W_VOVO = - t1_erisbb[noccb:, :noccb, noccb:, :noccb]
    einsum_('klcd,ljdb->bkcj', t1_erisbb[:noccb, :noccb, noccb:, noccb:], t2bb, out=W_VOVO, alpha=0.5, beta=1.0)
    einsum_('lkdc,ldjb->bkcj', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=W_VOVO, alpha=0.5, beta=1.0)
    W_vOVo = t1_erisab[nocca:, :noccb, :nocca, noccb:].transpose(0, 1, 3, 2).copy()
    einsum_('lkdc,ilad->akci', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2aa, out=W_vOVo, alpha=0.5, beta=1.0)
    einsum_('lkdc,iald->akci', t1_erisbb[:noccb, :noccb, noccb:, noccb:], t2ab, out=W_vOVo, alpha=0.5, beta=1.0)
    W_VovO = t1_erisab[:nocca, noccb:, nocca:, :noccb].transpose(1, 0, 2, 3).copy()
    einsum_('klcd,ljdb->bkcj', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2bb, out=W_VovO, alpha=0.5, beta=1.0)
    einsum_('lkdc,ldjb->bkcj', t1_erisaa[:nocca, :nocca, nocca:, nocca:], t2ab, out=W_VovO, alpha=0.5, beta=1.0)

    mycc.tf_vv = tf_vv
    mycc.tf_oo = tf_oo
    mycc.W_oooo = W_oooo
    mycc.W_ovvo = W_ovvo
    mycc.W_OvVo = W_OvVo
    mycc.tf_VV = tf_VV
    mycc.tf_OO = tf_OO
    mycc.W_OOOO = W_OOOO
    mycc.W_OVVO = W_OVVO
    mycc.W_oVvO = W_oVvO
    mycc.W_oOoO = W_oOoO
    mycc.W_vOvO = W_vOvO
    mycc.W_VoVo = W_VoVo
    mycc.W_vovo = W_vovo
    mycc.W_VOVO = W_VOVO
    mycc.W_vOVo = W_vOVo
    mycc.W_VovO = W_VovO
    return mycc

def compute_r1r2_uhf(mycc, t2):
    '''Compute r1 and r2, without the contributions from t3. r2 still needs to be symmetrized'''
    nocca, noccb = mycc.nocca, mycc.noccb
    t1_focka, t1_fockb = mycc.t1_focka, mycc.t1_fockb
    t1_erisaa, t1_erisab, t1_erisbb = mycc.t1_erisaa, mycc.t1_erisab, mycc.t1_erisbb
    t2aa, t2ab, t2bb = t2

    r1a = t1_focka[nocca:, :nocca].T.copy()
    einsum_('kc,ikac->ia', t1_focka[:nocca, nocca:], t2aa, out=r1a, alpha=1.0, beta=1.0)
    einsum_('kc,iakc->ia', t1_fockb[:noccb, noccb:], t2ab, out=r1a, alpha=1.0, beta=1.0)
    einsum_('akcd,ikcd->ia', t1_erisaa[nocca:, :nocca, nocca:, nocca:], t2aa, out=r1a, alpha=0.5, beta=1.0)
    einsum_('akcd,ickd->ia', t1_erisab[nocca:, :noccb, nocca:, noccb:], t2ab, out=r1a, alpha=1.0, beta=1.0)
    einsum_('klic,klac->ia', t1_erisaa[:nocca, :nocca, :nocca, nocca:], t2aa, out=r1a, alpha=-0.5, beta=1.0)
    einsum_('klic,kalc->ia', t1_erisab[:nocca, :noccb, :nocca, noccb:], t2ab, out=r1a, alpha=-1.0, beta=1.0)

    r1b = t1_fockb[noccb:, :noccb].T.copy()
    einsum_('kc,ikac->ia', t1_fockb[:noccb, noccb:], t2bb, out=r1b, alpha=1.0, beta=1.0)
    einsum_('kc,kcia->ia', t1_focka[:nocca, nocca:], t2ab, out=r1b, alpha=1.0, beta=1.0)
    einsum_('akcd,ikcd->ia', t1_erisbb[noccb:, :noccb, noccb:, noccb:], t2bb, out=r1b, alpha=0.5, beta=1.0)
    einsum_('kadc,kdic->ia', t1_erisab[:nocca, noccb:, nocca:, noccb:], t2ab, out=r1b, alpha=1.0, beta=1.0)
    einsum_('klic,klac->ia', t1_erisbb[:noccb, :noccb, :noccb, noccb:], t2bb, out=r1b, alpha=-0.5, beta=1.0)
    einsum_('lkci,lcka->ia', t1_erisab[:nocca, :noccb, nocca:, :noccb], t2ab, out=r1b, alpha=-1.0, beta=1.0)

    r2aa = 0.25 * t1_erisaa[nocca:, nocca:, :nocca, :nocca].T
    einsum_("bc,ijac->ijab", mycc.tf_vv, t2aa, out=r2aa, alpha=0.5, beta=1.0)
    einsum_("kj,ikab->ijab", mycc.tf_oo, t2aa, out=r2aa, alpha=-0.5, beta=1.0)
    einsum_("abcd,ijcd->ijab", t1_erisaa[nocca:, nocca:, nocca:, nocca:], t2aa, out=r2aa, alpha=0.125, beta=1.0)
    einsum_("klij,klab->ijab", mycc.W_oooo, t2aa, out=r2aa, alpha=0.125, beta=1.0)
    einsum_("kbcj,ikac->ijab", mycc.W_ovvo, t2aa, out=r2aa, alpha=1.0, beta=1.0)
    einsum_("kbcj,iakc->ijab", mycc.W_OvVo, t2ab, out=r2aa, alpha=1.0, beta=1.0)

    r2ab = t1_erisab[nocca:, noccb:, :nocca, :noccb].transpose(2, 3, 0, 1).copy()
    # FIXME
    r2ab = r2ab.transpose(0, 2, 1, 3)
    einsum_("bc,iajc->iajb", mycc.tf_VV, t2ab, out=r2ab, alpha=1.0, beta=1.0)
    einsum_("ac,icjb->iajb", mycc.tf_vv, t2ab, out=r2ab, alpha=1.0, beta=1.0)
    einsum_("kj,iakb->iajb", mycc.tf_OO, t2ab, out=r2ab, alpha=-1.0, beta=1.0)
    einsum_("ki,kajb->iajb", mycc.tf_oo, t2ab, out=r2ab, alpha=-1.0, beta=1.0)
    einsum_("abcd,icjd->iajb", t1_erisab[nocca:, noccb:, nocca:, noccb:], t2ab, out=r2ab, alpha=1.0, beta=1.0)
    einsum_("klij,kalb->iajb", mycc.W_oOoO, t2ab, out=r2ab, alpha=1.0, beta=1.0)
    einsum_("akcj,ickb->iajb", mycc.W_vOvO, t2ab, out=r2ab, alpha=1.0, beta=1.0)
    einsum_("akci,kcjb->iajb", mycc.W_vovo, t2ab, out=r2ab, alpha=1.0, beta=1.0)
    einsum_("akci,kjcb->iajb", mycc.W_vOVo, t2bb, out=r2ab, alpha=1.0, beta=1.0)
    einsum_("bkcj,ikac->iajb", mycc.W_VovO, t2aa, out=r2ab, alpha=1.0, beta=1.0)
    einsum_("bkcj,iakc->iajb", mycc.W_VOVO, t2ab, out=r2ab, alpha=1.0, beta=1.0)
    einsum_("bkci,kajc->iajb", mycc.W_VoVo, t2ab, out=r2ab, alpha=1.0, beta=1.0)

    r2bb = 0.25 * t1_erisbb[noccb:, noccb:, :noccb, :noccb].T
    einsum_("bc,ijac->ijab", mycc.tf_VV, t2bb, out=r2bb, alpha=0.5, beta=1.0)
    einsum_("kj,ikab->ijab", mycc.tf_OO, t2bb, out=r2bb, alpha=-0.5, beta=1.0)
    einsum_("abcd,ijcd->ijab", t1_erisbb[noccb:, noccb:, noccb:, noccb:], t2bb, out=r2bb, alpha=0.125, beta=1.0)
    einsum_("klij,klab->ijab", mycc.W_OOOO, t2bb, out=r2bb, alpha=0.125, beta=1.0)
    einsum_("kbcj,ikac->ijab", mycc.W_OVVO, t2bb, out=r2bb, alpha=1.0, beta=1.0)
    einsum_("kbcj,kcia->ijab", mycc.W_oVvO, t2ab, out=r2bb, alpha=1.0, beta=1.0)

    return (r1a, r1b), (r2aa, r2ab, r2bb)

def r1r2_add_t3_tril_uhf(mycc, t3, r1, r2):
    '''add the contributions from t3 amplitudes to r1r2'''
    nocca, nvira = mycc.nocca, mycc.nvira
    noccb, nvirb = mycc.noccb, mycc.nvirb
    blksize_o_aaa, blksize_v_aaa = mycc.blksize_o_aaa, mycc.blksize_v_aaa
    blksize_o_aab, blksize_v_aab = mycc.blksize_o_aab, mycc.blksize_v_aab
    t1_focka, t1_fockb = mycc.t1_focka, mycc.t1_fockb
    t1_erisaa, t1_erisab, t1_erisbb = mycc.t1_erisaa, mycc.t1_erisab, mycc.t1_erisbb

    t3aaa, t3aab, t3bba, t3bbb = t3
    (r1a, r1b), (r2aa, r2ab, r2bb) = r1, r2

    t3_tmp = np.empty((blksize_o_aaa,) * 3 + (blksize_v_aaa,) * 3, dtype=t3aaa.dtype)
    for i0, i1 in lib.prange(0, nocca, blksize_o_aaa):
        bi = i1 - i0
        for m0, m1 in lib.prange(0, nocca, blksize_o_aaa):
            bm = m1 - m0
            for n0, n1 in lib.prange(0, nocca, blksize_o_aaa):
                bn = n1 - n0
                for a0, a1 in lib.prange(0, nvira, blksize_v_aaa):
                    ba = a1 - a0
                    for e0, e1 in lib.prange(0, nvira, blksize_v_aaa):
                        be = e1 - e0
                        for f0, f1 in lib.prange(0, nvira, blksize_v_aaa):
                            bf = f1 - f0
                            _unp_aaa(mycc, t3aaa, t3_tmp, i0, i1, m0, m1, n0, n1, a0, a1, e0, e1, f0, f1)
                            einsum_('mnef,imnaef->ia',
                                t1_erisaa[m0:m1, n0:n1, nocca + e0:nocca + e1, nocca + f0:nocca + f1],
                                t3_tmp[:bi, :bm, :bn, :ba, :be, :bf], out=r1a[i0:i1, a0:a1], alpha=0.25, beta=1.0)

                            einsum_("nf,imnaef->imae", t1_focka[n0:n1, nocca + f0:nocca + f1],
                                t3_tmp[:bi, :bm, :bn, :ba, :be, :bf],
                                out=r2aa[i0:i1, m0:m1, a0:a1, e0:e1], alpha=0.25, beta=1.0)
                            einsum_("bnef,imnaef->imab",
                                t1_erisaa[nocca:, n0:n1, nocca + e0:nocca + e1, nocca + f0:nocca + f1],
                                t3_tmp[:bi, :bm, :bn, :ba, :be, :bf],
                                out=r2aa[i0:i1, m0:m1, a0:a1, :], alpha=0.25, beta=1.0)
                            einsum_("mnjf,imnaef->ijae", t1_erisaa[m0:m1, n0:n1, :nocca, nocca + f0:nocca + f1],
                                t3_tmp[:bi, :bm, :bn, :ba, :be, :bf],
                                out=r2aa[i0:i1, :, a0:a1, e0:e1], alpha=-0.25, beta=1.0)
    t3_tmp = None

    t3_tmp = np.empty((blksize_o_aab,) * 2 + (blksize_v_aab,) * 2 + (noccb,) + (nvirb,), dtype=t3aab.dtype)
    for i0, i1 in lib.prange(0, nocca, blksize_o_aab):
        bi = i1 - i0
        for n0, n1 in lib.prange(0, nocca, blksize_o_aab):
            bn = n1 - n0
            for a0, a1 in lib.prange(0, nvira, blksize_v_aab):
                ba = a1 - a0
                for f0, f1 in lib.prange(0, nvira, blksize_v_aab):
                    bf = f1 - f0
                    _unp_aab(mycc, t3aab, t3_tmp, i0, i1, n0, n1, a0, a1, f0, f1)
                    einsum_('nmfe,inafme->ia', t1_erisab[n0:n1, :noccb, nocca + f0:nocca + f1, noccb:],
                        t3_tmp[:bi, :bn, :ba, :bf], out=r1a[i0:i1, a0:a1], alpha=1.0, beta=1.0)
                    einsum_('inaf,inafme->me', t1_erisaa[i0:i1, n0:n1, nocca + a0:nocca + a1, nocca + f0:nocca + f1],
                        t3_tmp[:bi, :bn, :ba, :bf], out=r1b, alpha=0.25, beta=1.0)

                    einsum_("me,inafme->inaf", t1_fockb[:noccb, noccb:],
                        t3_tmp[:bi, :bn, :ba, :bf], out=r2aa[i0:i1, n0:n1, a0:a1, f0:f1], alpha=0.25, beta=1.0)
                    einsum_("emfb,inafmb->inae", t1_erisab[nocca:, :noccb, nocca + f0:nocca + f1, noccb:],
                        t3_tmp[:bi, :bn, :ba, :bf], out=r2aa[i0:i1, n0:n1, a0:a1, :], alpha=0.5, beta=1.0)
                    einsum_("njme,inafje->imaf", t1_erisab[n0:n1, :noccb, :nocca, noccb:],
                        t3_tmp[:bi, :bn, :ba, :bf], out=r2aa[i0:i1, :, a0:a1, f0:f1], alpha=-0.5, beta=1.0)

                    einsum_("nf,inafjb->iajb", t1_focka[n0:n1, nocca + f0:nocca + f1],
                        t3_tmp[:bi, :bn, :ba, :bf], out=r2ab[i0:i1, a0:a1, :, :], alpha=1.0, beta=1.0)
                    einsum_("nbfe,inafje->iajb", t1_erisab[n0:n1, noccb:, nocca + f0:nocca + f1, noccb:],
                        t3_tmp[:bi, :bn, :ba, :bf], out=r2ab[i0:i1, a0:a1, :, :], alpha=1.0, beta=1.0)
                    einsum_("enaf,inafjb->iejb", t1_erisaa[nocca:, n0:n1, nocca + a0:nocca + a1, nocca + f0:nocca + f1],
                        t3_tmp[:bi, :bn, :ba, :bf], out=r2ab[i0:i1, ...], alpha=0.5, beta=1.0)
                    einsum_("nmfj,inafmb->iajb", t1_erisab[n0:n1, :noccb, nocca + f0:nocca + f1, :noccb],
                        t3_tmp[:bi, :bn, :ba, :bf], out=r2ab[i0:i1, a0:a1, :, :], alpha=-1.0, beta=1.0)
                    einsum_("inmf,inafjb->majb", t1_erisaa[i0:i1, n0:n1, :nocca, nocca + f0:nocca + f1],
                        t3_tmp[:bi, :bn, :ba, :bf, :, :], out=r2ab[:, a0:a1, :, :], alpha=-0.5, beta=1.0)
    t3_tmp = None

    t3_tmp = np.empty((blksize_o_aab,) * 2 + (blksize_v_aab,) * 2 + (nocca,) + (nvira,), dtype=t3bba.dtype)
    for m0, m1 in lib.prange(0, noccb, blksize_o_aab):
        bm = m1 - m0
        for n0, n1 in lib.prange(0, noccb, blksize_o_aab):
            bn = n1 - n0
            for e0, e1 in lib.prange(0, nvirb, blksize_v_aab):
                be = e1 - e0
                for f0, f1 in lib.prange(0, nvirb, blksize_v_aab):
                    bf = f1 - f0
                    _unp_bba(mycc, t3bba, t3_tmp, m0, m1, n0, n1, e0, e1, f0, f1)
                    einsum_('mnef,mnefia->ia', t1_erisbb[m0:m1, n0:n1, noccb + e0:noccb + e1, noccb + f0:noccb + f1],
                        t3_tmp[:bm, :bn, :be, :bf], out=r1a, alpha=0.25, beta=1.0)

                    einsum_('inaf,mnefia->me', t1_erisab[:nocca, n0:n1, nocca:, noccb + f0:noccb + f1],
                        t3_tmp[:bm, :bn, :be, :bf], out=r1b[m0:m1, e0:e1], alpha=1.0, beta=1.0)

                    einsum_("nf,mnefia->iame", t1_fockb[n0:n1, noccb + f0:noccb + f1],
                        t3_tmp[:bm, :bn, :be, :bf], out=r2ab[:, :, m0:m1, e0:e1], alpha=1.0, beta=1.0)
                    einsum_("bnef,mnefia->iamb",
                        t1_erisbb[noccb:, n0:n1, noccb + e0:noccb + e1, noccb + f0:noccb + f1],
                        t3_tmp[:bm, :bn, :be, :bf], out=r2ab[:, :, m0:m1, :], alpha=0.5, beta=1.0)
                    einsum_("anbf,mnefib->iame", t1_erisab[nocca:, n0:n1, nocca:, noccb + f0:noccb + f1],
                        t3_tmp[:bm, :bn, :be, :bf], out=r2ab[:, :, m0:m1, e0:e1], alpha=1.0, beta=1.0)
                    einsum_("mnjf,mnefia->iaje", t1_erisbb[m0:m1, n0:n1, :noccb, noccb + f0:noccb + f1],
                        t3_tmp[:bm, :bn, :be, :bf], out=r2ab[:, :, :, e0:e1], alpha=-0.5, beta=1.0)
                    einsum_("jnif,mnefja->iame", t1_erisab[:nocca, n0:n1, :nocca, noccb + f0:noccb + f1],
                        t3_tmp[:bm, :bn, :be, :bf], out=r2ab[:, :, m0:m1, e0:e1], alpha=-1.0, beta=1.0)

                    einsum_("ia,mnefia->mnef", t1_focka[:nocca, nocca:],
                        t3_tmp[:bm, :bn, :be, :bf], out=r2bb[m0:m1, n0:n1, e0:e1, f0:f1], alpha=0.25, beta=1.0)
                    einsum_("iabf,mnefib->mnea", t1_erisab[:nocca, noccb:, nocca:, noccb + f0:noccb + f1],
                        t3_tmp[:bm, :bn, :be, :bf], out=r2bb[m0:m1, n0:n1, e0:e1, :], alpha=0.5, beta=1.0)
                    einsum_("jnai,mnefja->mief", t1_erisab[:nocca, n0:n1, nocca:, :noccb],
                        t3_tmp[:bm, :bn, :be, :bf], out=r2bb[m0:m1, :, e0:e1, f0:f1], alpha=-0.5, beta=1.0)
    t3_tmp = None

    t3_tmp = np.empty((blksize_o_aaa,) * 3 + (blksize_v_aaa,) * 3, dtype=t3bbb.dtype)
    for i0, i1 in lib.prange(0, noccb, blksize_o_aaa):
        bi = i1 - i0
        for m0, m1 in lib.prange(0, noccb, blksize_o_aaa):
            bm = m1 - m0
            for n0, n1 in lib.prange(0, noccb, blksize_o_aaa):
                bn = n1 - n0
                for a0, a1 in lib.prange(0, nvirb, blksize_v_aaa):
                    ba = a1 - a0
                    for e0, e1 in lib.prange(0, nvirb, blksize_v_aaa):
                        be = e1 - e0
                        for f0, f1 in lib.prange(0, nvirb, blksize_v_aaa):
                            bf = f1 - f0
                            _unp_bbb(mycc, t3bbb, t3_tmp, i0, i1, m0, m1, n0, n1, a0, a1, e0, e1, f0, f1)
                            einsum_('mnef,imnaef->ia',
                                t1_erisbb[m0:m1, n0:n1, noccb + e0 : noccb + e1, noccb + f0: noccb + f1],
                                t3_tmp[:bi, :bm, :bn, :ba, :be, :bf], out=r1b[i0:i1, a0:a1], alpha=0.25, beta=1.0)

                            einsum_("nf,imnaef->imae", t1_fockb[n0:n1, noccb + f0:noccb + f1],
                                t3_tmp[:bi, :bm, :bn, :ba, :be, :bf],
                                out=r2bb[i0:i1, m0:m1, a0:a1, e0:e1], alpha=0.25, beta=1.0)
                            einsum_("bnef,imnaef->imab",
                                t1_erisbb[noccb:, n0:n1, noccb + e0:noccb + e1, noccb + f0:noccb + f1],
                                t3_tmp[:bi, :bm, :bn, :ba, :be, :bf],
                                out=r2bb[i0:i1, m0:m1, a0:a1, :], alpha=0.25, beta=1.0)
                            einsum_("mnjf,imnaef->ijae", t1_erisbb[m0:m1, n0:n1, :noccb, noccb + f0:noccb + f1],
                                t3_tmp[:bi, :bm, :bn, :ba, :be, :bf],
                                out=r2bb[i0:i1, :, a0:a1, e0:e1], alpha=-0.25, beta=1.0)
    t3_tmp = None

    return (r1a, r1b), (r2aa, r2ab, r2bb)

def antisymmetrize_r2_uhf(r2):
    r2aa, r2ab, r2bb = r2
    r2aa -= r2aa.transpose(1, 0, 2, 3)
    r2aa -= r2aa.transpose(0, 1, 3, 2)
    r2bb -= r2bb.transpose(1, 0, 2, 3)
    r2bb -= r2bb.transpose(0, 1, 3, 2)
    return (r2aa, r2ab, r2bb)

def r1r2_divide_e_uhf(r1, r2, mo_energy, level_shift):
    r1a, r1b = r1
    r2aa, r2ab, r2bb = r2
    nocca, noccb = r1a.shape[0], r1b.shape[0]

    eia_a = mo_energy[0][:nocca, None] - mo_energy[0][None, nocca:] - level_shift
    r1a /= eia_a
    eia_b = mo_energy[1][:noccb, None] - mo_energy[1][None, noccb:] - level_shift
    r1b /= eia_b

    eijab_aa = eia_a[:, None, :, None] + eia_a[None, :, None, :]
    r2aa /= eijab_aa
    # eijab_ab = eia_a[:, None, :, None] + eia_b[None, :, None, :]
    eijab_ab = eia_a[:, :, None, None] + eia_b[None, None, :, :]
    r2ab /= eijab_ab
    eijab_bb = eia_b[:, None, :, None] + eia_b[None, :, None, :]
    r2bb /= eijab_bb

    eia_a, eia_b, eijab_aa, eijab_ab, eijab_bb = None, None, None, None, None
    return (r1a, r1b), (r2aa, r2ab, r2bb)

def update_amps_t1t2_with_t3_tril_uhf(mycc, tamps):
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    t1, t2, t3 = tamps
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2

    update_t1_fock_eris_uhf(mycc, t1)
    time1 = log.timer_debug1('t1t2: update fock and eris', *time0)

    intermediates_t1t2_uhf(mycc, t2)
    time1 = log.timer_debug1('t1t2: update intermediates', *time0)

    r1, r2 = compute_r1r2_uhf(mycc, t2)
    r1, r2 = r1r2_add_t3_tril_uhf(mycc, t3, r1, r2)
    # antisymmetrize R2
    r2 = antisymmetrize_r2_uhf(r2)
    # divide by eijkabc
    r1, r2 = r1r2_divide_e_uhf(r1, r2, mycc.mo_energy, mycc.level_shift)
    (r1a, r1b), (r2aa, r2ab, r2bb) = r1, r2

    mycc.r_norm[0] = np.sqrt(np.linalg.norm(r1a) ** 2 + np.linalg.norm(r1b) ** 2)
    mycc.r_norm[1] = np.sqrt(np.linalg.norm(r2aa) ** 2 + np.linalg.norm(r2ab) ** 2 + np.linalg.norm(r2bb) ** 2)

    t1a += r1a
    t1b += r1b
    t2aa += r2aa
    t2ab += r2ab
    t2bb += r2bb
    time1 = log.timer_debug1('t1t2: update t1 & t2', *time1)
    time0 = log.timer_debug1('t1t2 total', *time0)
    return (t1a, t1b), (t2aa, t2ab, t2bb)

def intermediates_t3_uhf(mycc, t2):
    '''intermediates for t3 residual equation, without contribution from t3'''
    nocca, noccb = mycc.nocca, mycc.noccb
    t1_focka, t1_fockb = mycc.t1_focka, mycc.t1_fockb
    t1_erisaa, t1_erisab, t1_erisbb = mycc.t1_erisaa, mycc.t1_erisab, mycc.t1_erisbb
    t2aa, t2ab, t2bb = t2
    # aaa
    W_vvvv = t1_erisaa[nocca:, nocca:, nocca:, nocca:].copy()
    einsum_('lmde,lmab->abde', t1_erisaa[:nocca, :nocca, nocca:, nocca:], t2aa, out=W_vvvv, alpha=0.5, beta=1.0)
    W_voov = t1_erisaa[nocca:, :nocca, :nocca, nocca:].copy()
    einsum_('mled,imae->alid', t1_erisaa[:nocca, :nocca, nocca:, nocca:], t2aa, out=W_voov, alpha=1.0, beta=1.0)
    einsum_('lmde,iame->alid', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=W_voov, alpha=1.0, beta=1.0)
    W_vOoV = t1_erisab[nocca:, :noccb, :nocca, noccb:].copy()
    einsum_('mled,imae->alid', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2aa, out=W_vOoV, alpha=1.0, beta=1.0)
    einsum_('mled,iame->alid', t1_erisbb[:noccb, :noccb, noccb:, noccb:], t2ab, out=W_vOoV, alpha=1.0, beta=1.0)
    W_vvvo = t1_erisaa[nocca:, nocca:, nocca:, :nocca].copy()
    einsum_('lbed,klce->bcdk', t1_erisaa[:nocca, nocca:, nocca:, nocca:], t2aa, out=W_vvvo, alpha=2.0, beta=1.0)
    einsum_('blde,kcle->bcdk', t1_erisab[nocca:, :noccb, nocca:, noccb:], t2ab, out=W_vvvo, alpha=2.0, beta=1.0)
    einsum_('lmdk,lmbc->bcdk', t1_erisaa[:nocca, :nocca, nocca:, :nocca], t2aa, out=W_vvvo, alpha=0.5, beta=1.0)
    W_ovoo = t1_erisaa[:nocca, nocca:, :nocca, :nocca].copy()
    einsum_('ld,jkdc->lcjk', t1_focka[:nocca, nocca:], t2aa, out=W_ovoo, alpha=1.0, beta=1.0)
    einsum_('mldj,kmcd->lcjk', t1_erisaa[:nocca, :nocca, nocca:, :nocca], t2aa, out=W_ovoo, alpha=2.0, beta=1.0)
    einsum_('lmjd,kcmd->lcjk', t1_erisab[:nocca, :noccb, :nocca, noccb:], t2ab, out=W_ovoo, alpha=2.0, beta=1.0)
    einsum_('lcde,jkde->lcjk', t1_erisaa[:nocca, nocca:, nocca:, nocca:], t2aa, out=W_ovoo, alpha=0.5, beta=1.0)
    # bbb
    W_VVVV = t1_erisbb[noccb:, noccb:, noccb:, noccb:].copy()
    einsum_('lmde,lmab->abde', t1_erisbb[:noccb, :noccb, noccb:, noccb:], t2bb, out=W_VVVV, alpha=0.5, beta=1.0)
    W_VOOV = t1_erisbb[noccb:, :noccb, :noccb, noccb:].copy()
    einsum_('mled,imae->alid', t1_erisbb[:noccb, :noccb, noccb:, noccb:], t2bb, out=W_VOOV, alpha=1.0, beta=1.0)
    einsum_('mled,meia->alid', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=W_VOOV, alpha=1.0, beta=1.0)
    W_VoOv = t1_erisab[:nocca, noccb:, nocca:, :noccb].transpose(1, 0, 3, 2).copy()
    einsum_('lmde,imae->alid', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2bb, out=W_VoOv, alpha=1.0, beta=1.0)
    einsum_('mled,meia->alid', t1_erisaa[:nocca, :nocca, nocca:, nocca:], t2ab, out=W_VoOv, alpha=1.0, beta=1.0)
    W_VVVO = t1_erisbb[noccb:, noccb:, noccb:, :noccb].copy()
    einsum_('lbed,klce->bcdk', t1_erisbb[:noccb, noccb:, noccb:, noccb:], t2bb, out=W_VVVO, alpha=2.0, beta=1.0)
    einsum_('lbed,lekc->bcdk', t1_erisab[:nocca, noccb:, nocca:, noccb:], t2ab, out=W_VVVO, alpha=2.0, beta=1.0)
    einsum_('lmdk,lmbc->bcdk', t1_erisbb[:noccb, :noccb, noccb:, :noccb], t2bb, out=W_VVVO, alpha=0.5, beta=1.0)
    W_OVOO = t1_erisbb[:noccb, noccb:, :noccb, :noccb].copy()
    einsum_('ld,jkdc->lcjk', t1_fockb[:noccb, noccb:], t2bb, out=W_OVOO, alpha=1.0, beta=1.0)
    einsum_('mldj,kmcd->lcjk', t1_erisbb[:noccb, :noccb, noccb:, :noccb], t2bb, out=W_OVOO, alpha=2.0, beta=1.0)
    einsum_('mldj,mdkc->lcjk', t1_erisab[:nocca, :noccb, nocca:, :noccb], t2ab, out=W_OVOO, alpha=2.0, beta=1.0)
    einsum_('lcde,jkde->lcjk', t1_erisbb[:noccb, noccb:, noccb:, noccb:], t2bb, out=W_OVOO, alpha=0.5, beta=1.0)
    # aab & bba
    W_vVvV = t1_erisab[nocca:, noccb:, nocca:, noccb:].copy()
    einsum_('lmed,lbmc->bced', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=W_vVvV, alpha=1.0, beta=1.0)
    W_oVoV = t1_erisab[:nocca, noccb:, :nocca, noccb:].copy()
    einsum_('lmed,iemc->lcid', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=W_oVoV, alpha=-1.0, beta=1.0)
    W_vOvO_tc = t1_erisab[nocca:, :noccb, nocca:, :noccb].copy()
    einsum_('mlde,make->aldk', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=W_vOvO_tc, alpha=-1.0, beta=1.0)
    W_vVvO = t1_erisab[nocca:, noccb:, nocca:, :noccb].copy()
    einsum_('lbed,lekc->bcdk', t1_erisaa[:nocca, nocca:, nocca:, nocca:], t2ab, out=W_vVvO, alpha=1.0, beta=1.0)
    einsum_('blde,lkec->bcdk', t1_erisab[nocca:, :noccb, nocca:, noccb:], t2bb, out=W_vVvO, alpha=1.0, beta=1.0)
    einsum_('lcde,lbke->bcdk', t1_erisab[:nocca, noccb:, nocca:, noccb:], t2ab, out=W_vVvO, alpha=-1.0, beta=1.0)
    einsum_('lmdk,lbmc->bcdk', t1_erisab[:nocca, :noccb, nocca:, :noccb], t2ab, out=W_vVvO, alpha=1.0, beta=1.0)
    W_oVoO = t1_erisab[:nocca, noccb:, :nocca, :noccb].copy()
    einsum_('ld,jdkc->lcjk', t1_focka[:nocca, nocca:], t2ab, out=W_oVoO, alpha=1.0, beta=1.0)
    einsum_('mldj,mdkc->lcjk', t1_erisaa[:nocca, :nocca, nocca:, :nocca], t2ab, out=W_oVoO, alpha=1.0, beta=1.0)
    einsum_('lmjd,mkdc->lcjk', t1_erisab[:nocca, :noccb, :nocca, noccb:], t2bb, out=W_oVoO, alpha=1.0, beta=1.0)
    einsum_('lmdk,jdmc->lcjk', t1_erisab[:nocca, :noccb, nocca:, :noccb], t2ab, out=W_oVoO, alpha=-1.0, beta=1.0)
    einsum_('lcde,jdke->lcjk', t1_erisab[:nocca, noccb:, nocca:, noccb:], t2ab, out=W_oVoO, alpha=1.0, beta=1.0)
    W_vVoV = t1_erisab[nocca:, noccb:, :nocca, noccb:].copy()
    einsum_('bled,jelc->bcjd', t1_erisab[nocca:, :noccb, nocca:, noccb:], t2ab, out=W_vVoV, alpha=-1.0, beta=1.0)
    einsum_('lced,jlbe->bcjd', t1_erisab[:nocca, noccb:, nocca:, noccb:], t2aa, out=W_vVoV, alpha=1.0, beta=1.0)
    einsum_('lced,jble->bcjd', t1_erisbb[:noccb, noccb:, noccb:, noccb:], t2ab, out=W_vVoV, alpha=1.0, beta=1.0)
    einsum_('mljd,mblc->bcjd', t1_erisab[:nocca, :noccb, :nocca, noccb:], t2ab, out=W_vVoV, alpha=1.0, beta=1.0)
    W_vOoO = t1_erisab[nocca:, :noccb, :nocca, :noccb].copy()
    einsum_('ld,jakd->aljk', t1_fockb[:noccb, noccb:], t2ab, out=W_vOoO, alpha=1.0, beta=1.0)
    einsum_('mljd,makd->aljk', t1_erisab[:nocca, :noccb, :nocca, noccb:], t2ab, out=W_vOoO, alpha=-1.0, beta=1.0)
    einsum_('mldk,jmad->aljk', t1_erisab[:nocca, :noccb, nocca:, :noccb], t2aa, out=W_vOoO, alpha=1.0, beta=1.0)
    einsum_('mldk,jamd->aljk', t1_erisbb[:noccb, :noccb, noccb:, :noccb], t2ab, out=W_vOoO, alpha=1.0, beta=1.0)
    einsum_('alde,jdke->aljk', t1_erisab[nocca:, :noccb, nocca:, noccb:], t2ab, out=W_vOoO, alpha=1.0, beta=1.0)

    mycc.W_vvvv = W_vvvv
    mycc.W_voov = W_voov
    mycc.W_vOoV = W_vOoV
    mycc.W_vvvo = W_vvvo
    mycc.W_ovoo = W_ovoo
    mycc.W_VVVV = W_VVVV
    mycc.W_VOOV = W_VOOV
    mycc.W_VoOv = W_VoOv
    mycc.W_VVVO = W_VVVO
    mycc.W_OVOO = W_OVOO
    mycc.W_vVvV = W_vVvV
    mycc.W_oVoV = W_oVoV
    mycc.W_vOvO_tc = W_vOvO_tc
    mycc.W_vVvO = W_vVvO
    mycc.W_oVoO = W_oVoO
    mycc.W_vVoV = W_vVoV
    mycc.W_vOoO = W_vOoO
    return mycc

def intermediates_t3_add_t3_tril_uhf(mycc, t3):
    '''Add the contributions of t3 to t3 intermediates'''
    nocca, nvira = mycc.nocca, mycc.nvira
    noccb, nvirb = mycc.noccb, mycc.nvirb
    blksize_o_aaa, blksize_v_aaa = mycc.blksize_o_aaa, mycc.blksize_v_aaa
    blksize_o_aab, blksize_v_aab = mycc.blksize_o_aab, mycc.blksize_v_aab
    t1_erisaa, t1_erisab, t1_erisbb = mycc.t1_erisaa, mycc.t1_erisab, mycc.t1_erisbb
    t3aaa, t3aab, t3bba, t3bbb = t3

    t3_tmp = np.empty((blksize_o_aaa,) * 3 + (blksize_v_aaa,) * 3, dtype=t3aaa.dtype)
    for l0, l1 in lib.prange(0, nocca, blksize_o_aaa):
        bl = l1 - l0
        for m0, m1 in lib.prange(0, nocca, blksize_o_aaa):
            bm = m1 - m0
            for k0, k1 in lib.prange(0, nocca, blksize_o_aaa):
                bk = k1 - k0
                for b0, b1 in lib.prange(0, nvira, blksize_v_aaa):
                    bb = b1 - b0
                    for e0, e1 in lib.prange(0, nvira, blksize_v_aaa):
                        be = e1 - e0
                        for c0, c1 in lib.prange(0, nvira, blksize_v_aaa):
                            bc = c1 - c0
                            _unp_aaa(mycc, t3aaa, t3_tmp, l0, l1, m0, m1, k0, k1, b0, b1, e0, e1, c0, c1)
                            einsum_('lmde,lmkbec->bcdk', t1_erisaa[l0:l1, m0:m1, nocca:, nocca + e0:nocca + e1],
                                    t3_tmp[:bl, :bm, :bk, :bb, :be, :bc],
                                    out=mycc.W_vvvo[b0:b1, c0:c1, :, k0:k1], alpha=-0.5, beta=1.0)
                            einsum_('jmbe,lmkbec->jclk',
                                    t1_erisaa[:nocca, m0:m1, nocca + b0:nocca + b1, nocca + e0:nocca + e1],
                                    t3_tmp[:bl, :bm, :bk, :bb, :be, :bc],
                                    out=mycc.W_ovoo[:, c0:c1, l0:l1, k0:k1], alpha=0.5, beta=1.0)
    t3_tmp = None

    t3_tmp = np.empty((blksize_o_aaa,) * 3 + (blksize_v_aaa,) * 3, dtype=t3bbb.dtype)
    for l0, l1 in lib.prange(0, noccb, blksize_o_aaa):
        bl = l1 - l0
        for m0, m1 in lib.prange(0, noccb, blksize_o_aaa):
            bm = m1 - m0
            for k0, k1 in lib.prange(0, noccb, blksize_o_aaa):
                bk = k1 - k0
                for b0, b1 in lib.prange(0, nvirb, blksize_v_aaa):
                    bb = b1 - b0
                    for e0, e1 in lib.prange(0, nvirb, blksize_v_aaa):
                        be = e1 - e0
                        for c0, c1 in lib.prange(0, nvirb, blksize_v_aaa):
                            bc = c1 - c0
                            _unp_bbb(mycc, t3bbb, t3_tmp, l0, l1, m0, m1, k0, k1, b0, b1, e0, e1, c0, c1)
                            einsum_('lmde,lmkbec->bcdk', t1_erisbb[l0:l1, m0:m1, noccb:, noccb + e0:noccb + e1],
                                    t3_tmp[:bl, :bm, :bk, :bb, :be, :bc],
                                    out=mycc.W_VVVO[b0:b1, c0:c1, :, k0:k1], alpha=-0.5, beta=1.0)
                            einsum_('jmbe,lmkbec->jclk',
                                    t1_erisbb[:noccb, m0:m1, noccb + b0:noccb + b1, noccb + e0:noccb + e1],
                                    t3_tmp[:bl, :bm, :bk, :bb, :be, :bc],
                                    out=mycc.W_OVOO[:, c0:c1, l0:l1, k0:k1], alpha=0.5, beta=1.0)
    t3_tmp = None

    t3_tmp = np.empty((blksize_o_aab,) * 2 + (blksize_v_aab,) * 2 + (noccb,) + (nvirb,), dtype=t3aab.dtype)
    for l0, l1 in lib.prange(0, nocca, blksize_o_aab):
        bl = l1 - l0
        for k0, k1 in lib.prange(0, nocca, blksize_o_aab):
            bk = k1 - k0
            for b0, b1 in lib.prange(0, nvira, blksize_v_aab):
                bb = b1 - b0
                for c0, c1 in lib.prange(0, nvira, blksize_v_aab):
                    bc = c1 - c0
                    _unp_aab(mycc, t3aab, t3_tmp, l0, l1, k0, k1, b0, b1, c0, c1)
                    einsum_('lmde,lkbcme->bcdk', t1_erisab[l0:l1, :noccb, nocca:, noccb:],
                        t3_tmp[:bl, :bk, :bb, :bc], out=mycc.W_vvvo[b0:b1, c0:c1, :, k0:k1], alpha=-1.0, beta=1.0)
                    einsum_('jmbe,lkbcme->jclk', t1_erisab[:nocca, :noccb, nocca + b0 : nocca + b1, noccb:],
                        t3_tmp[:bl, :bk, :bb, :bc], out=mycc.W_ovoo[:, c0:c1, l0:l1, k0:k1], alpha=1.0, beta=1.0)
                    einsum_('lkdc,lkbcme->bedm', t1_erisaa[l0:l1, k0:k1, nocca:, nocca + c0: nocca + c1],
                        t3_tmp[:bl, :bk, :bb, :bc], out=mycc.W_vVvO[b0:b1, :, :, :], alpha=-0.5, beta=1.0)
                    einsum_('jkbc,lkbcme->jelm',
                        t1_erisaa[:nocca, k0:k1, nocca + b0:nocca + b1, nocca + c0:nocca + c1],
                        t3_tmp[:bl, :bk, :bb, :bc], out=mycc.W_oVoO[:, :, l0:l1, :], alpha=0.5, beta=1.0)
                    einsum_('kmcd,lkbcme->beld', t1_erisab[k0:k1, :noccb, nocca + c0: nocca + c1, noccb:],
                        t3_tmp[:bl, :bk, :bb, :bc], out=mycc.W_vVoV[b0:b1, :, l0:l1, :], alpha=-1.0, beta=1.0)
                    einsum_('kjce,lkbcme->bjlm', t1_erisab[k0:k1, :noccb, nocca + c0:nocca + c1, noccb:],
                        t3_tmp[:bl, :bk, :bb, :bc], out=mycc.W_vOoO[b0:b1, :, l0:l1, :], alpha=1.0, beta=1.0)
    t3_tmp = None

    t3_tmp = np.empty((blksize_o_aab,) * 2 + (blksize_v_aab,) * 2 + (nocca,) + (nvira,), dtype=t3bba.dtype)
    for l0, l1 in lib.prange(0, noccb, blksize_o_aab):
        bl = l1 - l0
        for k0, k1 in lib.prange(0, noccb, blksize_o_aab):
            bk = k1 - k0
            for b0, b1 in lib.prange(0, nvirb, blksize_v_aab):
                bb = b1 - b0
                for c0, c1 in lib.prange(0, nvirb, blksize_v_aab):
                    bc = c1 - c0
                    _unp_bba(mycc, t3bba, t3_tmp, l0, l1, k0, k1, b0, b1, c0, c1)
                    einsum_('mled,lkbcme->bcdk', t1_erisab[:nocca, l0:l1, nocca:, noccb:],
                        t3_tmp[:bl, :bk, :bb, :bc], out=mycc.W_VVVO[b0:b1, c0:c1, :, k0:k1], alpha=-1.0, beta=1.0)
                    einsum_('mjeb,lkbcme->jclk', t1_erisab[:nocca, :noccb, nocca:, noccb + b0:noccb + b1],
                        t3_tmp[:bl, :bk, :bb, :bc], out=mycc.W_OVOO[:, c0:c1, l0:l1, k0:k1], alpha=1.0, beta=1.0)
                    einsum_('mldb,lkbcme->ecdk', t1_erisab[:nocca, l0:l1, nocca:, noccb + b0:noccb + b1],
                        t3_tmp[:bl, :bk, :bb, :bc], out=mycc.W_vVvO[:, c0:c1, :, k0:k1], alpha=-1.0, beta=1.0)
                    einsum_('jleb,lkbcme->jcmk', t1_erisab[:nocca, l0:l1, nocca:, noccb + b0:noccb + b1],
                        t3_tmp[:bl, :bk, :bb, :bc], out=mycc.W_oVoO[:, c0:c1, :, k0:k1], alpha=1.0, beta=1.0)
                    einsum_('kldb,lkbcme->ecmd', t1_erisbb[k0:k1, l0:l1, noccb:, noccb + b0:noccb + b1],
                        t3_tmp[:bl, :bk, :bb, :bc], out=mycc.W_vVoV[:, c0:c1, :, :], alpha=-0.5, beta=1.0)
                    einsum_('jlcb,lkbcme->ejmk',
                        t1_erisbb[:noccb, l0:l1, noccb + c0:noccb + c1, noccb + b0:noccb + b1],
                        t3_tmp[:bl, :bk, :bb, :bc], out=mycc.W_vOoO[:, :, :, k0:k1], alpha=0.5, beta=1.0)
    t3_tmp = None
    t1_erisaa, t1_erisab, t1_erisbb = None, None, None
    mycc.t1_erisaa, mycc.t1_erisab, mycc.t1_erisbb = None, None, None
    return mycc

def compute_r3aaa_tril_uhf(mycc, t2, t3):
    time1 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    nocca, nvira = mycc.nocca, mycc.nvira
    noccb, nvirb = mycc.noccb, mycc.nvirb
    blksize_o_aaa, blksize_v_aaa = mycc.blksize_o_aaa, mycc.blksize_v_aaa
    t2aa, _, _ = t2
    t3aaa, t3aab, _, _ = t3

    r3aaa = np.zeros_like(t3aaa)
    time2 = logger.process_clock(), logger.perf_counter()

    W_vvvo, W_ovoo = mycc.W_vvvo, mycc.W_ovoo
    tf_vv, tf_oo = mycc.tf_vv, mycc.tf_oo
    r3_tmp = np.empty((blksize_o_aaa,) * 3 + (blksize_v_aaa,) * 3, dtype=t3aaa.dtype)
    t3_tmp = np.empty((blksize_o_aaa,) * 3 + (blksize_v_aaa,) * 3, dtype=t3aaa.dtype)
    t3_tmp_2 = np.empty((blksize_o_aaa,) * 2 + (blksize_v_aaa,) * 2 + (noccb,) + (nvirb,), dtype=t3aaa.dtype)
    for k0, k1 in lib.prange(0, nocca, blksize_o_aaa):
        bk = k1 - k0
        for j0, j1 in lib.prange(0, k1 - 1, blksize_o_aaa):
            bj = j1 - j0
            for i0, i1 in lib.prange(0, j1 - 1, blksize_o_aaa):
                bi = i1 - i0
                for c0, c1 in lib.prange(0, nvira, blksize_v_aaa):
                    bc = c1 - c0
                    for b0, b1 in lib.prange(0, c1 - 1, blksize_v_aaa):
                        bb = b1 - b0
                        for a0, a1 in lib.prange(0, b1 - 1, blksize_v_aaa):
                            ba = a1 - a0
                            # R3aaa: P0
                            einsum_("bcdk,ijad->ijkabc", W_vvvo[b0:b1, c0:c1, :, k0:k1], t2aa[i0:i1, j0:j1, a0:a1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=0.0)
                            einsum_("cbdk,ijad->ijkabc", W_vvvo[c0:c1, b0:b1, :, k0:k1], t2aa[i0:i1, j0:j1, a0:a1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("acdk,ijbd->ijkabc", W_vvvo[a0:a1, c0:c1, :, k0:k1], t2aa[i0:i1, j0:j1, b0:b1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("abdk,ijcd->ijkabc", W_vvvo[a0:a1, b0:b1, :, k0:k1], t2aa[i0:i1, j0:j1, c0:c1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("cadk,ijbd->ijkabc", W_vvvo[c0:c1, a0:a1, :, k0:k1], t2aa[i0:i1, j0:j1, b0:b1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("badk,ijcd->ijkabc", W_vvvo[b0:b1, a0:a1, :, k0:k1], t2aa[i0:i1, j0:j1, c0:c1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("bcdj,ikad->ijkabc", W_vvvo[b0:b1, c0:c1, :, j0:j1], t2aa[i0:i1, k0:k1, a0:a1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("cbdj,ikad->ijkabc", W_vvvo[c0:c1, b0:b1, :, j0:j1], t2aa[i0:i1, k0:k1, a0:a1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("acdj,ikbd->ijkabc", W_vvvo[a0:a1, c0:c1, :, j0:j1], t2aa[i0:i1, k0:k1, b0:b1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("abdj,ikcd->ijkabc", W_vvvo[a0:a1, b0:b1, :, j0:j1], t2aa[i0:i1, k0:k1, c0:c1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("cadj,ikbd->ijkabc", W_vvvo[c0:c1, a0:a1, :, j0:j1], t2aa[i0:i1, k0:k1, b0:b1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("badj,ikcd->ijkabc", W_vvvo[b0:b1, a0:a1, :, j0:j1], t2aa[i0:i1, k0:k1, c0:c1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("bcdi,jkad->ijkabc", W_vvvo[b0:b1, c0:c1, :, i0:i1], t2aa[j0:j1, k0:k1, a0:a1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("cbdi,jkad->ijkabc", W_vvvo[c0:c1, b0:b1, :, i0:i1], t2aa[j0:j1, k0:k1, a0:a1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("acdi,jkbd->ijkabc", W_vvvo[a0:a1, c0:c1, :, i0:i1], t2aa[j0:j1, k0:k1, b0:b1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("abdi,jkcd->ijkabc", W_vvvo[a0:a1, b0:b1, :, i0:i1], t2aa[j0:j1, k0:k1, c0:c1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("cadi,jkbd->ijkabc", W_vvvo[c0:c1, a0:a1, :, i0:i1], t2aa[j0:j1, k0:k1, b0:b1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("badi,jkcd->ijkabc", W_vvvo[b0:b1, a0:a1, :, i0:i1], t2aa[j0:j1, k0:k1, c0:c1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            # R3aaa: P1
                            einsum_("lcjk,ilab->ijkabc", W_ovoo[:, c0:c1, j0:j1, k0:k1], t2aa[i0:i1, :, a0:a1, b0:b1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("lbjk,ilac->ijkabc", W_ovoo[:, b0:b1, j0:j1, k0:k1], t2aa[i0:i1, :, a0:a1, c0:c1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("lajk,ilbc->ijkabc", W_ovoo[:, a0:a1, j0:j1, k0:k1], t2aa[i0:i1, :, b0:b1, c0:c1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("lckj,ilab->ijkabc", W_ovoo[:, c0:c1, k0:k1, j0:j1], t2aa[i0:i1, :, a0:a1, b0:b1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("lbkj,ilac->ijkabc", W_ovoo[:, b0:b1, k0:k1, j0:j1], t2aa[i0:i1, :, a0:a1, c0:c1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("lakj,ilbc->ijkabc", W_ovoo[:, a0:a1, k0:k1, j0:j1], t2aa[i0:i1, :, b0:b1, c0:c1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("lcik,jlab->ijkabc", W_ovoo[:, c0:c1, i0:i1, k0:k1], t2aa[j0:j1, :, a0:a1, b0:b1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("lbik,jlac->ijkabc", W_ovoo[:, b0:b1, i0:i1, k0:k1], t2aa[j0:j1, :, a0:a1, c0:c1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("laik,jlbc->ijkabc", W_ovoo[:, a0:a1, i0:i1, k0:k1], t2aa[j0:j1, :, b0:b1, c0:c1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("lcij,klab->ijkabc", W_ovoo[:, c0:c1, i0:i1, j0:j1], t2aa[k0:k1, :, a0:a1, b0:b1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("lbij,klac->ijkabc", W_ovoo[:, b0:b1, i0:i1, j0:j1], t2aa[k0:k1, :, a0:a1, c0:c1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("laij,klbc->ijkabc", W_ovoo[:, a0:a1, i0:i1, j0:j1], t2aa[k0:k1, :, b0:b1, c0:c1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("lcki,jlab->ijkabc", W_ovoo[:, c0:c1, k0:k1, i0:i1], t2aa[j0:j1, :, a0:a1, b0:b1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("lbki,jlac->ijkabc", W_ovoo[:, b0:b1, k0:k1, i0:i1], t2aa[j0:j1, :, a0:a1, c0:c1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("laki,jlbc->ijkabc", W_ovoo[:, a0:a1, k0:k1, i0:i1], t2aa[j0:j1, :, b0:b1, c0:c1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("lcji,klab->ijkabc", W_ovoo[:, c0:c1, j0:j1, i0:i1], t2aa[k0:k1, :, a0:a1, b0:b1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("lbji,klac->ijkabc", W_ovoo[:, b0:b1, j0:j1, i0:i1], t2aa[k0:k1, :, a0:a1, c0:c1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("laji,klbc->ijkabc", W_ovoo[:, a0:a1, j0:j1, i0:i1], t2aa[k0:k1, :, b0:b1, c0:c1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)

                            time2 = log.timer_debug1('t3aaa: (vvvo + ovoo) * t2aa iter: '
                                '[%2d, %2d] [%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d] [%3d, %3d]:' % (
                                    k0, k1, j0, j1, i0, i1, c0, c1, b0, b1, a0, a1), *time2,)

                            # R3aaa: P2, P4
                            for d0, d1 in lib.prange(0, nvira, blksize_v_aaa):
                                bd = d1 - d0
                                _unp_aaa(mycc, t3aaa, t3_tmp, i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, d0, d1)
                                einsum_("cd,ijkabd->ijkabc", tf_vv[c0:c1, d0:d1], t3_tmp[:bi, :bj, :bk, :ba, :bb, :bd],
                                    out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=1.0, beta=1.0)
                                _unp_aaa(mycc, t3aaa, t3_tmp, i0, i1, j0, j1, k0, k1, a0, a1, c0, c1, d0, d1)
                                einsum_("bd,ijkacd->ijkabc", tf_vv[b0:b1, d0:d1], t3_tmp[:bi, :bj, :bk, :ba, :bc, :bd],
                                    out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-1.0, beta=1.0)
                                _unp_aaa(mycc, t3aaa, t3_tmp, i0, i1, j0, j1, k0, k1, b0, b1, c0, c1, d0, d1)
                                einsum_("ad,ijkbcd->ijkabc", tf_vv[a0:a1, d0:d1], t3_tmp[:bi, :bj, :bk, :bb, :bc, :bd],
                                    out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=1.0, beta=1.0)
                                for e0, e1 in lib.prange(0, nvira, blksize_v_aaa):
                                    be = e1 - e0
                                    _unp_aaa(mycc, t3aaa, t3_tmp, i0, i1, j0, j1, k0, k1, d0, d1, e0, e1, c0, c1)
                                    einsum_("abde,ijkdec->ijkabc", mycc.W_vvvv[a0:a1, b0:b1, d0:d1, e0:e1],
                                        t3_tmp[:bi, :bj, :bk, :bd, :be, :bc],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                                    _unp_aaa(mycc, t3aaa, t3_tmp, i0, i1, j0, j1, k0, k1, d0, d1, e0, e1, b0, b1)
                                    einsum_("acde,ijkdeb->ijkabc", mycc.W_vvvv[a0:a1, c0:c1, d0:d1, e0:e1],
                                        t3_tmp[:bi, :bj, :bk, :bd, :be, :bb],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                                    _unp_aaa(mycc, t3aaa, t3_tmp, i0, i1, j0, j1, k0, k1, d0, d1, e0, e1, a0, a1)
                                    einsum_("bcde,ijkdea->ijkabc", mycc.W_vvvv[b0:b1, c0:c1, d0:d1, e0:e1],
                                        t3_tmp[:bi, :bj, :bk, :bd, :be, :ba],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)

                            time2 = log.timer_debug1('t3aaa: (vv + vvvv) * t3aaa iter: '
                                '[%2d, %2d] [%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d] [%3d, %3d]:' % (
                                    k0, k1, j0, j1, i0, i1, c0, c1, b0, b1, a0, a1), *time2)

                            # R3aaa: P3, P5
                            for l0, l1 in lib.prange(0, nocca, blksize_o_aaa):
                                bl = l1 - l0
                                _unp_aaa(mycc, t3aaa, t3_tmp, i0, i1, j0, j1, l0, l1, a0, a1, b0, b1, c0, c1)
                                einsum_("lk,ijlabc->ijkabc", tf_oo[l0:l1, k0:k1], t3_tmp[:bi, :bj, :bl, :ba, :bb, :bc],
                                    out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-1.0, beta=1.0)
                                _unp_aaa(mycc, t3aaa, t3_tmp, i0, i1, k0, k1, l0, l1, a0, a1, b0, b1, c0, c1)
                                einsum_("lj,iklabc->ijkabc", tf_oo[l0:l1, j0:j1], t3_tmp[:bi, :bk, :bl, :ba, :bb, :bc],
                                    out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=1.0, beta=1.0)
                                _unp_aaa(mycc, t3aaa, t3_tmp, j0, j1, k0, k1, l0, l1, a0, a1, b0, b1, c0, c1)
                                einsum_("li,jklabc->ijkabc", tf_oo[l0:l1, i0:i1], t3_tmp[:bj, :bk, :bl, :ba, :bb, :bc],
                                    out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-1.0, beta=1.0)
                                for m0, m1 in lib.prange(0, nocca, blksize_o_aaa):
                                    bm = m1 - m0
                                    _unp_aaa(mycc, t3aaa, t3_tmp, l0, l1, m0, m1, k0, k1, a0, a1, b0, b1, c0, c1)
                                    einsum_("lmij,lmkabc->ijkabc", mycc.W_oooo[l0:l1, m0:m1, i0:i1, j0:j1],
                                        t3_tmp[:bl, :bm, :bk, :ba, :bb, :bc],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                                    _unp_aaa(mycc, t3aaa, t3_tmp, l0, l1, m0, m1, j0, j1, a0, a1, b0, b1, c0, c1)
                                    einsum_("lmik,lmjabc->ijkabc", mycc.W_oooo[l0:l1, m0:m1, i0:i1, k0:k1],
                                        t3_tmp[:bl, :bm, :bj, :ba, :bb, :bc],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                                    _unp_aaa(mycc, t3aaa, t3_tmp, l0, l1, m0, m1, i0, i1, a0, a1, b0, b1, c0, c1)
                                    einsum_("lmjk,lmiabc->ijkabc", mycc.W_oooo[l0:l1, m0:m1, j0:j1, k0:k1],
                                        t3_tmp[:bl, :bm, :bi, :ba, :bb, :bc],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)

                            time2 = log.timer_debug1('t3aaa: (oo + oooo) * t3aaa iter: '
                                '[%2d, %2d] [%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d] [%3d, %3d]:' % (
                                    k0, k1, j0, j1, i0, i1, c0, c1, b0, b1, a0, a1), *time2)

                            # R3aaa: P6
                            for l0, l1 in lib.prange(0, nocca, blksize_o_aaa):
                                bl = l1 - l0
                                for d0, d1 in lib.prange(0, nvira, blksize_v_aaa):
                                    bd = d1 - d0
                                    _unp_aaa(mycc, t3aaa, t3_tmp, l0, l1, j0, j1, k0, k1, d0, d1, b0, b1, c0, c1)
                                    einsum_("alid,ljkdbc->ijkabc", mycc.W_voov[a0:a1, l0:l1, i0:i1, d0:d1],
                                        t3_tmp[:bl, :bj, :bk, :bd, :bb, :bc],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=1.0, beta=1.0)
                                    _unp_aaa(mycc, t3aaa, t3_tmp, l0, l1, j0, j1, k0, k1, d0, d1, a0, a1, c0, c1)
                                    einsum_("blid,ljkdac->ijkabc", mycc.W_voov[b0:b1, l0:l1, i0:i1, d0:d1],
                                        t3_tmp[:bl, :bj, :bk, :bd, :ba, :bc],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-1.0, beta=1.0)
                                    _unp_aaa(mycc, t3aaa, t3_tmp, l0, l1, j0, j1, k0, k1, d0, d1, a0, a1, b0, b1)
                                    einsum_("clid,ljkdab->ijkabc", mycc.W_voov[c0:c1, l0:l1, i0:i1, d0:d1],
                                        t3_tmp[:bl, :bj, :bk, :bd, :ba, :bb],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=1.0, beta=1.0)
                                    _unp_aaa(mycc, t3aaa, t3_tmp, l0, l1, i0, i1, k0, k1, d0, d1, b0, b1, c0, c1)
                                    einsum_("aljd,likdbc->ijkabc", mycc.W_voov[a0:a1, l0:l1, j0:j1, d0:d1],
                                        t3_tmp[:bl, :bi, :bk, :bd, :bb, :bc],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-1.0, beta=1.0)
                                    _unp_aaa(mycc, t3aaa, t3_tmp, l0, l1, i0, i1, k0, k1, d0, d1, a0, a1, c0, c1)
                                    einsum_("bljd,likdac->ijkabc", mycc.W_voov[b0:b1, l0:l1, j0:j1, d0:d1],
                                        t3_tmp[:bl, :bi, :bk, :bd, :ba, :bc],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=1.0, beta=1.0)
                                    _unp_aaa(mycc, t3aaa, t3_tmp, l0, l1, i0, i1, k0, k1, d0, d1, a0, a1, b0, b1)
                                    einsum_("cljd,likdab->ijkabc", mycc.W_voov[c0:c1, l0:l1, j0:j1, d0:d1],
                                        t3_tmp[:bl, :bi, :bk, :bd, :ba, :bb],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-1.0, beta=1.0)
                                    _unp_aaa(mycc, t3aaa, t3_tmp, l0, l1, i0, i1, j0, j1, d0, d1, b0, b1, c0, c1)
                                    einsum_("alkd,lijdbc->ijkabc", mycc.W_voov[a0:a1, l0:l1, k0:k1, d0:d1],
                                        t3_tmp[:bl, :bi, :bj, :bd, :bb, :bc],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=1.0, beta=1.0)
                                    _unp_aaa(mycc, t3aaa, t3_tmp, l0, l1, i0, i1, j0, j1, d0, d1, a0, a1, c0, c1)
                                    einsum_("blkd,lijdac->ijkabc", mycc.W_voov[b0:b1, l0:l1, k0:k1, d0:d1],
                                        t3_tmp[:bl, :bi, :bj, :bd, :ba, :bc],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-1.0, beta=1.0)
                                    _unp_aaa(mycc, t3aaa, t3_tmp, l0, l1, i0, i1, j0, j1, d0, d1, a0, a1, b0, b1)
                                    einsum_("clkd,lijdab->ijkabc", mycc.W_voov[c0:c1, l0:l1, k0:k1, d0:d1],
                                        t3_tmp[:bl, :bi, :bj, :bd, :ba, :bb],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=1.0, beta=1.0)

                            time2 = log.timer_debug1('t3aaa: voov * t3aaa iter: '
                                '[%2d, %2d] [%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d] [%3d, %3d]:' % (
                                    k0, k1, j0, j1, i0, i1, c0, c1, b0, b1, a0, a1), *time2)

                            # R3aaa: P7
                            _unp_aab(mycc, t3aab, t3_tmp_2, j0, j1, k0, k1, b0, b1, c0, c1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum_("alid,jkbcld->ijkabc", mycc.W_vOoV[a0:a1, :, i0:i1, :],
                                t3_tmp_2[:bj, :bk, :bb, :bc, :, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=1.0, beta=1.0)
                            _unp_aab(mycc, t3aab, t3_tmp_2, j0, j1, k0, k1, a0, a1, c0, c1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum_("blid,jkacld->ijkabc", mycc.W_vOoV[b0:b1, :, i0:i1, :],
                                t3_tmp_2[:bj, :bk, :ba, :bc, :, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-1.0, beta=1.0)
                            _unp_aab(mycc, t3aab, t3_tmp_2, j0, j1, k0, k1, a0, a1, b0, b1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum_("clid,jkabld->ijkabc", mycc.W_vOoV[c0:c1, :, i0:i1, :],
                                t3_tmp_2[:bj, :bk, :ba, :bb, :, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=1.0, beta=1.0)
                            _unp_aab(mycc, t3aab, t3_tmp_2, i0, i1, k0, k1, b0, b1, c0, c1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum_("aljd,ikbcld->ijkabc", mycc.W_vOoV[a0:a1, :, j0:j1, :],
                                t3_tmp_2[:bi, :bk, :bb, :bc, :, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-1.0, beta=1.0)
                            _unp_aab(mycc, t3aab, t3_tmp_2, i0, i1, k0, k1, a0, a1, c0, c1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum_("bljd,ikacld->ijkabc", mycc.W_vOoV[b0:b1, :, j0:j1, :],
                                t3_tmp_2[:bi, :bk, :ba, :bc, :, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=1.0, beta=1.0)
                            _unp_aab(mycc, t3aab, t3_tmp_2, i0, i1, k0, k1, a0, a1, b0, b1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum_("cljd,ikabld->ijkabc", mycc.W_vOoV[c0:c1, :, j0:j1, :],
                                t3_tmp_2[:bi, :bk, :ba, :bb, :, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-1.0, beta=1.0)
                            _unp_aab(mycc, t3aab, t3_tmp_2, i0, i1, j0, j1, b0, b1, c0, c1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum_("alkd,ijbcld->ijkabc", mycc.W_vOoV[a0:a1, :, k0:k1, :],
                                t3_tmp_2[:bi, :bj, :bb, :bc, :, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=1.0, beta=1.0)
                            _unp_aab(mycc, t3aab, t3_tmp_2, i0, i1, j0, j1, a0, a1, c0, c1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum_("blkd,ijacld->ijkabc", mycc.W_vOoV[b0:b1, :, k0:k1, :],
                                t3_tmp_2[:bi, :bj, :ba, :bc, :, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-1.0, beta=1.0)
                            _unp_aab(mycc, t3aab, t3_tmp_2, i0, i1, j0, j1, a0, a1, b0, b1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum_("clkd,ijabld->ijkabc", mycc.W_vOoV[c0:c1, :, k0:k1, :],
                                t3_tmp_2[:bi, :bj, :ba, :bb, :, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=1.0, beta=1.0)

                            time2 = log.timer_debug1('t3aaa: vOov * t3aab iter: '
                                '[%2d, %2d] [%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d] [%3d, %3d]:' % (
                                    k0, k1, j0, j1, i0, i1, c0, c1, b0, b1, a0, a1), *time2)

                            _update_packed_aaa(mycc, r3aaa, r3_tmp, i0, i1, j0, j1, k0, k1,
                                a0, a1, b0, b1, c0, c1, alpha=1.0, beta=0.0)
    r3_tmp = None
    t3_tmp = None
    t3_tmp_2 = None

    time1 = log.timer_debug1('t3: r3aaa', *time1)
    return r3aaa

def compute_r3bbb_tril_uhf(mycc, t2, t3):
    time1 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    nocca, nvira = mycc.nocca, mycc.nvira
    noccb, nvirb = mycc.noccb, mycc.nvirb
    blksize_o_aaa, blksize_v_aaa = mycc.blksize_o_aaa, mycc.blksize_v_aaa
    _, _, t2bb = t2
    _, _, t3bba, t3bbb = t3

    r3bbb = np.zeros_like(t3bbb)
    time2 = logger.process_clock(), logger.perf_counter()

    W_VVVO, W_OVOO = mycc.W_VVVO, mycc.W_OVOO
    tf_VV, tf_OO = mycc.tf_VV, mycc.tf_OO
    r3_tmp = np.empty((blksize_o_aaa,) * 3 + (blksize_v_aaa,) * 3, dtype=t3bbb.dtype)
    t3_tmp = np.empty((blksize_o_aaa,) * 3 + (blksize_v_aaa,) * 3, dtype=t3bbb.dtype)
    t3_tmp_2 = np.empty((blksize_o_aaa,) * 2 + (blksize_v_aaa,) * 2 + (nocca,) + (nvira,), dtype=t3bbb.dtype)
    for k0, k1 in lib.prange(0, noccb, blksize_o_aaa):
        bk = k1 - k0
        for j0, j1 in lib.prange(0, k1 - 1, blksize_o_aaa):
            bj = j1 - j0
            for i0, i1 in lib.prange(0, j1 - 1, blksize_o_aaa):
                bi = i1 - i0
                for c0, c1 in lib.prange(0, nvirb, blksize_v_aaa):
                    bc = c1 - c0
                    for b0, b1 in lib.prange(0, c1 - 1, blksize_v_aaa):
                        bb = b1 - b0
                        for a0, a1 in lib.prange(0, b1 - 1, blksize_v_aaa):
                            ba = a1 - a0
                            # R3bbb: P0
                            einsum_("bcdk,ijad->ijkabc", W_VVVO[b0:b1, c0:c1, :, k0:k1], t2bb[i0:i1, j0:j1, a0:a1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=0.0)
                            einsum_("cbdk,ijad->ijkabc", W_VVVO[c0:c1, b0:b1, :, k0:k1], t2bb[i0:i1, j0:j1, a0:a1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("acdk,ijbd->ijkabc", W_VVVO[a0:a1, c0:c1, :, k0:k1], t2bb[i0:i1, j0:j1, b0:b1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("abdk,ijcd->ijkabc", W_VVVO[a0:a1, b0:b1, :, k0:k1], t2bb[i0:i1, j0:j1, c0:c1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("cadk,ijbd->ijkabc", W_VVVO[c0:c1, a0:a1, :, k0:k1], t2bb[i0:i1, j0:j1, b0:b1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("badk,ijcd->ijkabc", W_VVVO[b0:b1, a0:a1, :, k0:k1], t2bb[i0:i1, j0:j1, c0:c1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("bcdj,ikad->ijkabc", W_VVVO[b0:b1, c0:c1, :, j0:j1], t2bb[i0:i1, k0:k1, a0:a1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("cbdj,ikad->ijkabc", W_VVVO[c0:c1, b0:b1, :, j0:j1], t2bb[i0:i1, k0:k1, a0:a1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("acdj,ikbd->ijkabc", W_VVVO[a0:a1, c0:c1, :, j0:j1], t2bb[i0:i1, k0:k1, b0:b1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("abdj,ikcd->ijkabc", W_VVVO[a0:a1, b0:b1, :, j0:j1], t2bb[i0:i1, k0:k1, c0:c1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("cadj,ikbd->ijkabc", W_VVVO[c0:c1, a0:a1, :, j0:j1], t2bb[i0:i1, k0:k1, b0:b1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("badj,ikcd->ijkabc", W_VVVO[b0:b1, a0:a1, :, j0:j1], t2bb[i0:i1, k0:k1, c0:c1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("bcdi,jkad->ijkabc", W_VVVO[b0:b1, c0:c1, :, i0:i1], t2bb[j0:j1, k0:k1, a0:a1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("cbdi,jkad->ijkabc", W_VVVO[c0:c1, b0:b1, :, i0:i1], t2bb[j0:j1, k0:k1, a0:a1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("acdi,jkbd->ijkabc", W_VVVO[a0:a1, c0:c1, :, i0:i1], t2bb[j0:j1, k0:k1, b0:b1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("abdi,jkcd->ijkabc", W_VVVO[a0:a1, b0:b1, :, i0:i1], t2bb[j0:j1, k0:k1, c0:c1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("cadi,jkbd->ijkabc", W_VVVO[c0:c1, a0:a1, :, i0:i1], t2bb[j0:j1, k0:k1, b0:b1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("badi,jkcd->ijkabc", W_VVVO[b0:b1, a0:a1, :, i0:i1], t2bb[j0:j1, k0:k1, c0:c1, :],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            # R3bbb: P1
                            einsum_("lcjk,ilab->ijkabc", W_OVOO[:, c0:c1, j0:j1, k0:k1], t2bb[i0:i1, :, a0:a1, b0:b1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("lbjk,ilac->ijkabc", W_OVOO[:, b0:b1, j0:j1, k0:k1], t2bb[i0:i1, :, a0:a1, c0:c1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("lajk,ilbc->ijkabc", W_OVOO[:, a0:a1, j0:j1, k0:k1], t2bb[i0:i1, :, b0:b1, c0:c1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("lckj,ilab->ijkabc", W_OVOO[:, c0:c1, k0:k1, j0:j1], t2bb[i0:i1, :, a0:a1, b0:b1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("lbkj,ilac->ijkabc", W_OVOO[:, b0:b1, k0:k1, j0:j1], t2bb[i0:i1, :, a0:a1, c0:c1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("lakj,ilbc->ijkabc", W_OVOO[:, a0:a1, k0:k1, j0:j1], t2bb[i0:i1, :, b0:b1, c0:c1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("lcik,jlab->ijkabc", W_OVOO[:, c0:c1, i0:i1, k0:k1], t2bb[j0:j1, :, a0:a1, b0:b1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("lbik,jlac->ijkabc", W_OVOO[:, b0:b1, i0:i1, k0:k1], t2bb[j0:j1, :, a0:a1, c0:c1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("laik,jlbc->ijkabc", W_OVOO[:, a0:a1, i0:i1, k0:k1], t2bb[j0:j1, :, b0:b1, c0:c1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("lcij,klab->ijkabc", W_OVOO[:, c0:c1, i0:i1, j0:j1], t2bb[k0:k1, :, a0:a1, b0:b1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("lbij,klac->ijkabc", W_OVOO[:, b0:b1, i0:i1, j0:j1], t2bb[k0:k1, :, a0:a1, c0:c1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("laij,klbc->ijkabc", W_OVOO[:, a0:a1, i0:i1, j0:j1], t2bb[k0:k1, :, b0:b1, c0:c1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("lcki,jlab->ijkabc", W_OVOO[:, c0:c1, k0:k1, i0:i1], t2bb[j0:j1, :, a0:a1, b0:b1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("lbki,jlac->ijkabc", W_OVOO[:, b0:b1, k0:k1, i0:i1], t2bb[j0:j1, :, a0:a1, c0:c1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("laki,jlbc->ijkabc", W_OVOO[:, a0:a1, k0:k1, i0:i1], t2bb[j0:j1, :, b0:b1, c0:c1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("lcji,klab->ijkabc", W_OVOO[:, c0:c1, j0:j1, i0:i1], t2bb[k0:k1, :, a0:a1, b0:b1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                            einsum_("lbji,klac->ijkabc", W_OVOO[:, b0:b1, j0:j1, i0:i1], t2bb[k0:k1, :, a0:a1, c0:c1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                            einsum_("laji,klbc->ijkabc", W_OVOO[:, a0:a1, j0:j1, i0:i1], t2bb[k0:k1, :, b0:b1, c0:c1],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)

                            time2 = log.timer_debug1('t3bbb: (VVVO + OVOO) * t2bb iter: '
                                '[%2d, %2d] [%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d] [%3d, %3d]:' % (
                                    k0, k1, j0, j1, i0, i1, c0, c1, b0, b1, a0, a1), *time2)

                            # R3bbb: P2, P4
                            for d0, d1 in lib.prange(0, nvirb, blksize_v_aaa):
                                bd = d1 - d0
                                _unp_bbb(mycc, t3bbb, t3_tmp, i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, d0, d1)
                                einsum_("cd,ijkabd->ijkabc", tf_VV[c0:c1, d0:d1], t3_tmp[:bi, :bj, :bk, :ba, :bb, :bd],
                                    out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=1.0, beta=1.0)
                                _unp_bbb(mycc, t3bbb, t3_tmp, i0, i1, j0, j1, k0, k1, a0, a1, c0, c1, d0, d1)
                                einsum_("bd,ijkacd->ijkabc", tf_VV[b0:b1, d0:d1], t3_tmp[:bi, :bj, :bk, :ba, :bc, :bd],
                                    out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-1.0, beta=1.0)
                                _unp_bbb(mycc, t3bbb, t3_tmp, i0, i1, j0, j1, k0, k1, b0, b1, c0, c1, d0, d1)
                                einsum_("ad,ijkbcd->ijkabc", tf_VV[a0:a1, d0:d1], t3_tmp[:bi, :bj, :bk, :bb, :bc, :bd],
                                    out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=1.0, beta=1.0)
                                for e0, e1 in lib.prange(0, nvirb, blksize_v_aaa):
                                    be = e1 - e0
                                    _unp_bbb(mycc, t3bbb, t3_tmp, i0, i1, j0, j1, k0, k1, d0, d1, e0, e1, c0, c1)
                                    einsum_("abde,ijkdec->ijkabc", mycc.W_VVVV[a0:a1, b0:b1, d0:d1, e0:e1],
                                        t3_tmp[:bi, :bj, :bk, :bd, :be, :bc],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                                    _unp_bbb(mycc, t3bbb, t3_tmp, i0, i1, j0, j1, k0, k1, d0, d1, e0, e1, b0, b1)
                                    einsum_("acde,ijkdeb->ijkabc", mycc.W_VVVV[a0:a1, c0:c1, d0:d1, e0:e1],
                                        t3_tmp[:bi, :bj, :bk, :bd, :be, :bb],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                                    _unp_bbb(mycc, t3bbb, t3_tmp, i0, i1, j0, j1, k0, k1, d0, d1, e0, e1, a0, a1)
                                    einsum_("bcde,ijkdea->ijkabc", mycc.W_VVVV[b0:b1, c0:c1, d0:d1, e0:e1],
                                        t3_tmp[:bi, :bj, :bk, :bd, :be, :ba],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)

                            time2 = log.timer_debug1('t3bbb: (VV + VVVV) * t3bbb iter: '
                                '[%2d, %2d] [%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d] [%3d, %3d]:' % (
                                    k0, k1, j0, j1, i0, i1, c0, c1, b0, b1, a0, a1), *time2)

                            # R3bbb: P3, P5
                            for l0, l1 in lib.prange(0, noccb, blksize_o_aaa):
                                bl = l1 - l0
                                _unp_bbb(mycc, t3bbb, t3_tmp, i0, i1, j0, j1, l0, l1, a0, a1, b0, b1, c0, c1)
                                einsum_("lk,ijlabc->ijkabc", tf_OO[l0:l1, k0:k1], t3_tmp[:bi, :bj, :bl, :ba, :bb, :bc],
                                    out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-1.0, beta=1.0)
                                _unp_bbb(mycc, t3bbb, t3_tmp, i0, i1, k0, k1, l0, l1, a0, a1, b0, b1, c0, c1)
                                einsum_("lj,iklabc->ijkabc", tf_OO[l0:l1, j0:j1], t3_tmp[:bi, :bk, :bl, :ba, :bb, :bc],
                                    out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=1.0, beta=1.0)
                                _unp_bbb(mycc, t3bbb, t3_tmp, j0, j1, k0, k1, l0, l1, a0, a1, b0, b1, c0, c1)
                                einsum_("li,jklabc->ijkabc", tf_OO[l0:l1, i0:i1], t3_tmp[:bj, :bk, :bl, :ba, :bb, :bc],
                                    out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-1.0, beta=1.0)
                                for m0, m1 in lib.prange(0, noccb, blksize_o_aaa):
                                    bm = m1 - m0
                                    _unp_bbb(mycc, t3bbb, t3_tmp, l0, l1, m0, m1, k0, k1, a0, a1, b0, b1, c0, c1)
                                    einsum_("lmij,lmkabc->ijkabc", mycc.W_OOOO[l0:l1, m0:m1, i0:i1, j0:j1],
                                        t3_tmp[:bl, :bm, :bk, :ba, :bb, :bc],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)
                                    _unp_bbb(mycc, t3bbb, t3_tmp, l0, l1, m0, m1, j0, j1, a0, a1, b0, b1, c0, c1)
                                    einsum_("lmik,lmjabc->ijkabc", mycc.W_OOOO[l0:l1, m0:m1, i0:i1, k0:k1],
                                        t3_tmp[:bl, :bm, :bj, :ba, :bb, :bc],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-0.5, beta=1.0)
                                    _unp_bbb(mycc, t3bbb, t3_tmp, l0, l1, m0, m1, i0, i1, a0, a1, b0, b1, c0, c1)
                                    einsum_("lmjk,lmiabc->ijkabc", mycc.W_OOOO[l0:l1, m0:m1, j0:j1, k0:k1],
                                        t3_tmp[:bl, :bm, :bi, :ba, :bb, :bc],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=0.5, beta=1.0)

                            time2 = log.timer_debug1('t3bbb: (OO + OOOO) * t3bbb iter: '
                                '[%2d, %2d] [%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d] [%3d, %3d]:' % (
                                    k0, k1, j0, j1, i0, i1, c0, c1, b0, b1, a0, a1), *time2)

                            # R3bbb: P6
                            for l0, l1 in lib.prange(0, noccb, blksize_o_aaa):
                                bl = l1 - l0
                                for d0, d1 in lib.prange(0, nvirb, blksize_v_aaa):
                                    bd = d1 - d0
                                    _unp_bbb(mycc, t3bbb, t3_tmp, l0, l1, j0, j1, k0, k1, d0, d1, b0, b1, c0, c1)
                                    einsum_("alid,ljkdbc->ijkabc", mycc.W_VOOV[a0:a1, l0:l1, i0:i1, d0:d1],
                                        t3_tmp[:bl, :bj, :bk, :bd, :bb, :bc],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=1.0, beta=1.0)
                                    _unp_bbb(mycc, t3bbb, t3_tmp, l0, l1, j0, j1, k0, k1, d0, d1, a0, a1, c0, c1)
                                    einsum_("blid,ljkdac->ijkabc", mycc.W_VOOV[b0:b1, l0:l1, i0:i1, d0:d1],
                                        t3_tmp[:bl, :bj, :bk, :bd, :ba, :bc],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-1.0, beta=1.0)
                                    _unp_bbb(mycc, t3bbb, t3_tmp, l0, l1, j0, j1, k0, k1, d0, d1, a0, a1, b0, b1)
                                    einsum_("clid,ljkdab->ijkabc", mycc.W_VOOV[c0:c1, l0:l1, i0:i1, d0:d1],
                                        t3_tmp[:bl, :bj, :bk, :bd, :ba, :bb],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=1.0, beta=1.0)
                                    _unp_bbb(mycc, t3bbb, t3_tmp, l0, l1, i0, i1, k0, k1, d0, d1, b0, b1, c0, c1)
                                    einsum_("aljd,likdbc->ijkabc", mycc.W_VOOV[a0:a1, l0:l1, j0:j1, d0:d1],
                                        t3_tmp[:bl, :bi, :bk, :bd, :bb, :bc],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-1.0, beta=1.0)
                                    _unp_bbb(mycc, t3bbb, t3_tmp, l0, l1, i0, i1, k0, k1, d0, d1, a0, a1, c0, c1)
                                    einsum_("bljd,likdac->ijkabc", mycc.W_VOOV[b0:b1, l0:l1, j0:j1, d0:d1],
                                        t3_tmp[:bl, :bi, :bk, :bd, :ba, :bc],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=1.0, beta=1.0)
                                    _unp_bbb(mycc, t3bbb, t3_tmp, l0, l1, i0, i1, k0, k1, d0, d1, a0, a1, b0, b1)
                                    einsum_("cljd,likdab->ijkabc", mycc.W_VOOV[c0:c1, l0:l1, j0:j1, d0:d1],
                                        t3_tmp[:bl, :bi, :bk, :bd, :ba, :bb],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-1.0, beta=1.0)
                                    _unp_bbb(mycc, t3bbb, t3_tmp, l0, l1, i0, i1, j0, j1, d0, d1, b0, b1, c0, c1)
                                    einsum_("alkd,lijdbc->ijkabc", mycc.W_VOOV[a0:a1, l0:l1, k0:k1, d0:d1],
                                        t3_tmp[:bl, :bi, :bj, :bd, :bb, :bc],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=1.0, beta=1.0)
                                    _unp_bbb(mycc, t3bbb, t3_tmp, l0, l1, i0, i1, j0, j1, d0, d1, a0, a1, c0, c1)
                                    einsum_("blkd,lijdac->ijkabc", mycc.W_VOOV[b0:b1, l0:l1, k0:k1, d0:d1],
                                        t3_tmp[:bl, :bi, :bj, :bd, :ba, :bc],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-1.0, beta=1.0)
                                    _unp_bbb(mycc, t3bbb, t3_tmp, l0, l1, i0, i1, j0, j1, d0, d1, a0, a1, b0, b1)
                                    einsum_("clkd,lijdab->ijkabc", mycc.W_VOOV[c0:c1, l0:l1, k0:k1, d0:d1],
                                        t3_tmp[:bl, :bi, :bj, :bd, :ba, :bb],
                                        out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=1.0, beta=1.0)

                            time2 = log.timer_debug1('t3bbb: VOOV * t3bbb iter: '
                                '[%2d, %2d] [%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d] [%3d, %3d]:' % (
                                    k0, k1, j0, j1, i0, i1, c0, c1, b0, b1, a0, a1), *time2)

                            # R3bbb: P7
                            _unp_bba(mycc, t3bba, t3_tmp_2, j0, j1, k0, k1, b0, b1, c0, c1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum_("alid,jkbcld->ijkabc", mycc.W_VoOv[a0:a1, :, i0:i1, :],
                                t3_tmp_2[:bj, :bk, :bb, :bc],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=1.0, beta=1.0)
                            _unp_bba(mycc, t3bba, t3_tmp_2, j0, j1, k0, k1, a0, a1, c0, c1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum_("blid,jkacld->ijkabc", mycc.W_VoOv[b0:b1, :, i0:i1, :],
                                t3_tmp_2[:bj, :bk, :ba, :bc],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-1.0, beta=1.0)
                            _unp_bba(mycc, t3bba, t3_tmp_2, j0, j1, k0, k1, a0, a1, b0, b1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum_("clid,jkabld->ijkabc", mycc.W_VoOv[c0:c1, :, i0:i1, :],
                                t3_tmp_2[:bj, :bk, :ba, :bb],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=1.0, beta=1.0)
                            _unp_bba(mycc, t3bba, t3_tmp_2, i0, i1, k0, k1, b0, b1, c0, c1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum_("aljd,ikbcld->ijkabc", mycc.W_VoOv[a0:a1, :, j0:j1, :],
                                t3_tmp_2[:bi, :bk, :bb, :bc],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-1.0, beta=1.0)
                            _unp_bba(mycc, t3bba, t3_tmp_2, i0, i1, k0, k1, a0, a1, c0, c1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum_("bljd,ikacld->ijkabc", mycc.W_VoOv[b0:b1, :, j0:j1, :],
                                t3_tmp_2[:bi, :bk, :ba, :bc],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=1.0, beta=1.0)
                            _unp_bba(mycc, t3bba, t3_tmp_2, i0, i1, k0, k1, a0, a1, b0, b1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum_("cljd,ikabld->ijkabc", mycc.W_VoOv[c0:c1, :, j0:j1, :],
                                t3_tmp_2[:bi, :bk, :ba, :bb],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-1.0, beta=1.0)
                            _unp_bba(mycc, t3bba, t3_tmp_2, i0, i1, j0, j1, b0, b1, c0, c1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum_("alkd,ijbcld->ijkabc", mycc.W_VoOv[a0:a1, :, k0:k1, :],
                                t3_tmp_2[:bi, :bj, :bb, :bc],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=1.0, beta=1.0)
                            _unp_bba(mycc, t3bba, t3_tmp_2, i0, i1, j0, j1, a0, a1, c0, c1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum_("blkd,ijacld->ijkabc", mycc.W_VoOv[b0:b1, :, k0:k1, :],
                                t3_tmp_2[:bi, :bj, :ba, :bc],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=-1.0, beta=1.0)
                            _unp_bba(mycc, t3bba, t3_tmp_2, i0, i1, j0, j1, a0, a1, b0, b1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum_("clkd,ijabld->ijkabc", mycc.W_VoOv[c0:c1, :, k0:k1, :],
                                t3_tmp_2[:bi, :bj, :ba, :bb],
                                out=r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc], alpha=1.0, beta=1.0)

                            time2 = log.timer_debug1('t3bbb: VoOv * t3bba iter: '
                                '[%2d, %2d] [%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d] [%3d, %3d]:' % (
                                    k0, k1, j0, j1, i0, i1, c0, c1, b0, b1, a0, a1), *time2)

                            _update_packed_bbb(mycc, r3bbb, r3_tmp, i0, i1, j0, j1, k0, k1,
                                                a0, a1, b0, b1, c0, c1, alpha=1.0, beta=0.0)
    r3_tmp = None
    t3_tmp = None
    t3_tmp_2 = None

    time1 = log.timer_debug1('t3: r3bbb', *time1)
    return r3bbb

def compute_r3aab_tril_uhf(mycc, t2, t3):
    time1 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    nocca, nvira = mycc.nocca, mycc.nvira
    noccb, nvirb = mycc.noccb, mycc.nvirb
    blksize_o_aab, blksize_v_aab = mycc.blksize_o_aab, mycc.blksize_v_aab
    t2aa, t2ab, _ = t2
    t3aaa, t3aab, t3bba, _ = t3

    r3aab = np.zeros_like(t3aab)
    time2 = logger.process_clock(), logger.perf_counter()

    r3_tmp = np.empty((blksize_o_aab,) * 2 + (blksize_v_aab,) * 2 + (noccb,) + (nvirb,), dtype=t3aaa.dtype)
    t3_tmp = np.empty((blksize_o_aab,) * 2 + (blksize_v_aab,) * 2 + (noccb,) + (nvirb,), dtype=t3aaa.dtype)
    t3_tmp_2 = np.empty((blksize_o_aab,) + (noccb,) + (blksize_v_aab,) + (nvirb,)
                        + (nocca,) + (nvira,), dtype=t3aaa.dtype)
    t3_tmp_3 = np.empty((blksize_o_aab,) * 2 + (nocca,) + (blksize_v_aab,) * 2 + (nvira,), dtype=t3aaa.dtype)
    for j0, j1 in lib.prange(0, nocca, blksize_o_aab):
        bj = j1 - j0
        for i0, i1 in lib.prange(0, j1 - 1, blksize_o_aab):
            bi = i1 - i0
            for b0, b1 in lib.prange(0, nvira, blksize_v_aab):
                bb = b1 - b0
                for a0, a1 in lib.prange(0, b1 - 1, blksize_v_aab):
                    ba = a1 - a0
                    # R3aab: P0
                    einsum_("bcdk,ijad->ijabkc", mycc.W_vVvO[b0:b1], t2aa[i0:i1, j0:j1, a0:a1, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=0.0)
                    einsum_("acdk,ijbd->ijabkc", mycc.W_vVvO[a0:a1], t2aa[i0:i1, j0:j1, b0:b1, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                    # R3aab: P3
                    einsum_("lcjk,ilab->ijabkc", mycc.W_oVoO[:, :, j0:j1, :], t2aa[i0:i1, :, a0:a1, b0:b1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                    einsum_("lcik,jlab->ijabkc", mycc.W_oVoO[:, :, i0:i1, :], t2aa[j0:j1, :, a0:a1, b0:b1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)

                    time2 = log.timer_debug1('t3aab: (vVvO + oVoO) * t2aa iter: '
                        '[%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d]:' % (j0, j1, i0, i1, b0, b1, a0, a1), *time2)

                    # R3aab: P1
                    einsum_("bcjd,iakd->ijabkc", mycc.W_vVoV[b0:b1, :, j0:j1, :], t2ab[i0:i1, a0:a1, :, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                    einsum_("acjd,ibkd->ijabkc", mycc.W_vVoV[a0:a1, :, j0:j1, :], t2ab[i0:i1, b0:b1, :, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                    einsum_("bcid,jakd->ijabkc", mycc.W_vVoV[b0:b1, :, i0:i1, :], t2ab[j0:j1, a0:a1, :, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                    einsum_("acid,jbkd->ijabkc", mycc.W_vVoV[a0:a1, :, i0:i1, :], t2ab[j0:j1, b0:b1, :, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                    # R3aab: P2
                    einsum_("abdi,jdkc->ijabkc", mycc.W_vvvo[a0:a1, b0:b1, :, i0:i1], t2ab[j0:j1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-0.5, beta=1.0)
                    einsum_("badi,jdkc->ijabkc", mycc.W_vvvo[b0:b1, a0:a1, :, i0:i1], t2ab[j0:j1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=0.5, beta=1.0)
                    einsum_("abdj,idkc->ijabkc", mycc.W_vvvo[a0:a1, b0:b1, :, j0:j1], t2ab[i0:i1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=0.5, beta=1.0)
                    einsum_("badj,idkc->ijabkc", mycc.W_vvvo[b0:b1, a0:a1, :, j0:j1], t2ab[i0:i1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-0.5, beta=1.0)
                    # R3aab: P4
                    einsum_("aljk,iblc->ijabkc", mycc.W_vOoO[a0:a1, :, j0:j1, :], t2ab[i0:i1, b0:b1, :, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                    einsum_("bljk,ialc->ijabkc", mycc.W_vOoO[b0:b1, :, j0:j1, :], t2ab[i0:i1, a0:a1, :, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                    einsum_("alik,jblc->ijabkc", mycc.W_vOoO[a0:a1, :, i0:i1, :], t2ab[j0:j1, b0:b1, :, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                    einsum_("blik,jalc->ijabkc", mycc.W_vOoO[b0:b1, :, i0:i1, :], t2ab[j0:j1, a0:a1, :, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                    # R3aab: P5
                    einsum_("laij,lbkc->ijabkc", mycc.W_ovoo[:, a0:a1, i0:i1, j0:j1], t2ab[:, b0:b1, :, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=0.5, beta=1.0)
                    einsum_("lbij,lakc->ijabkc", mycc.W_ovoo[:, b0:b1, i0:i1, j0:j1], t2ab[:, a0:a1, :, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-0.5, beta=1.0)
                    einsum_("laji,lbkc->ijabkc", mycc.W_ovoo[:, a0:a1, j0:j1, i0:i1], t2ab[:, b0:b1, :, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-0.5, beta=1.0)
                    einsum_("lbji,lakc->ijabkc", mycc.W_ovoo[:, b0:b1, j0:j1, i0:i1], t2ab[:, a0:a1, :, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=0.5, beta=1.0)

                    time2 = log.timer_debug1('t3aab: (vVoV + vvvo + vOoO + ovoo) * t2ab iter: '
                        '[%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d]:' % (j0, j1, i0, i1, b0, b1, a0, a1), *time2)

                    # R3aab: P6 & P8 & P19
                    _unp_aab(mycc, t3aab, t3_tmp, i0, i1, j0, j1, a0, a1, b0, b1)
                    einsum_("cd,ijabkd->ijabkc", mycc.tf_VV, t3_tmp[:bi, :bj, :ba, :bb],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                    einsum_("lk,ijablc->ijabkc", mycc.tf_OO, t3_tmp[:bi, :bj, :ba, :bb],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                    einsum_("clkd,ijabld->ijabkc", mycc.W_VOOV, t3_tmp[:bi, :bj, :ba, :bb],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)

                    time2 = log.timer_debug1('t3aab: (VV + OO + VOOV) * t3aab iter: '
                        '[%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d]:' % (j0, j1, i0, i1, b0, b1, a0, a1), *time2)

                    # R3aab: P7 & P11 & P17
                    for d0, d1 in lib.prange(0, nvira, blksize_v_aab):
                        bd = d1 - d0
                        _unp_aab(mycc, t3aab, t3_tmp, i0, i1, j0, j1, b0, b1, d0, d1)
                        einsum_("ad,ijbdkc->ijabkc", mycc.tf_vv[a0:a1, d0:d1], t3_tmp[:bi, :bj, :bb, :bd],
                            out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                        einsum_("acde,ijbdke->ijabkc", mycc.W_vVvV[a0:a1, :, d0:d1, :], t3_tmp[:bi, :bj, :bb, :bd],
                            out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                        einsum_("aldk,ijbdlc->ijabkc", mycc.W_vOvO_tc[a0:a1, :, d0:d1, :],
                            t3_tmp[:bi, :bj, :bb, :bd], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                        _unp_aab(mycc, t3aab, t3_tmp, i0, i1, j0, j1, a0, a1, d0, d1)
                        einsum_("bd,ijadkc->ijabkc", mycc.tf_vv[b0:b1, d0:d1], t3_tmp[:bi, :bj, :ba, :bd],
                            out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                        einsum_("bcde,ijadke->ijabkc", mycc.W_vVvV[b0:b1, :, d0:d1, :], t3_tmp[:bi, :bj, :ba, :bd],
                            out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                        einsum_("bldk,ijadlc->ijabkc", mycc.W_vOvO_tc[b0:b1, :, d0:d1, :],
                            t3_tmp[:bi, :bj, :ba, :bd], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)

                    time2 = log.timer_debug1('t3aab: (vv + vVvV + vOvO) * t3aab iter: '
                        '[%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d]:' % (j0, j1, i0, i1, b0, b1, a0, a1), *time2)

                    # R3aab: P9 & P13 & P16
                    for l0, l1 in lib.prange(0, nocca, blksize_o_aab):
                        bl = l1 - l0
                        _unp_aab(mycc, t3aab, t3_tmp, j0, j1, l0, l1, a0, a1, b0, b1)
                        einsum_("li,jlabkc->ijabkc", mycc.tf_oo[l0:l1, i0:i1], t3_tmp[:bj, :bl, :ba, :bb],
                            out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                        einsum_("lmik,jlabmc->ijabkc", mycc.W_oOoO[l0:l1, :, i0:i1, :], t3_tmp[:bj, :bl, :ba, :bb],
                            out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                        einsum_("lcid,jlabkd->ijabkc", mycc.W_oVoV[l0:l1, :, i0:i1, :], t3_tmp[:bj, :bl, :ba, :bb],
                            out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                        _unp_aab(mycc, t3aab, t3_tmp, i0, i1, l0, l1, a0, a1, b0, b1)
                        einsum_("lj,ilabkc->ijabkc", mycc.tf_oo[l0:l1, j0:j1], t3_tmp[:bi, :bl, :ba, :bb],
                            out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                        einsum_("lmjk,ilabmc->ijabkc", mycc.W_oOoO[l0:l1, :, j0:j1, :], t3_tmp[:bi, :bl, :ba, :bb],
                            out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                        einsum_("lcjd,ilabkd->ijabkc", mycc.W_oVoV[l0:l1, :, j0:j1, :], t3_tmp[:bi, :bl, :ba, :bb],
                            out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)

                    time2 = log.timer_debug1('t3aab: (oo + oOoO + oVoV) * t3aab iter: '
                        '[%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d]:' % (j0, j1, i0, i1, b0, b1, a0, a1), *time2)

                    # R3aab: P10
                    for d0, d1 in lib.prange(0, nvira, blksize_v_aab):
                        bd = d1 - d0
                        for e0, e1 in lib.prange(0, nvira, blksize_v_aab):
                            be = e1 - e0
                            _unp_aab(mycc, t3aab, t3_tmp, i0, i1, j0, j1, d0, d1, e0, e1)
                            einsum_("abde,ijdekc->ijabkc", mycc.W_vvvv[a0:a1, b0:b1, d0:d1, e0:e1],
                                t3_tmp[:bi, :bj, :bd, :be], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=0.5, beta=1.0)

                    time2 = log.timer_debug1('t3aab: vvvv * t3aab iter: '
                        '[%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d]:' % (j0, j1, i0, i1, b0, b1, a0, a1), *time2)

                    # R3aab: P12
                    for l0, l1 in lib.prange(0, nocca, blksize_o_aab):
                        bl = l1 - l0
                        for m0, m1 in lib.prange(0, nocca, blksize_o_aab):
                            bm = m1 - m0
                            _unp_aab(mycc, t3aab, t3_tmp, l0, l1, m0, m1, a0, a1, b0, b1)
                            einsum_("lmij,lmabkc->ijabkc", mycc.W_oooo[l0:l1, m0:m1, i0:i1, j0:j1],
                                t3_tmp[:bl, :bm, :ba, :bb], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=0.5, beta=1.0)

                    time2 = log.timer_debug1('t3aab: oooo * t3aab iter: '
                        '[%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d]:' % (j0, j1, i0, i1, b0, b1, a0, a1), *time2)

                    # R3aab: P14
                    for l0, l1 in lib.prange(0, nocca, blksize_o_aab):
                        bl = l1 - l0
                        for d0, d1 in lib.prange(0, nvira, blksize_v_aab):
                            bd = d1 - d0
                            _unp_aab(mycc, t3aab, t3_tmp, l0, l1, j0, j1, d0, d1, b0, b1)
                            einsum_("alid,ljdbkc->ijabkc", mycc.W_voov[a0:a1, l0:l1, i0:i1, d0:d1],
                                t3_tmp[:bl, :bj, :bd, :bb], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                            _unp_aab(mycc, t3aab, t3_tmp, l0, l1, j0, j1, d0, d1, a0, a1)
                            einsum_("blid,ljdakc->ijabkc", mycc.W_voov[b0:b1, l0:l1, i0:i1, d0:d1],
                                t3_tmp[:bl, :bj, :bd, :ba], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                            _unp_aab(mycc, t3aab, t3_tmp, l0, l1, i0, i1, d0, d1, b0, b1)
                            einsum_("aljd,lidbkc->ijabkc", mycc.W_voov[a0:a1, l0:l1, j0:j1, d0:d1],
                                t3_tmp[:bl, :bi, :bd, :bb], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                            _unp_aab(mycc, t3aab, t3_tmp, l0, l1, i0, i1, d0, d1, a0, a1)
                            einsum_("bljd,lidakc->ijabkc", mycc.W_voov[b0:b1, l0:l1, j0:j1, d0:d1],
                                t3_tmp[:bl, :bi, :bd, :ba], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)

                    time2 = log.timer_debug1('t3aab: voov * t3aab iter: '
                        '[%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d]:' % (j0, j1, i0, i1, b0, b1, a0, a1), *time2)

                    # R3aab: P15
                    # FIXME: This unpacking step performs redundant operations and should be optimized
                    for l0, l1 in lib.prange(0, noccb, blksize_o_aab):
                        bl = l1 - l0
                        for d0, d1 in lib.prange(0, nvirb, blksize_v_aab):
                            bd = d1 - d0
                            _unp_bba(mycc, t3bba, t3_tmp_2, l0, l1, 0, noccb,
                                    d0, d1, 0, nvirb, blk_j=noccb, blk_b=nvirb)
                            einsum_("alid,lkdcjb->ijabkc", mycc.W_vOoV[a0:a1, l0:l1, i0:i1, d0:d1],
                                t3_tmp_2[:bl, :, :bd, :, j0:j1, b0:b1],
                                out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                            einsum_("blid,lkdcja->ijabkc", mycc.W_vOoV[b0:b1, l0:l1, i0:i1, d0:d1],
                                t3_tmp_2[:bl, :, :bd, :, j0:j1, a0:a1],
                                out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                            einsum_("aljd,lkdcib->ijabkc", mycc.W_vOoV[a0:a1, l0:l1, j0:j1, d0:d1],
                                t3_tmp_2[:bl, :, :bd, :, i0:i1, b0:b1],
                                out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                            einsum_("bljd,lkdcia->ijabkc", mycc.W_vOoV[b0:b1, l0:l1, j0:j1, d0:d1],
                                t3_tmp_2[:bl, :, :bd, :, i0:i1, a0:a1],
                                out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)

                    time2 = log.timer_debug1('t3aab: vOoV * t3bba iter: '
                        '[%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d]:' % (j0, j1, i0, i1, b0, b1, a0, a1), *time2)

                    # R3aab: P18
                    _unp_aaa(mycc, t3aaa, t3_tmp_3, i0, i1, j0, j1, 0, nocca, a0, a1, b0, b1, 0, nvira,
                            blk_i=blksize_o_aab, blk_j=blksize_o_aab, blk_k=nocca,
                            blk_a=blksize_v_aab, blk_b=blksize_v_aab, blk_c=nvira)
                    einsum_("clkd,ijlabd->ijabkc", mycc.W_VoOv, t3_tmp_3[:bi, :bj, :, :ba, :bb, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)

                    time2 = log.timer_debug1('t3aab: VoOv * t3aaa iter: '
                        '[%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d]:' % (j0, j1, i0, i1, b0, b1, a0, a1), *time2)

                    _update_packed_aab(mycc, r3aab, r3_tmp, i0, i1, j0, j1, a0, a1, b0, b1)
    r3_tmp = None
    t3_tmp = None
    t3_tmp_2 = None
    t3_tmp_3 = None

    time1 = log.timer_debug1('t3: r3aab', *time1)
    return r3aab

def compute_r3bba_tril_uhf(mycc, t2, t3):
    time1 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    nocca, nvira = mycc.nocca, mycc.nvira
    noccb, nvirb = mycc.noccb, mycc.nvirb
    blksize_o_aab, blksize_v_aab = mycc.blksize_o_aab, mycc.blksize_v_aab
    _, t2ab, t2bb = t2
    _, t3aab, t3bba, t3bbb = t3

    r3bba = np.zeros_like(t3bba)
    time2 = logger.process_clock(), logger.perf_counter()

    r3_tmp = np.empty((blksize_o_aab,) * 2 + (blksize_v_aab,) * 2 + (nocca,) + (nvira,), dtype=t3bbb.dtype)
    t3_tmp = np.empty((blksize_o_aab,) * 2 + (blksize_v_aab,) * 2 + (nocca,) + (nvira,), dtype=t3bbb.dtype)
    t3_tmp_2 = np.empty((blksize_o_aab,) + (nocca,) + (blksize_v_aab,)
                        + (nvira,) + (noccb,) + (nvirb,), dtype=t3bbb.dtype)
    t3_tmp_3 = np.empty((blksize_o_aab,) * 2 + (noccb,) + (blksize_v_aab,) * 2 + (nvirb,), dtype=t3bbb.dtype)
    for j0, j1 in lib.prange(0, noccb, blksize_o_aab):
        bj = j1 - j0
        for i0, i1 in lib.prange(0, j1 - 1, blksize_o_aab):
            bi = i1 - i0
            for b0, b1 in lib.prange(0, nvirb, blksize_v_aab):
                bb = b1 - b0
                for a0, a1 in lib.prange(0, b1 - 1, blksize_v_aab):
                    ba = a1 - a0
                    # R3bba: P0
                    einsum_("cbkd,ijad->ijabkc", mycc.W_vVoV[:, b0:b1], t2bb[i0:i1, j0:j1, a0:a1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=0.0)
                    einsum_("cakd,ijbd->ijabkc", mycc.W_vVoV[:, a0:a1], t2bb[i0:i1, j0:j1, b0:b1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                    # R3bba: P3
                    einsum_("clkj,ilab->ijabkc", mycc.W_vOoO[..., j0:j1], t2bb[i0:i1, :, a0:a1, b0:b1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                    einsum_("clki,jlab->ijabkc", mycc.W_vOoO[..., i0:i1], t2bb[j0:j1, :, a0:a1, b0:b1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)

                    time2 = log.timer_debug1('t3bba: (vVoV + vOoO) * t2bb iter: '
                        '[%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d]:' % (j0, j1, i0, i1, b0, b1, a0, a1), *time2)

                    # R3bba: P1
                    einsum_("cbdj,kdia->ijabkc", mycc.W_vVvO[:, b0:b1, :, j0:j1], t2ab[:, :, i0:i1, a0:a1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                    einsum_("cadj,kdib->ijabkc", mycc.W_vVvO[:, a0:a1, :, j0:j1], t2ab[:, :, i0:i1, b0:b1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                    einsum_("cbdi,kdja->ijabkc", mycc.W_vVvO[:, b0:b1, :, i0:i1], t2ab[:, :, j0:j1, a0:a1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                    einsum_("cadi,kdjb->ijabkc", mycc.W_vVvO[:, a0:a1, :, i0:i1], t2ab[:, :, j0:j1, b0:b1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                    # R3bba: P2
                    einsum_("abdi,kcjd->ijabkc", mycc.W_VVVO[a0:a1, b0:b1, :, i0:i1], t2ab[:, :, j0:j1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-0.5, beta=1.0)
                    einsum_("badi,kcjd->ijabkc", mycc.W_VVVO[b0:b1, a0:a1, :, i0:i1], t2ab[:, :, j0:j1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=0.5, beta=1.0)
                    einsum_("abdj,kcid->ijabkc", mycc.W_VVVO[a0:a1, b0:b1, :, j0:j1], t2ab[:, :, i0:i1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=0.5, beta=1.0)
                    einsum_("badj,kcid->ijabkc", mycc.W_VVVO[b0:b1, a0:a1, :, j0:j1], t2ab[:, :, i0:i1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-0.5, beta=1.0)
                    # R3bba: P4
                    einsum_("lakj,lcib->ijabkc", mycc.W_oVoO[:, a0:a1, :, j0:j1], t2ab[:, :, i0:i1, b0:b1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                    einsum_("lbkj,lcia->ijabkc", mycc.W_oVoO[:, b0:b1, :, j0:j1], t2ab[:, :, i0:i1, a0:a1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                    einsum_("laki,lcjb->ijabkc", mycc.W_oVoO[:, a0:a1, :, i0:i1], t2ab[:, :, j0:j1, b0:b1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                    einsum_("lbki,lcja->ijabkc", mycc.W_oVoO[:, b0:b1, :, i0:i1], t2ab[:, :, j0:j1, a0:a1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                    # R3bba: P5
                    einsum_("laij,kclb->ijabkc", mycc.W_OVOO[:, a0:a1, i0:i1, j0:j1], t2ab[..., b0:b1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=0.5, beta=1.0)
                    einsum_("lbij,kcla->ijabkc", mycc.W_OVOO[:, b0:b1, i0:i1, j0:j1], t2ab[..., a0:a1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-0.5, beta=1.0)
                    einsum_("laji,kclb->ijabkc", mycc.W_OVOO[:, a0:a1, j0:j1, i0:i1], t2ab[..., b0:b1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-0.5, beta=1.0)
                    einsum_("lbji,kcla->ijabkc", mycc.W_OVOO[:, b0:b1, j0:j1, i0:i1], t2ab[..., a0:a1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=0.5, beta=1.0)

                    time2 = log.timer_debug1('t3bba: (vVvO + VVVO + oVoO + OVOO) * t2ab iter: '
                        '[%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d]:' % (j0, j1, i0, i1, b0, b1, a0, a1), *time2)

                    # R3bba: P6 & P8 & P19
                    _unp_bba(mycc, t3bba, t3_tmp, i0, i1, j0, j1, a0, a1, b0, b1)
                    einsum_("cd,ijabkd->ijabkc", mycc.tf_vv, t3_tmp[:bi, :bj, :ba, :bb],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                    einsum_("lk,ijablc->ijabkc", mycc.tf_oo, t3_tmp[:bi, :bj, :ba, :bb],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                    einsum_("clkd,ijabld->ijabkc", mycc.W_voov, t3_tmp[:bi, :bj, :ba, :bb],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)

                    time2 = log.timer_debug1('t3bba: (vv + oo + voov) * t3bba iter: '
                        '[%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d]:' % (j0, j1, i0, i1, b0, b1, a0, a1), *time2)

                    # R3bba: P7 & P11 & P17
                    for d0, d1 in lib.prange(0, nvirb, blksize_v_aab):
                        bd = d1 - d0
                        _unp_bba(mycc, t3bba, t3_tmp, i0, i1, j0, j1, b0, b1, d0, d1)
                        einsum_("ad,ijbdkc->ijabkc", mycc.tf_VV[a0:a1, d0:d1], t3_tmp[:bi, :bj, :bb, :bd],
                            out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                        einsum_("caed,ijbdke->ijabkc", mycc.W_vVvV[:, a0:a1, :, d0:d1], t3_tmp[:bi, :bj, :bb, :bd],
                            out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                        einsum_("lakd,ijbdlc->ijabkc", mycc.W_oVoV[:, a0:a1, :, d0:d1], t3_tmp[:bi, :bj, :bb, :bd],
                            out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                        _unp_bba(mycc,t3bba, t3_tmp, i0, i1, j0, j1, a0, a1, d0, d1)
                        einsum_("bd,ijadkc->ijabkc", mycc.tf_VV[b0:b1, d0:d1], t3_tmp[:bi, :bj, :ba, :bd],
                            out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                        einsum_("cbed,ijadke->ijabkc", mycc.W_vVvV[:, b0:b1, :, d0:d1], t3_tmp[:bi, :bj, :ba, :bd],
                            out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                        einsum_("lbkd,ijadlc->ijabkc", mycc.W_oVoV[:, b0:b1, :, d0:d1], t3_tmp[:bi, :bj, :ba, :bd],
                            out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)

                    time2 = log.timer_debug1('t3bba: (VV + vVvV + oVoV) * t3bba iter: '
                        '[%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d]:' % (j0, j1, i0, i1, b0, b1, a0, a1), *time2)

                    # R3bba: P9 & P13 & P16
                    for l0, l1 in lib.prange(0, noccb, blksize_o_aab):
                        bl = l1 - l0
                        _unp_bba(mycc, t3bba, t3_tmp, l0, l1, j0, j1, a0, a1, b0, b1)
                        einsum_("li,ljabkc->ijabkc", mycc.tf_OO[l0:l1, i0:i1], t3_tmp[:bl, :bj, :ba, :bb],
                            out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                        einsum_("mlki,ljabmc->ijabkc", mycc.W_oOoO[:, l0:l1, :, i0:i1], t3_tmp[:bl, :bj, :ba, :bb],
                            out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                        einsum_("cldi,ljabkd->ijabkc", mycc.W_vOvO_tc[:, l0:l1, :, i0:i1],
                            t3_tmp[:bl, :bj, :ba, :bb], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                        _unp_bba(mycc, t3bba, t3_tmp, l0, l1, i0, i1, a0, a1, b0, b1)
                        einsum_("lj,liabkc->ijabkc", mycc.tf_OO[l0:l1, j0:j1], t3_tmp[:bl, :bi, :ba, :bb],
                            out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                        einsum_("mlkj,liabmc->ijabkc", mycc.W_oOoO[:, l0:l1, :, j0:j1], t3_tmp[:bl, :bi, :ba, :bb],
                            out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                        einsum_("cldj,liabkd->ijabkc", mycc.W_vOvO_tc[:, l0:l1, :, j0:j1],
                            t3_tmp[:bl, :bi, :ba, :bb], out=r3_tmp[:bi, :bj, :ba, :bb, :, :], alpha=1.0, beta=1.0)

                    time2 = log.timer_debug1('t3bba: (oo + oOoO + vOvO) * t3bba iter: '
                        '[%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d]:' % (j0, j1, i0, i1, b0, b1, a0, a1), *time2)

                    # R3bba: P10
                    for d0, d1 in lib.prange(0, nvirb, blksize_v_aab):
                        bd = d1 - d0
                        for e0, e1 in lib.prange(0, nvirb, blksize_v_aab):
                            be = e1 - e0
                            _unp_bba(mycc, t3bba, t3_tmp, i0, i1, j0, j1, d0, d1, e0, e1)
                            einsum_("abde,ijdekc->ijabkc", mycc.W_VVVV[a0:a1, b0:b1, d0:d1, e0:e1],
                                t3_tmp[:bi, :bj, :bd, :be], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=0.5, beta=1.0)

                    time2 = log.timer_debug1('t3bba: VVVV * t3bba iter: '
                        '[%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d]:' % (j0, j1, i0, i1, b0, b1, a0, a1), *time2)

                    # R3bba: P12
                    for l0, l1 in lib.prange(0, noccb, blksize_o_aab):
                        bl = l1 - l0
                        for m0, m1 in lib.prange(0, noccb, blksize_o_aab):
                            bm = m1 - m0
                            _unp_bba(mycc, t3bba, t3_tmp, l0, l1, m0, m1, a0, a1, b0, b1)
                            einsum_("lmij,lmabkc->ijabkc", mycc.W_OOOO[l0:l1, m0:m1, i0:i1, j0:j1],
                                t3_tmp[:bl, :bm, :ba, :bb], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=0.5, beta=1.0)

                    time2 = log.timer_debug1('t3bba: OOOO * t3bba iter: '
                        '[%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d]:' % (j0, j1, i0, i1, b0, b1, a0, a1), *time2)

                    # R3bba: P14
                    for l0, l1 in lib.prange(0, noccb, blksize_o_aab):
                        bl = l1 - l0
                        for d0, d1 in lib.prange(0, nvirb, blksize_v_aab):
                            bd = d1 - d0
                            _unp_bba(mycc, t3bba, t3_tmp, l0, l1, j0, j1, d0, d1, b0, b1)
                            einsum_("alid,ljdbkc->ijabkc", mycc.W_VOOV[a0:a1, l0:l1, i0:i1, d0:d1],
                                t3_tmp[:bl, :bj, :bd, :bb], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                            _unp_bba(mycc, t3bba, t3_tmp, l0, l1, j0, j1, d0, d1, a0, a1)
                            einsum_("blid,ljdakc->ijabkc", mycc.W_VOOV[b0:b1, l0:l1, i0:i1, d0:d1],
                                t3_tmp[:bl, :bj, :bd, :ba], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                            _unp_bba(mycc, t3bba, t3_tmp, l0, l1, i0, i1, d0, d1, b0, b1)
                            einsum_("aljd,lidbkc->ijabkc", mycc.W_VOOV[a0:a1, l0:l1, j0:j1, d0:d1],
                                t3_tmp[:bl, :bi, :bd, :bb], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                            _unp_bba(mycc, t3bba, t3_tmp, l0, l1, i0, i1, d0, d1, a0, a1)
                            einsum_("bljd,lidakc->ijabkc", mycc.W_VOOV[b0:b1, l0:l1, j0:j1, d0:d1],
                                t3_tmp[:bl, :bi, :bd, :ba], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)

                    time2 = log.timer_debug1('t3bba: VOOV * t3bba iter: '
                        '[%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d]:' % (j0, j1, i0, i1, b0, b1, a0, a1), *time2)

                    # R3bba: P15
                    # FIXME: This unpacking step performs redundant operations and should be optimized.
                    for l0, l1 in lib.prange(0, nocca, blksize_o_aab):
                        bl = l1 - l0
                        for d0, d1 in lib.prange(0, nvira, blksize_v_aab):
                            bd = d1 - d0
                            _unp_aab(mycc, t3aab, t3_tmp_2, l0, l1, 0, nocca,
                                    d0, d1, 0, nvira, blk_j=nocca, blk_b=nvira)
                            einsum_("alid,lkdcjb->ijabkc", mycc.W_VoOv[a0:a1, l0:l1, i0:i1, d0:d1],
                                t3_tmp_2[:bl, :, :bd, :, j0:j1, b0:b1],
                                out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                            einsum_("blid,lkdcja->ijabkc", mycc.W_VoOv[b0:b1, l0:l1, i0:i1, d0:d1],
                                t3_tmp_2[:bl, :, :bd, :, j0:j1, a0:a1],
                                out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                            einsum_("aljd,lkdcib->ijabkc", mycc.W_VoOv[a0:a1, l0:l1, j0:j1, d0:d1],
                                t3_tmp_2[:bl, :, :bd, :, i0:i1, b0:b1],
                                out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                            einsum_("bljd,lkdcia->ijabkc", mycc.W_VoOv[b0:b1, l0:l1, j0:j1, d0:d1],
                                t3_tmp_2[:bl, :, :bd, :, i0:i1, a0:a1],
                                out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)

                    time2 = log.timer_debug1('t3bba: VoOv * t3aab iter: '
                        '[%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d]:' % (j0, j1, i0, i1, b0, b1, a0, a1), *time2)

                    # R3bba: P18
                    _unp_bbb(mycc, t3bbb, t3_tmp_3, i0, i1, j0, j1, 0, noccb, a0, a1, b0, b1, 0, nvirb,
                        blk_i=blksize_o_aab, blk_j=blksize_o_aab, blk_k=noccb,
                        blk_a=blksize_v_aab, blk_b=blksize_v_aab, blk_c=nvirb)
                    einsum_("clkd,ijlabd->ijabkc", mycc.W_vOoV, t3_tmp_3[:bi, :bj, :, :ba, :bb, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)

                    time2 = log.timer_debug1('t3bba: vOoV * t3bbb iter: '
                        '[%2d, %2d] [%2d, %2d] [%3d, %3d] [%3d, %3d]:' % (j0, j1, i0, i1, b0, b1, a0, a1), *time2)

                    _update_packed_bba(mycc, r3bba, r3_tmp, i0, i1, j0, j1, a0, a1, b0, b1)
    r3_tmp = None
    t3_tmp = None
    t3_tmp_2 = None
    t3_tmp_3 = None

    time1 = log.timer_debug1('t3: r3bba', *time1)
    return r3bba

def r3_tril_divide_e_uhf(mycc, r3):
    nocca, nvira = mycc.nocca, mycc.nvira
    noccb, nvirb = mycc.noccb, mycc.nvirb
    blksize_o_aaa, blksize_v_aaa = mycc.blksize_o_aaa, mycc.blksize_v_aaa
    blksize_o_aab, blksize_v_aab = mycc.blksize_o_aab, mycc.blksize_v_aab
    eia_a = mycc.mo_energy[0][:nocca, None] - mycc.mo_energy[0][None, nocca:] - mycc.level_shift
    eia_b = mycc.mo_energy[1][:noccb, None] - mycc.mo_energy[1][None, noccb:] - mycc.level_shift

    r3aaa, r3aab, r3bba, r3bbb = r3

    # aaa
    r3_tmp = np.empty((blksize_o_aaa,) * 3 + (blksize_v_aaa,) * 3, dtype=r3aaa.dtype)
    eijkabc_blk = np.empty((blksize_o_aaa,) * 3 + (blksize_v_aaa,) * 3, dtype=r3aaa.dtype)
    for k0, k1 in lib.prange(0, nocca, blksize_o_aaa):
        bk = k1 - k0
        for j0, j1 in lib.prange(0, k1 - 1, blksize_o_aaa):
            bj = j1 - j0
            for i0, i1 in lib.prange(0, j1 - 1, blksize_o_aaa):
                bi = i1 - i0
                for c0, c1 in lib.prange(0, nvira, blksize_v_aaa):
                    bc = c1 - c0
                    for b0, b1 in lib.prange(0, c1 - 1, blksize_v_aaa):
                        bb = b1 - b0
                        for a0, a1 in lib.prange(0, b1 - 1, blksize_v_aaa):
                            ba = a1 - a0
                            _unp_aaa(mycc, r3aaa, r3_tmp, i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, c0, c1)
                            eijkabc_blk = (eia_a[i0:i1, None, None, a0:a1, None, None]
                                        + eia_a[None, j0:j1, None, None, b0:b1, None]
                                        + eia_a[None, None, k0:k1, None, None, c0:c1])
                            r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc] /= eijkabc_blk
                            _update_packed_aaa(mycc, r3aaa, r3_tmp, i0, i1, j0, j1, k0, k1,
                                                a0, a1, b0, b1, c0, c1, alpha=1.0, beta=0.0)
    r3_tmp = None
    eijkabc_blk = None

    # bbb
    r3_tmp = np.empty((blksize_o_aaa,) * 3 + (blksize_v_aaa,) * 3, dtype=r3aaa.dtype)
    eijkabc_blk = np.empty((blksize_o_aaa,) * 3 + (blksize_v_aaa,) * 3, dtype=r3aaa.dtype)
    for k0, k1 in lib.prange(0, noccb, blksize_o_aaa):
        bk = k1 - k0
        for j0, j1 in lib.prange(0, k1 - 1, blksize_o_aaa):
            bj = j1 - j0
            for i0, i1 in lib.prange(0, j1 - 1, blksize_o_aaa):
                bi = i1 - i0
                for c0, c1 in lib.prange(0, nvirb, blksize_v_aaa):
                    bc = c1 - c0
                    for b0, b1 in lib.prange(0, c1 - 1, blksize_v_aaa):
                        bb = b1 - b0
                        for a0, a1 in lib.prange(0, b1 - 1, blksize_v_aaa):
                            ba = a1 - a0
                            _unp_bbb(mycc, r3bbb, r3_tmp, i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, c0, c1)
                            eijkabc_blk = (eia_b[i0:i1, None, None, a0:a1, None, None]
                                        + eia_b[None, j0:j1, None, None, b0:b1, None]
                                        + eia_b[None, None, k0:k1, None, None, c0:c1])
                            r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc] /= eijkabc_blk
                            _update_packed_bbb(mycc, r3bbb, r3_tmp, i0, i1, j0, j1, k0, k1,
                                                a0, a1, b0, b1, c0, c1, alpha=1.0, beta=0.0)
    r3_tmp = None
    eijkabc_blk = None

    # aab
    r3_tmp = np.empty((blksize_o_aab,) * 2 + (blksize_v_aab,) * 2 + (noccb,) + (nvirb,), dtype=r3aaa.dtype)
    eijkabc_blk = np.empty((blksize_o_aab,) * 2 + (blksize_v_aab,) * 2 + (noccb,) + (nvirb,), dtype=r3aaa.dtype)
    for j0, j1 in lib.prange(0, nocca, blksize_o_aab):
        bj = j1 - j0
        for i0, i1 in lib.prange(0, j1 - 1, blksize_o_aab):
            bi = i1 - i0
            for b0, b1 in lib.prange(0, nvira, blksize_v_aab):
                bb = b1 - b0
                for a0, a1 in lib.prange(0, b1 - 1, blksize_v_aab):
                    ba = a1 - a0
                    _unp_aab(mycc, r3aab, r3_tmp, i0, i1, j0, j1, a0, a1, b0, b1)
                    eijkabc_blk = (eia_a[i0:i1, None, a0:a1, None, None, None]
                            + eia_a[None, j0:j1, None, b0:b1, None, None] + eia_b[None, None, None, None, :, :])
                    r3_tmp[:bi, :bj, :ba, :bb] /= eijkabc_blk
                    _update_packed_aab(mycc, r3aab, r3_tmp, i0, i1, j0, j1, a0, a1, b0, b1)
    r3_tmp = None
    eijkabc_blk = None

    # bba
    r3_tmp = np.empty((blksize_o_aab,) * 2 + (blksize_v_aab,) * 2 + (nocca,) + (nvira,), dtype=r3aaa.dtype)
    eijkabc_blk = np.empty((blksize_o_aab,) * 2 + (blksize_v_aab,) * 2 + (nocca,) + (nvira,), dtype=r3aaa.dtype)
    for j0, j1 in lib.prange(0, noccb, blksize_o_aab):
        bj = j1 - j0
        for i0, i1 in lib.prange(0, j1 - 1, blksize_o_aab):
            bi = i1 - i0
            for b0, b1 in lib.prange(0, nvirb, blksize_v_aab):
                bb = b1 - b0
                for a0, a1 in lib.prange(0, b1 - 1, blksize_v_aab):
                    ba = a1 - a0
                    _unp_bba(mycc, r3bba, r3_tmp, i0, i1, j0, j1, a0, a1, b0, b1)
                    eijkabc_blk = (eia_b[i0:i1, None, a0:a1, None, None, None]
                            + eia_b[None, j0:j1, None, b0:b1, None, None] + eia_a[None, None, None, None, :, :])
                    r3_tmp[:bi, :bj, :ba, :bb] /= eijkabc_blk
                    _update_packed_bba(mycc, r3bba, r3_tmp, i0, i1, j0, j1, a0, a1, b0, b1)
    r3_tmp = None
    eijkabc_blk = None

    return (r3aaa, r3aab, r3bba, r3bbb)

def update_amps_t3_tril_uhf(mycc, tamps):
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    _, t2, t3 = tamps
    t3aaa, t3aab, t3bba, t3bbb = t3

    intermediates_t3_uhf(mycc, t2)
    intermediates_t3_add_t3_tril_uhf(mycc, t3)
    time1 = log.timer_debug1('t3: update intermediates', *time0)

    r3aaa = compute_r3aaa_tril_uhf(mycc, t2, t3)
    r3aab = compute_r3aab_tril_uhf(mycc, t2, t3)
    r3bba = compute_r3bba_tril_uhf(mycc, t2, t3)
    r3bbb = compute_r3bbb_tril_uhf(mycc, t2, t3)
    r3 = (r3aaa, r3aab, r3bba, r3bbb)
    time1 = log.timer_debug1('t3: compute r3', *time1)

    # divide by eijkabc
    r3 = r3_tril_divide_e_uhf(mycc, r3)
    r3aaa, r3aab, r3bba, r3bbb = r3
    time1 = log.timer_debug1('t3: divide r3 by eijkabc', *time1)

    mycc.r_norm[2] = np.sqrt(np.linalg.norm(r3aaa)**2 + np.linalg.norm(r3aab)**2
                            + np.linalg.norm(r3bba)**2 + np.linalg.norm(r3bbb)**2)

    t3aaa += r3aaa
    r3aaa = None
    t3bbb += r3bbb
    r3bbb = None
    t3aab += r3aab
    r3aab = None
    t3bba += r3bba
    r3bba = None
    t3 = (t3aaa, t3aab, t3bba, t3bbb)
    time1 = log.timer_debug1('t3: update t3', *time1)
    time0 = log.timer_debug1('t3 total', *time0)
    return t3

def amplitudes_to_vector_uhf(mycc, tamps):
    from math import prod, factorial
    nx = lambda nocc, order: prod(nocc - i for i in range(order)) // factorial(order)

    nocca, noccb = mycc.nocca, mycc.noccb
    nvira, nvirb = mycc.nvira, mycc.nvirb

    tamps_size = [0]
    for i in range(len(tamps)):
        for na in range(i + 1, -1, -1):
            nb = i + 1 - na
            if na >= nb:
                tamps_size.append(nx(nocca, na) * nx(nvira, na) * nx(noccb, nb) * nx(nvirb, nb))
            else:
                tamps_size.append(nx(noccb, nb) * nx(nvirb, nb) * nx(nocca, na) * nx(nvira, na))
    cum_sizes = np.cumsum(tamps_size)
    vector = np.zeros(cum_sizes[-1], dtype=tamps[0][0].dtype)
    st = 0
    for i, t in enumerate(tamps):
        for j in range(i + 2):
            idx = mycc.unique_tamps_map[i][j]
            vector[cum_sizes[st] : cum_sizes[st + 1]] = t[j][idx].ravel()
            st += 1
    return vector

def vector_to_amplitudes_uhf(mycc, vector):
    from math import prod, factorial
    nx = lambda nocc, order: prod(nocc - i for i in range(order)) // factorial(order)

    nocca, noccb = mycc.nocca, mycc.noccb
    nvira, nvirb = mycc.nvira, mycc.nvirb

    tamps_size = [0]
    for i in range(mycc.cc_order):
        for na in range(i + 1, -1, -1):
            nb = i + 1 - na
            if na >= nb:
                tamps_size.append(nx(nocca, na) * nx(nvira, na) * nx(noccb, nb) * nx(nvirb, nb))
            else:
                tamps_size.append(nx(noccb, nb) * nx(nvirb, nb) * nx(nocca, na) * nx(nvira, na))
    cum_sizes = np.cumsum(tamps_size)

    try:
        endpoint = cum_sizes.tolist().index(vector.shape[0])
    except ValueError:
        raise ValueError("Mismatch between vector size and tamps size")

    st = 0
    tamps = []
    for i in range(mycc.cc_order):
        if st == endpoint:
            break
        tamps_ = []
        for na in range(i + 1, -1, -1):
            nb = i + 1 - na
            if na >= nb:
                nocc1, nocc2 = nocca, noccb
                nvir1, nvir2 = nvira, nvirb
                n1, n2 = na, nb
            else:
                nocc1, nocc2 = noccb, nocca
                nvir1, nvir2 = nvirb, nvira
                n1, n2 = nb, na

            if mycc.do_tril[i]:
                if n2 == 0:
                    xshape = (nx(nocc1, n1),) + (nx(nvir1, n1),)
                else:
                    xshape = (nx(nocc1, n1),) + (nx(nvir1, n1),) + (nx(nocc2, n2),) + (nx(nvir2, n2),)
                t = np.zeros(xshape, dtype=vector.dtype)
            else:
                t = np.zeros((nocc1,) * n1 + (nvir1,) * n1 + (nocc2,) * n2 + (nvir2,) * n2, dtype=vector.dtype)

            idx = mycc.unique_tamps_map[i][nb]
            t[idx] = vector[cum_sizes[st] : cum_sizes[st + 1]].reshape(t[idx].shape)
            t = restore_t_uhf(t, order=i + 1, pos=nb, do_tril=mycc.do_tril[i])
            tamps_.append(t)
            st += 1
        tamps.append(tamps_)

    return tamps

def restore_t_uhf(t, order=1, pos=0, do_tril=False):
    import itertools
    def permutation_sign(p):
        inv_count = sum(p[i] > p[j] for i in range(len(p)) for j in range(i + 1, len(p)))
        return 1 if inv_count % 2 == 0 else -1

    na, nb = order - pos, pos
    if do_tril:
        return t
    else:
        tt = np.zeros_like(t)
        if na >= nb:
            n1, n2 = na, nb
        else:
            n1, n2 = nb, na

        if n1 >= 2:
            perms = list(itertools.permutations(range(n1)))
            for idx, perm in enumerate(perms):
                sign = permutation_sign(perm)
                msg = (*perm, *range(n1, 2 * n1), *range(2 * n1, 2 * n1 + 2 * n2))
                tt += sign * t.transpose(msg)
            t[:] = 0.0
            for idx, perm in enumerate(perms):
                sign = permutation_sign(perm)
                msg = (*range(n1), *[p + n1 for p in perm], *range(2 * n1, 2 * (n1 + n2)))
                t += sign * tt.transpose(msg)
        if n2 >= 2:
            perms = list(itertools.permutations(range(n2)))
            for idx, perm in enumerate(perms):
                sign = permutation_sign(perm)
                msg = (*range(2 * n1), *[p + 2 * n1 for p in perm], *range(2 * n1 + n2, 2 * (n1 + n2)))
                tt += sign * t.transpose(msg)
            t[:] = 0.0
            for idx, perm in enumerate(perms):
                sign = permutation_sign(perm)
                msg = (*range(2 * n1), *range(2 * n1, 2 * n1 + n2), *[p + 2 * n1 + n2 for p in perm])
                t += sign * tt.transpose(msg)
        return t

def kernel(mycc, eris=None, t1=None, t2=None, t3=None, tol=1e-8, tolnormt=1e-6, max_cycle=50,
        verbose=None, callback=None, diis_with_t3=False, num_of_subiters=1):
    log = logger.new_logger(mycc, verbose)

    nocca, nocca2, nocca3 = mycc.nocca, mycc.nocca2, mycc.nocca3
    noccb, noccb2, noccb3 = mycc.noccb, mycc.noccb2, mycc.noccb3
    nvira, nvira2, nvira3 = mycc.nvira, mycc.nvira2, mycc.nvira3
    nvirb, nvirb2, nvirb3 = mycc.nvirb, mycc.nvirb2, mycc.nvirb3
    ccdtype = mycc.mo_coeff.dtype

    if eris is None:
        eris = ao2mo_uccsdt(mycc, mycc.mo_coeff)
    if t3 is None:
        shape = ((nocca3 if isinstance(nocca3, tuple) else (nocca3,))
                + (nvira3 if isinstance(nvira3, tuple) else (nvira3,)))
        t3aaa = np.zeros(shape, dtype=ccdtype)
        shape = ((nocca2 if isinstance(nocca2, tuple) else (nocca2,))
                + (nvira2 if isinstance(nvira2, tuple) else (nvira2,)) + (noccb, nvirb))
        t3aab = np.zeros(shape, dtype=ccdtype)
        shape = ((noccb2 if isinstance(noccb2, tuple) else (noccb2,))
                + (nvirb2 if isinstance(nvirb2, tuple) else (nvirb2,)) + (nocca, nvira))
        t3bba = np.zeros(shape, dtype=ccdtype)
        shape = ((noccb3 if isinstance(noccb3, tuple) else (noccb3,))
                + (nvirb3 if isinstance(nvirb3, tuple) else (nvirb3,)))
        t3bbb = np.zeros(shape, dtype=ccdtype)
        t3 = (t3aaa, t3aab, t3bba, t3bbb)
    else:
        t3aaa, t3aab, t3bba, t3bbb = t3
        t3aaa = np.asarray(t3aaa, dtype=ccdtype)
        t3aab = np.asarray(t3aab, dtype=ccdtype)
        t3bba = np.asarray(t3bba, dtype=ccdtype)
        t3bbb = np.asarray(t3bbb, dtype=ccdtype)
        t3 = (t3aaa, t3aab, t3bba, t3bbb)
    if t1 is None and t2 is None:
        t1, t2 = mycc.init_amps()[1:3]
    elif t1 is not None and t2 is not None:
        t1a, t1b = t1
        t1a, t1b = np.asarray(t1a), np.asarray(t1b)
        t1 = (t1a, t1b)
        t2aa, t2ab, t2bb = t2
        t2aa = np.asarray(t2aa, dtype=ccdtype)
        t2ab = np.asarray(t2ab, dtype=ccdtype)
        t2bb = np.asarray(t2bb, dtype=ccdtype)
        t2 = (t2aa, t2ab, t2bb)
    else:
        ValueError('Input tamps do not satisfy the expected conditions')

    name = mycc.__class__.__name__
    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    e_corr_old = 0.0
    e_corr = mycc.energy(t1, t2)
    log.info('Init E_corr(%s) = %.15g', name, e_corr)

    if isinstance(mycc.diis, lib.diis.DIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
        adiis.space = mycc.diis_space
    else:
        adiis = None

    converged = False
    mycc.cycles = 0
    for istep in range(max_cycle):
        for i in range(num_of_subiters):
            t1, t2 = mycc.update_amps_t1t2_with_t3_uhf((t1, t2, t3))
        t3 = mycc.update_amps_t3_uhf((t1, t2, t3))

        # NOTE: What does this stand for?
        if callback is not None:
            callback(locals())

        normt = np.linalg.norm(mycc.r_norm)

        if mycc.iterative_damping < 1.0:
            raise NotImplementedError("Damping is not implemented")

        if diis_with_t3:
            (t1, t2, t3) = mycc.run_diis((t1, t2, t3), istep, normt, mycc.e_corr - e_corr_old, adiis)
        else:
            (t1, t2) = mycc.run_diis((t1, t2), istep, normt, mycc.e_corr - e_corr_old, adiis)

        e_corr_old, e_corr = e_corr, mycc.energy(t1, t2)
        mycc.e_corr_ss = getattr(e_corr, 'e_corr_ss', 0)
        mycc.e_corr_os = getattr(e_corr, 'e_corr_os', 0)

        mycc.cycles = istep + 1
        log.info("cycle = %2d  E_corr(UCCSDT) = % .12f  dE = % .12e  norm(t1,t2,t3) = %.8e" % (
            istep + 1, e_corr, e_corr - e_corr_old, normt))
        cput1 = log.timer(f'{name} iter', *cput1)

        if abs(e_corr - e_corr_old) < tol and normt < tolnormt:
            converged = True
            break
    log.timer(name, *cput0)
    return converged, e_corr, t1, t2, t3

def ao2mo_uccsdt(mycc, mo_coeff=None):
    if mycc._scf._eri is not None:
        logger.note(mycc, '_make_eris_incore_' + mycc.__class__.__name__)
        return _make_eris_incore_uccsdt(mycc, mo_coeff)
    elif getattr(mycc._scf, 'with_df', None):
        logger.note(mycc, '_make_df_eris_incore_' + mycc.__class__.__name__)
        return _make_df_eris_incore_uccsdt(mycc, mo_coeff)

def restore_from_diis_(mycc, diis_file, inplace=True, diis_with_t3=True):
    nocca, nocca2, nocca3 = mycc.nocca, mycc.nocca2, mycc.nocca3
    noccb, noccb2, noccb3 = mycc.noccb, mycc.noccb2, mycc.noccb3
    nvira, nvira2, nvira3 = mycc.nvira, mycc.nvira2, mycc.nvira3
    nvirb, nvirb2, nvirb3 = mycc.nvirb, mycc.nvirb2, mycc.nvirb3

    adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
    adiis.restore(diis_file, inplace=inplace)

    ccvec = adiis.extrapolate()
    tamps = mycc.vector_to_amplitudes(ccvec)
    if diis_with_t3:
        t1, t2, t3 = tamps
        mycc.t1a, mycc.t1b = t1
        mycc.t2aa, mycc.t2ab, mycc.t2bb = t2
        mycc.t3aaa, mycc.t3aab, mycc.t3bba, mycc.t3bbb = t3
    else:
        t1, t2 = tamps
        mycc.t1a, mycc.t1b = t1
        mycc.t2aa, mycc.t2ab, mycc.t2bb = t2
        shape = ((nocca3 if isinstance(nocca3, tuple) else (nocca3,))
                + (nvira3 if isinstance(nvira3, tuple) else (nvira3,)))
        mycc.t3aaa = np.zeros(shape, dtype=ccvec.dtype)
        shape = ((nocca2 if isinstance(nocca2, tuple) else (nocca2,))
                + (nvira2 if isinstance(nvira2, tuple) else (nvira2,)) + (noccb, nvirb))
        mycc.t3aab = np.zeros(shape, dtype=ccvec.dtype)
        shape = ((noccb2 if isinstance(noccb2, tuple) else (noccb2,))
                + (nvirb2 if isinstance(nvirb2, tuple) else (nvirb2,)) + (nocca, nvira))
        mycc.t3bba = np.zeros(shape, dtype=ccvec.dtype)
        shape = ((noccb3 if isinstance(noccb3, tuple) else (noccb3,))
                + (nvirb3 if isinstance(nvirb3, tuple) else (nvirb3,)))
        mycc.t3bbb = np.zeros(shape, dtype=ccvec.dtype)
    if inplace:
        mycc.diis = adiis
    return mycc


class UCCSDT(ccsd.CCSDBase):

    # conv_tol = getattr(__config__, 'cc_uccsdt_UCCSDT_conv_tol', 1e-7)
    # conv_tol_normt = getattr(__config__, 'cc_uccsdt_UCCSDT_conv_tol_normt', 1e-6)

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        # ccsd.CCSDBase.__init__(self, mf, frozen, mo_coeff, mo_occ)
        super().__init__(mf, frozen, mo_coeff, mo_occ)

        self.cc_order = 3
        self.do_tril = [False, False, True]

        if mo_coeff is None: mo_coeff = mf.mo_coeff
        if mo_occ is None: mo_occ = mf.mo_occ

        self.t3 = None
        self.diis_with_t3 = True
        self.num_of_subiters = 1

        self.blksize_o_aaa = 8
        self.blksize_v_aaa = 64
        self.blksize_o_aab = 8
        self.blksize_v_aab = 64

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask
    get_e_hf = get_e_hf
    ao2mo = ao2mo_uccsdt
    energy = energy
    init_amps = init_amps
    update_amps_t1t2_with_t3_uhf = update_amps_t1t2_with_t3_tril_uhf
    update_amps_t3_uhf = update_amps_t3_tril_uhf
    amplitudes_to_vector = amplitudes_to_vector_uhf
    vector_to_amplitudes = vector_to_amplitudes_uhf
    run_diis = run_diis
    _finalize = _finalize

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        # log.info('CC2 = %g', self.cc2)
        log.info('UCCSDT nocc = %s, nmo = %s', self.nocc, self.nmo)
        if self.do_tril[-1]:
            log.info("Allocating only the unique part of the T3 amplitudes in memory")
        else:
            log.info("Allocating the entire T3 amplitudes in memory")
        if self.frozen is not None:
            log.info('frozen orbitals %s', self.frozen)
        log.info('max_cycle = %d', self.max_cycle)
        log.info('direct = %d', self.direct)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('conv_tol_normt = %s', self.conv_tol_normt)
        log.info('diis_space = %d', self.diis_space)
        #log.info('diis_file = %s', self.diis_file)
        log.info('diis_start_cycle = %d', self.diis_start_cycle)
        log.info('diis_start_energy_diff = %g', self.diis_start_energy_diff)
        log.info('max_memory %d MB (current use %d MB)', self.max_memory, lib.current_memory()[0])
        return self

    def kernel(self, t1=None, t2=None, t3=None, eris=None):
        return self.ccsdt(t1, t2, t3, eris)

    def ccsdt(self, t1=None, t2=None, t3=None, eris=None):
        log = logger.Logger(self.stdout, self.verbose)

        assert (self.mo_coeff is not None)
        assert (self.mo_occ is not None)

        if self.verbose >= logger.WARN:
            # NOTE: What's this?
            self.check_sanity()
        self.dump_flags()

        self.e_hf = self.get_e_hf()

        if eris is None:
            eris = self.ao2mo(self.mo_coeff)

        ccdtype = self.mo_coeff.dtype
        # (pq|rs) -> <pr|qs>
        self.erisaa = np.asarray(eris.pqrs.transpose(0, 2, 1, 3), dtype=ccdtype)
        self.erisbb = np.asarray(eris.PQRS.transpose(0, 2, 1, 3), dtype=ccdtype)
        self.erisab = np.asarray(eris.pqRS.transpose(0, 2, 1, 3), dtype=ccdtype)
        self.eris_ovov = np.asarray(eris.ovov, dtype=ccdtype)
        self.eris_OVOV = np.asarray(eris.OVOV, dtype=ccdtype)
        self.eris_ovOV = np.asarray(eris.ovOV, dtype=ccdtype)
        if not self.erisaa.flags['C_CONTIGUOUS']:
            self.erisaa = np.ascontiguousarray(self.erisaa)
        if not self.erisbb.flags['C_CONTIGUOUS']:
            self.erisbb = np.ascontiguousarray(self.erisbb)
        if not self.erisab.flags['C_CONTIGUOUS']:
            self.erisab = np.ascontiguousarray(self.erisab)

        self.focka = np.asarray(eris.fock[0], dtype=ccdtype)
        self.fockb = np.asarray(eris.fock[1], dtype=ccdtype)

        self.mo_energy = np.asarray(eris.mo_energy, dtype=ccdtype)
        self.mo_coeff = np.asarray(eris.mo_coeff, dtype=ccdtype)

        nocca, noccb = self.nocca, self.noccb = eris.nocc[0], eris.nocc[1]
        nmoa, nmob = self.nmoa, self.nmob = eris.fock[0].shape[0], eris.fock[1].shape[0]
        nvira, nvirb = self.nvira, self.nvirb = nmoa - nocca, nmob - noccb
        log.info('nocca %5d    nvira %5d    nmoa %5d'%(nocca, nvira, nmoa))
        log.info('noccb %5d    nvirb %5d    nmob %5d'%(noccb, nvirb, nmob))

        if self.do_tril[-1]:
            nocca3 = self.nocca3 = nocca * (nocca - 1) * (nocca - 2) // 6
            nvira3 = self.nvira3 = nvira * (nvira - 1) * (nvira - 2) // 6
            noccb3 = self.noccb3 = noccb * (noccb - 1) * (noccb - 2) // 6
            nvirb3 = self.nvirb3 = nvirb * (nvirb - 1) * (nvirb - 2) // 6
            nocca2 = self.nocca2 = nocca * (nocca - 1) // 2
            nvira2 = self.nvira2 = nvira * (nvira - 1) // 2
            noccb2 = self.noccb2 = noccb * (noccb - 1) // 2
            nvirb2 = self.nvirb2 = nvirb * (nvirb - 1) // 2
        else:
            nocca3 = self.nocca3 = (nocca,) * 3
            nvira3 = self.nvira3 = (nvira,) * 3
            noccb3 = self.noccb3 = (noccb,) * 3
            nvirb3 = self.nvirb3 = (nvirb,) * 3
            nocca2 = self.nocca2 = (nocca,) * 2
            nvira2 = self.nvira2 = (nvira,) * 2
            noccb2 = self.noccb2 = (noccb,) * 2
            nvirb2 = self.nvirb2 = (nvirb,) * 2

        # estimate the memory cost
        if self.do_tril[-1]:
            t3_memory = (nocca3 * nvira3 + nocca2 * nvira2 * noccb * nvirb +
                        nocca * nvira * noccb2 * nvirb2 + noccb3 * nvirb3) * 8 / 1024**2
        else:
            t3_memory = (nocca**3 * nvira**3 + nocca**2 * nvira**2 * noccb * nvirb +
                        nocca * nvira * noccb**2 * nvirb**2 + noccb**3 * nvirb**3) * 8 / 1024**2
        log.info('T3 memory             %8.5e MB' % (t3_memory))
        eris_memory = (nmoa**4 + nmoa**2 * nmob**2 + nmob**4) * 8 / 1024**2
        log.info('eris memory           %8.5e MB' % (eris_memory))
        if self.diis_with_t3:
            diis_memory = ((nocca * (nocca - 1) * (nocca - 2) // 6 * nvira * (nvira - 1) * (nvira - 2) // 6 +
                            nocca * (nocca - 1)// 2 * nvira * (nvira - 1) // 2 * noccb * nvirb +
                            nocca * nvira * noccb * (noccb - 1) // 2 * nvirb * (nvirb - 1) // 2 +
                            noccb * (noccb - 1) * (noccb - 2) // 6 * nvirb * (nvirb - 1) * (nvirb - 2) // 6)
                            * 8 / 1024**2 * self.diis_space * 2)
        else:
            diis_memory = (nocca * (nocca - 1) // 2 * nvira * (nvira - 1) // 2
                        + nocca * nvira * noccb * nvirb
                        + noccb * (nvirb - 1) // 2 * nvirb * (nvirb - 1) // 2) * 8 / 1024**2 * self.diis_space * 2
        log.info('diis memory           %8.5e MB' % (diis_memory))
        if self.do_tril[-1]:
            total_memory = 2 * t3_memory + 3 * eris_memory + diis_memory
        else:
            total_memory = 3 * t3_memory + 3 * eris_memory + diis_memory
        log.info('total estimate memory %8.5e MB' % (total_memory))
        max_memory = self.max_memory - lib.current_memory()[0]
        if total_memory > max_memory:
            logger.warn(self, 'There may not be enough memory for the %s calculation' % self.__class__.__name__)

        self.r_norm = np.zeros(self.cc_order, dtype=ccdtype)

        self.unique_tamps_map = []
        # t1
        t1a_idx = (slice(None), slice(None))
        t1b_idx = (slice(None), slice(None))
        self.unique_tamps_map.append([t1a_idx, t1b_idx])
        # t2
        def compute_t2_idx_uhf(nocc, nvir):
            i_idx, j_idx = np.triu_indices(nocc, k=1)
            a_idx, b_idx = np.triu_indices(nvir, k=1)
            I = np.repeat(i_idx, len(a_idx))
            J = np.repeat(j_idx, len(a_idx))
            A = np.tile(a_idx, len(i_idx))
            B = np.tile(b_idx, len(i_idx))
            t2_idx = (I, J, A, B)
            return t2_idx

        t2aa_idx = compute_t2_idx_uhf(nocca, nvira)
        t2ab_idx = (slice(None), slice(None), slice(None), slice(None))
        t2bb_idx = compute_t2_idx_uhf(noccb, nvirb)
        self.unique_tamps_map.append([t2aa_idx, t2ab_idx, t2bb_idx])
        # t3
        if self.diis_with_t3:
            if self.do_tril[-1]:
                t3aaa_idx = (slice(None), slice(None))
                t3aab_idx = (slice(None), slice(None), slice(None), slice(None))
                t3bba_idx = (slice(None), slice(None), slice(None), slice(None))
                t3bbb_idx = (slice(None), slice(None))
                self.unique_tamps_map.append([t3aaa_idx, t3aab_idx, t3bba_idx, t3bbb_idx])
            else:
                def build_t3aaa_indices_uhf(nocc, nvir):
                    ii, jj, kk = np.meshgrid(np.arange(nocc), np.arange(nocc), np.arange(nocc), indexing='ij')
                    i_idx, j_idx, k_idx = np.where((ii < jj) & (jj < kk))
                    aa, bb, cc = np.meshgrid(np.arange(nvir), np.arange(nvir), np.arange(nvir), indexing='ij')
                    a_idx, b_idx, c_idx = np.where((aa < bb) & (bb < cc))
                    I = np.repeat(i_idx, len(a_idx))
                    J = np.repeat(j_idx, len(a_idx))
                    K = np.repeat(k_idx, len(a_idx))
                    A = np.tile(a_idx, len(i_idx))
                    B = np.tile(b_idx, len(i_idx))
                    C = np.tile(c_idx, len(i_idx))
                    t3aaa_idx = (I, J, K, A, B, C)
                    return t3aaa_idx

                def build_t3aab_indices_uhf(nocca, nvira, noccb, nvirb):
                    i_idx, j_idx = np.triu_indices(nocca, k=1)
                    a_idx, b_idx = np.triu_indices(nvira, k=1)
                    I = np.repeat(i_idx, len(a_idx))
                    J = np.repeat(j_idx, len(a_idx))
                    A = np.tile(a_idx, len(i_idx))
                    B = np.tile(b_idx, len(i_idx))
                    t3aab_idx = (I, J, A, B, slice(None), slice(None))
                    return t3aab_idx

                t3aaa_idx = build_t3aaa_indices_uhf(nocca, nvira)
                t3aab_idx = build_t3aab_indices_uhf(nocca, nvira, noccb, nvirb)
                t3bba_idx = build_t3aab_indices_uhf(noccb, nvirb, nocca, nvira)
                t3bbb_idx = build_t3aaa_indices_uhf(noccb, nvirb)
                self.unique_tamps_map.append([t3aaa_idx, t3aab_idx, t3bba_idx, t3bbb_idx])

        if self.do_tril[-1]:
            setup_tril2cube_t3_uhf(self)

            # setup the blksize for (un)packing and contraction
            # FIXME
            self.blksize_o_aaa = min(self.blksize_o_aaa, max(nocca, noccb))
            self.blksize_v_aaa = min(self.blksize_v_aaa, max((nvira + 1) // 2, (nvirb + 1) // 2))
            self.blksize_o_aab = min(self.blksize_o_aab, max(nocca, noccb))
            self.blksize_v_aab = min(self.blksize_v_aab, max((nvira + 1) // 2, (nvirb + 1) // 2))
            log.info('blksize_o_aaa %5d    blksize_v_aaa %5d'%(self.blksize_o_aaa, self.blksize_v_aaa))
            log.info('blksize_o_aab %5d    blksize_v_aab %5d'%(self.blksize_o_aab, self.blksize_v_aab))

        self.converged, self.e_corr, self.t1, self.t2, self.t3 = \
                kernel(self, eris, t1, t2, t3, max_cycle=self.max_cycle,
                       tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                       verbose=self.verbose, callback=self.callback,
                       diis_with_t3=self.diis_with_t3, num_of_subiters=self.num_of_subiters)
        self._finalize()
        return self.e_corr, self.t1, self.t2, self.t3

    def ccsdt_q(self, t1=None, t2=None, t3=None, eris=None):
        raise NotImplementedError

    def ipccsdt(self, nroots=1, left=False, koopmans=False, guess=None, partition=None, eris=None):
        raise NotImplementedError

    def eaccsdt(self, nroots=1, left=False, koopmans=False, guess=None, partition=None, eris=None):
        raise NotImplementedError

    def eeccsdt(self, nroots=1, koopmans=False, guess=None, eris=None):
        raise NotImplementedError

    def eomee_ccsdt_singlet(self, nroots=1, koopmans=False, guess=None, eris=None):
        raise NotImplementedError

    def eomee_ccsdt_triplet(self, nroots=1, koopmans=False, guess=None, eris=None):
        raise NotImplementedError

    def eomsf_ccsdt(self, nroots=1, koopmans=False, guess=None, eris=None):
        raise NotImplementedError

    def spin_square(self, mo_coeff=None, s=None):
        raise NotImplementedError


class _ChemistsERIs:
    '''(pq|rs)'''
    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nocc = None
        self.fock = None

    def _common_init_(self, mycc, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        mo_idx = mycc.get_frozen_mask()
        self.mo_coeff = mo_coeff = (mo_coeff[0][:, mo_idx[0]], mo_coeff[1][:, mo_idx[1]])

        dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        vhf = mycc._scf.get_veff(mycc.mol, dm)
        fockao = mycc._scf.get_fock(vhf=vhf, dm=dm)
        self.focka = reduce(np.dot, (mo_coeff[0].conj().T, fockao[0], mo_coeff[0]))
        self.fockb = reduce(np.dot, (mo_coeff[1].conj().T, fockao[1], mo_coeff[1]))
        self.fock = (self.focka, self.fockb)

        nocca, noccb = self.nocc = mycc.nocc
        self.mol = mycc.mol

        mo_ea = self.focka.diagonal().real
        mo_eb = self.fockb.diagonal().real
        self.mo_energy = (mo_ea, mo_eb)
        gap_a = abs(mo_ea[:nocca, None] - mo_ea[None, nocca:])
        gap_b = abs(mo_eb[:noccb, None] - mo_eb[None, noccb:])
        if gap_a.size > 0:
            gap_a = gap_a.min()
        else:
            gap_a = 1e9
        if gap_b.size > 0:
            gap_b = gap_b.min()
        else:
            gap_b = 1e9
        if gap_a < 1e-5 or gap_b < 1e-5:
            logger.warn(mycc, 'HOMO-LUMO gap (%s,%s) too small for UCCSD', gap_a, gap_b)
        return self

def _make_eris_incore_uccsdt(mycc, mo_coeff=None, ao2mofn=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo

    moa = eris.mo_coeff[0]
    mob = eris.mo_coeff[1]
    nmoa = moa.shape[1]
    nmob = mob.shape[1]

    if callable(ao2mofn):
        eri_aa = ao2mofn(moa).reshape([nmoa]*4)
        eri_bb = ao2mofn(mob).reshape([nmob]*4)
        eri_ab = ao2mofn((moa, moa, mob, mob))
    else:
        eri_aa = ao2mo.restore(1, ao2mo.full(mycc._scf._eri, moa), nmoa)
        eri_bb = ao2mo.restore(1, ao2mo.full(mycc._scf._eri, mob), nmob)
        eri_ab = ao2mo.general(mycc._scf._eri, (moa, moa, mob, mob), compact=False)
    eri_ba = eri_ab.reshape(nmoa, nmoa, nmob, nmob).transpose(2, 3, 0, 1)

    eri_aa = eri_aa.reshape(nmoa, nmoa, nmoa, nmoa)
    eri_ab = eri_ab.reshape(nmoa, nmoa, nmob, nmob)
    eri_ba = eri_ba.reshape(nmob, nmob, nmoa, nmoa)
    eri_bb = eri_bb.reshape(nmob, nmob, nmob, nmob)

    eris.pqrs = eri_aa
    eris.PQRS = eri_bb
    eris.pqRS = eri_ab
    eris.PQrs = eri_ba
    eris.ovov = eris.pqrs[:nocca, nocca:, :nocca, nocca:].copy()
    eris.OVOV = eris.PQRS[:noccb, noccb:, :noccb, noccb:].copy()
    eris.ovOV = eris.pqRS[:nocca, nocca:, :noccb, noccb:].copy()
    eris.OVov = eris.PQrs[:noccb, noccb:, :nocca, nocca:].copy()

    logger.timer(mycc, mycc.__class__.__name__ + ' integral transformation', *cput0)
    return eris

def _make_df_eris_incore_uccsdt(mycc, mo_coeff=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    moa, mob = eris.mo_coeff
    nocca, noccb = eris.nocc
    nao = moa.shape[0]
    nmoa = moa.shape[1]
    nmob = mob.shape[1]
    naux = mycc._scf.with_df.get_naoaux()

    # --- Three-center integrals
    # (L|aa)
    Lpq = numpy.empty((naux, nmoa, nmoa))
    # (L|bb)
    LPQ = numpy.empty((naux, nmob, nmob))
    p1 = 0
    # Transform three-center integrals to MO basis
    einsum = lib.einsum
    Lpq_tmp = None
    for eri1 in mycc._scf.with_df.loop():
        eri1 = lib.unpack_tril(eri1).reshape(-1, nao, nao)
        # (L|aa)
        Lpq_tmp = einsum('Lab,ap,bq->Lpq', eri1, moa, moa)
        p0, p1 = p1, p1 + Lpq_tmp.shape[0]
        Lpq[p0:p1, :, :] = Lpq_tmp[:, :, :]
        Lpq_tmp = None
        # (L|bb)
        Lpq_tmp = einsum('Lab,ap,bq->Lpq', eri1, mob, mob)
        LPQ[p0:p1, :, :] = Lpq_tmp[:, :, :]
        Lpq_tmp = None
    Lpq = Lpq.reshape(naux, nmoa * nmoa)
    LPQ = LPQ.reshape(naux, nmob * nmob)

    # --- Four-center integrals
    # (aa|aa)
    eris.pqrs = lib.ddot(Lpq.T, Lpq).reshape(nmoa, nmoa, nmoa, nmoa)
    eris.ovov = eris.pqrs[:nocca, nocca:, :nocca, nocca:].copy()
    # (bb|bb)
    eris.PQRS = lib.ddot(LPQ.T, LPQ).reshape(nmob, nmob, nmob, nmob)
    eris.OVOV = eris.pqrs[:noccb, noccb:, :noccb, noccb:].copy()
    # (aa|bb)
    eris.pqRS = lib.ddot(Lpq.T, LPQ).reshape(nmoa, nmoa, nmob, nmob)
    eris.ovOV = eris.pqrs[:nocca, nocca:, :noccb, noccb:].copy()
    # (bb|aa)
    eris.PQrs = lib.ddot(LPQ.T, Lpq).reshape(nmob, nmob, nmoa, nmoa)
    eris.OVov = eris.pqrs[:noccb, noccb:, :nocca, nocca:].copy()

    logger.timer(mycc, mycc.__class__.__name__ + ' integral transformation', *cput0)
    return eris


if __name__ == "__main__":

    from pyscf import gto, scf, df

    test_cases = {
        # 'Li_spin3': {'atom': "Li 0.0 0.0 0.0", 'basis': 'ccpvdz', 'spin': 3, 'ref_ecorr': -0.002851158707014802},
        # 'Li_spinm3': {'atom': "Li 0.0 0.0 0.0", 'basis': 'ccpvdz', 'spin': -3, 'ref_ecorr': -0.002851158707014802},
        # 'Li_spin1': {'atom': "Li 0.0 0.0 0.0", 'basis': 'ccpvdz', 'spin': 1, 'ref_ecorr': -0.0002169873693235238},
        # 'Li_spinm1': {'atom': "Li 0.0 0.0 0.0", 'basis': 'ccpvdz', 'spin': -1, 'ref_ecorr': -0.0002169873693235238},
        # 'Be_spin4': {'atom': "Be 0.0 0.0 0.0", 'basis': 'ccpvdz', 'spin': 4, 'ref_ecorr': -0.00798895384046908},
        # 'Be_spinm4': {'atom': "Be 0.0 0.0 0.0", 'basis': 'ccpvdz', 'spin': -4, 'ref_ecorr': -0.00798895384046908},
        # 'Be_spin2': {'atom': "Be 0.0 0.0 0.0", 'basis': 'ccpvdz', 'spin': 2, 'ref_ecorr': -0.004375255089139482},
        # 'Be_spinm2': {'atom': "Be 0.0 0.0 0.0", 'basis': 'ccpvdz', 'spin': -2, 'ref_ecorr': -0.004375255089139482},
        # 'Be_spin0': {'atom': "Be 0.0 0.0 0.0", 'basis': 'ccpvdz', 'spin': 0, 'ref_ecorr': -0.04507000858611861},
        # 'B_spin5': {'atom': "B 0.0 0.0 0.0", 'basis': 'ccpvdz', 'spin': 5, 'ref_ecorr': -0.01477294876671822},
        # 'B_spinm5': {'atom': "B 0.0 0.0 0.0", 'basis': 'ccpvdz', 'spin': -5, 'ref_ecorr': -0.01477294876671822},
        # 'B_spin3': {'atom': "B 0.0 0.0 0.0", 'basis': 'ccpvdz', 'spin': 3, 'ref_ecorr': -0.01325814002194714},
        # 'B_spinm3': {'atom': "B 0.0 0.0 0.0", 'basis': 'ccpvdz', 'spin': -3, 'ref_ecorr': -0.01325814002194714},
        # 'B_spin1': {'atom': "B 0.0 0.0 0.0", 'basis': 'ccpvdz', 'spin': 1, 'ref_ecorr': -0.06066524073415023},
        # 'B_spinm1': {'atom': "B 0.0 0.0 0.0", 'basis': 'ccpvdz', 'spin': -1, 'ref_ecorr': -0.06066524073415023},
        # 'O2': {'atom': "O 0 0 0; O 0 0 1.21", 'basis': 'sto3g', 'spin': 2, 'ref_ecorr': -0.1095053789680588},
        'O2_631g': {'atom': "O 0 0 0; O 0 0 1.21", 'basis': '631g', 'spin': 2, 'ref_ecorr': -0.2413923134881862},
    }

    run_test = {
        # 'Li_spin3': False,
        # 'Li_spinm3': False,
        # 'Li_spin1': True,
        # 'Li_spinm1': True,
        # 'Be_spin4': False,
        # 'Be_spinm4': False,
        # 'Be_spin2': True,
        # 'Be_spinm2': True,
        # 'Be_spin0': True,
        # 'B_spin5': False,
        # 'B_spinm5': False,
        # 'B_spin3': True,
        # 'B_spinm3': True,
        # 'B_spin1': True,
        # 'B_spinm1': True,
        # 'O2': True,
        # 'O2_631g': False,
        'O2_631g': True,
    }

    for case in test_cases.keys():
        if not run_test[case]:
            continue
        print(case)
        atom = test_cases[case]['atom']
        basis = test_cases[case]['basis']
        spin = test_cases[case]['spin']
        ref_ecorr = test_cases[case]['ref_ecorr']

        mol = gto.M(atom=atom, basis=basis, verbose=0, spin=spin)

        mf = scf.UHF(mol)
        mf.level_shift = 0.0
        mf.conv_tol = 1e-14
        mf.max_cycle = 1000
        mf.kernel()

        frozen = 0

        mycc = UCCSDT(mf, frozen=frozen)
        mycc.conv_tol = 1e-12
        mycc.conv_tol_normt = 1e-10
        mycc.max_cycle = 100
        mycc.verbose = 5
        mycc.diis_with_t3 = True
        mycc.num_of_subiters = 2
        ecorr, t1, t2, t3 = mycc.kernel()
        print("My E_corr: % .16f    Ref E_corr: % .16f    Diff: % .16e"%(ecorr, ref_ecorr, ecorr - ref_ecorr))
        print()
