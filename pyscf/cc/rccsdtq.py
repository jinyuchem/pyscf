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
RHF-CCSDTQ with T4 amplitudes stored only for the i <= j <= k <= l index combinations.
T1-dressed formalism is used, where the T1 amplitudes are absorbed into the Fock matrix and ERIs.

Ref:
J. Chem. Phys. 142, 064108 (2015); DOI:10.1063/1.4907278
Chem. Phys. Lett. 228, 233 (1994); DOI:10.1016/0009-2614(94)00898-1
'''

import numpy as np
import numpy
import ctypes
from pyscf import lib
from pyscf.lib import logger
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask, get_e_hf, _mo_without_core
from pyscf.cc import ccsd, _ccsd, rccsdt
from pyscf.cc.rccsdt import (einsum_, t3_spin_summation_inplace_, symmetrize_tamps_tril_, purify_tamps_tril_,
                            update_t1_fock_eris_, intermediates_t1t2_, compute_r1r2, r1r2_divide_e_,
                            intermediates_t3_, kernel, _PhysicistsERIs)
from pyscf.cc.rccsdt_highm import (t3_spin_summation, t3_perm_symmetrize_inplace_, purify_tamps_, r1r2_add_t3_,
                                    intermediates_t3_add_t3_, compute_r3, r3_divide_e_)
from pyscf import __config__


def t4_spin_summation_inplace_c(A, nocc4, nvir, pattern, alpha=1.0, beta=0.0):
    assert A.dtype == np.float64 and A.flags['C_CONTIGUOUS'], "A must be a contiguous float64 array"
    pattern_c = pattern.encode('utf-8')
    _ccsd.libccsdt.t4_spin_summation_inplace_c(
        A.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int64(nocc4),
        ctypes.c_int64(nvir),
        ctypes.c_char_p(pattern_c),
        ctypes.c_double(alpha),
        ctypes.c_double(beta)
    )

def t4_add_(t4, r4, nocc4, nvir):
    assert t4.dtype == np.float64 and t4.flags['C_CONTIGUOUS'], "t4 must be a contiguous float64 array"
    assert r4.dtype == np.float64 and r4.flags['C_CONTIGUOUS'], "r4 must be a contiguous float64 array"
    _ccsd.libccsdt.t4_add_c(
        t4.ctypes.data_as(ctypes.c_void_p),
        r4.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int64(nocc4),
        ctypes.c_int64(nvir),
    )

def unpack_t4_tril2block_(t4, t4_blk, map_, mask, i0, i1, j0, j1, k0, k1, l0, l1,
                            nocc, nvir, blk_i, blk_j, blk_k, blk_l):
    assert t4.dtype == np.float64 and t4_blk.dtype == np.float64
    assert map_.dtype == np.int64 and mask.dtype == np.bool_
    t4_c = np.ascontiguousarray(t4)
    t4_blk_c = np.ascontiguousarray(t4_blk)
    map_c = np.ascontiguousarray(map_)
    mask_c = np.ascontiguousarray(mask)
    _ccsd.libccsdt.unpack_t4_tril2block_c(
        t4_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        t4_blk_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        map_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        mask_c.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
        ctypes.c_int64(i0), ctypes.c_int64(i1),
        ctypes.c_int64(j0), ctypes.c_int64(j1),
        ctypes.c_int64(k0), ctypes.c_int64(k1),
        ctypes.c_int64(l0), ctypes.c_int64(l1),
        ctypes.c_int64(nocc), ctypes.c_int64(nvir),
        ctypes.c_int64(blk_i), ctypes.c_int64(blk_j), ctypes.c_int64(blk_k), ctypes.c_int64(blk_l)
    )
    return t4_blk

def accumulate_t4_block2tril_(t4, t4_blk, map_, i0, i1, j0, j1, k0, k1, l0, l1,
                                nocc, nvir, blk_i, blk_j, blk_k, blk_l, alpha, beta):
    assert t4.dtype == np.float64 and t4_blk.dtype == np.float64
    assert map_.dtype == np.int64
    t4_c = np.ascontiguousarray(t4)
    t4_blk_c = np.ascontiguousarray(t4_blk)
    map_c = np.ascontiguousarray(map_)
    _ccsd.libccsdt.accumulate_t4_block2tril_c(
        t4_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        t4_blk_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        map_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(i0), ctypes.c_int64(i1),
        ctypes.c_int64(j0), ctypes.c_int64(j1),
        ctypes.c_int64(k0), ctypes.c_int64(k1),
        ctypes.c_int64(l0), ctypes.c_int64(l1),
        ctypes.c_int64(nocc), ctypes.c_int64(nvir),
        ctypes.c_int64(blk_i), ctypes.c_int64(blk_j), ctypes.c_int64(blk_k), ctypes.c_int64(blk_l),
        ctypes.c_double(alpha), ctypes.c_double(beta)
    )
    return t4

def _unpack_t4_(mycc, t4, t4_blk, i0, i1, j0, j1, k0, k1, l0, l1,
                    blksize0=None, blksize1=None, blksize2=None, blksize3=None):
    if blksize0 is None: blksize0 = mycc.blksize
    if blksize1 is None: blksize1 = mycc.blksize
    if blksize2 is None: blksize2 = mycc.blksize
    if blksize3 is None: blksize3 = mycc.blksize
    unpack_t4_tril2block_(t4, t4_blk, mycc.tril2cube_map, mycc.tril2cube_mask, i0, i1, j0, j1, k0, k1, l0, l1,
                        mycc.nocc, mycc.nmo - mycc.nocc, blksize0, blksize1, blksize2, blksize3)

def _accumulate_t4_(mycc, t4, t4_blk, i0, i1, j0, j1, k0, k1, l0, l1,
                    blksize0=None, blksize1=None, blksize2=None, blksize3=None, alpha=1.0, beta=0.0):
    if blksize0 is None: blksize0 = mycc.blksize
    if blksize1 is None: blksize1 = mycc.blksize
    if blksize2 is None: blksize2 = mycc.blksize
    if blksize3 is None: blksize3 = mycc.blksize
    accumulate_t4_block2tril_(t4, t4_blk, mycc.tril2cube_map, i0, i1, j0, j1, k0, k1, l0, l1, mycc.nocc,
                                mycc.nmo - mycc.nocc, blksize0, blksize1, blksize2, blksize3, alpha=alpha, beta=beta)

def r2_add_t4_tril_(mycc, t4, r2):
    '''Add the T4 contributions to r2. T4 amplitudes are stored in triangular form.'''
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    blksize = mycc.blksize

    t4_tmp = np.empty((blksize,) * 4 + (nvir,) * 4, dtype=np.float64)
    for m0, m1 in lib.prange(0, nocc, blksize):
        bm = m1 - m0
        for n0, n1 in lib.prange(0, nocc, blksize):
            bn = n1 - n0
            for i0, i1 in lib.prange(0, nocc, blksize):
                bi = i1 - i0
                for j0, j1 in lib.prange(0, nocc, blksize):
                    bj = j1 - j0
                    _unpack_t4_(mycc, t4, t4_tmp, m0, m1, n0, n1, i0, i1, j0, j1)
                    t4_spin_summation_inplace_c(t4_tmp, blksize**4, nvir, "P4_442", 1.0, 0.0)
                    einsum_(mycc, 'mnef,mnijefab->ijab', mycc.t1_eris[m0:m1, n0:n1, nocc:, nocc:],
                        t4_tmp[:bm, :bn, :bi, :bj], out=r2[i0:i1, j0:j1, :, :], alpha=0.25, beta=1.0)
    t4_tmp = None

def r3_add_t4_tril_(mycc, t4, r3):
    '''Add the T4 contributions to r3. T4 amplitudes are stored in triangular form.'''
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    blksize = mycc.blksize
    t1_fock, t1_eris = mycc.t1_fock, mycc.t1_eris

    t4_tmp = np.empty((blksize,) * 4 + (nvir,) * 4, dtype=t4.dtype)
    for m0, m1 in lib.prange(0, nocc, blksize):
        bm = m1 - m0
        for i0, i1 in lib.prange(0, nocc, blksize):
            bi = i1 - i0
            for j0, j1 in lib.prange(0, nocc, blksize):
                bj = j1 - j0
                for k0, k1 in lib.prange(0, nocc, blksize):
                    bk = k1 - k0
                    _unpack_t4_(mycc, t4, t4_tmp, m0, m1, i0, i1, j0, j1, k0, k1)
                    t4_spin_summation_inplace_c(t4_tmp, blksize**4, nvir, "P4_201", 1.0, 0.0)
                    # R3: P9
                    einsum_(mycc, 'me,mijkeabc->ijkabc', t1_fock[m0:m1, nocc:], t4_tmp[:bm, :bi, :bj, :bk],
                            out=r3[i0:i1, j0:j1, k0:k1, ...], alpha=1.0 / 6.0, beta=1.0)
                    # R3: P10
                    einsum_(mycc, 'amef,mijkfebc->ijkabc', t1_eris[nocc:, m0:m1, nocc:, nocc:],
                        t4_tmp[:bm, :bi, :bj, :bk], out=r3[i0:i1, j0:j1, k0:k1, ...], alpha=0.5, beta=1.0)
                    # R3: P11
                    einsum_(mycc, 'mjen,mijkeabc->inkabc', t1_eris[m0:m1, j0:j1, nocc:, :nocc],
                        t4_tmp[:bm, :bi, :bj, :bk], out=r3[i0:i1, :, k0:k1, ...], alpha=-0.5, beta=1.0)
    t4_tmp = None

def intermediates_t4_tril_(mycc, t2, t3, t4):
    '''Intermediates for the T4 residual equation, with T4 amplitudes stored in triangular form.'''
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    blksize = mycc.blksize
    t1_fock, t1_eris = mycc.t1_fock, mycc.t1_eris

    c_t2 = 2.0 * t2 - t2.transpose(0, 1, 3, 2)
    c_t3 = np.empty_like(t3)
    t3_spin_summation(t3, c_t3, nocc**3, nvir, "P3_201", 1.0, 0.0)

    einsum_(mycc, 'me,mjab->abej', t1_fock[:nocc, nocc:], t2, out=mycc.W_vvvo_tc, alpha=-1.0, beta=1.0)

    W_ovvvoo = np.empty((nocc,) + (nvir,) * 3 + (nocc,) * 2)
    einsum_(mycc, 'maef,jibf->mabeij', t1_eris[:nocc, nocc:, nocc:, nocc:], t2, out=W_ovvvoo, alpha=2.0, beta=0.0)
    einsum_(mycc, 'mafe,jibf->mabeij', t1_eris[:nocc, nocc:, nocc:, nocc:], t2, out=W_ovvvoo, alpha=-1.0, beta=1.0)
    einsum_(mycc, 'mnei,njab->mabeij', t1_eris[:nocc, :nocc, nocc:, :nocc], t2, out=W_ovvvoo, alpha=-2.0, beta=1.0)
    einsum_(mycc, 'nmei,njab->mabeij', t1_eris[:nocc, :nocc, nocc:, :nocc], t2, out=W_ovvvoo, alpha=1.0, beta=1.0)
    einsum_(mycc, 'nmfe,nijfab->mabeij', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t3, out=W_ovvvoo, alpha=0.5, beta=1.0)
    einsum_(mycc, 'mnfe,nijfab->mabeij', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t3, out=W_ovvvoo, alpha=-0.25, beta=1.0)
    W_ovvvoo += W_ovvvoo.transpose(0, 2, 1, 3, 5, 4)
    mycc.W_ovvvoo = W_ovvvoo
    c_t3 = None

    W_ovvovo = np.empty((nocc, nvir, nvir, nocc, nvir, nocc))
    einsum_(mycc, 'mafe,jibf->mabiej', t1_eris[:nocc, nocc:, nocc:, nocc:], t2, out=W_ovvovo, alpha=1.0, beta=0.0)
    einsum_(mycc, 'mnie,njab->mabiej', t1_eris[:nocc, :nocc, :nocc, nocc:], t2, out=W_ovvovo, alpha=-1.0, beta=1.0)
    einsum_(mycc, 'nmef,injfab->mabiej', t1_eris[:nocc, :nocc, nocc:, nocc:], t3, out=W_ovvovo, alpha=-0.5, beta=1.0)
    mycc.W_ovvovo = W_ovvovo

    W_vooooo = np.empty((nvir,) + (nocc,) * 5)
    einsum_(mycc, 'mnek,ijae->amnijk', t1_eris[:nocc, :nocc, nocc:, :nocc], t2, out=W_vooooo, alpha=1.0, beta=0.0)
    einsum_(mycc, 'mnef,ijkaef->amnijk', t1_eris[:nocc, :nocc, nocc:, nocc:], t3, out=W_vooooo, alpha=0.5, beta=1.0)
    W_vooooo += W_vooooo.transpose(0, 2, 1, 3, 5, 4)
    mycc.W_vooooo = W_vooooo

    W_vvoooo = np.empty((nvir,) * 2 + (nocc,) * 4)
    einsum_(mycc, 'amef,ijkebf->abmijk', t1_eris[nocc:, :nocc, nocc:, nocc:], t3, out=W_vvoooo, alpha=1.0, beta=0.0)
    # TODO: Derive an alternative expression for this term
    W_ovvo_c = t1_eris[:nocc, nocc:, nocc:, :nocc].copy()
    einsum_(mycc, 'nmfe,nifa->maei', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t2, out=W_ovvo_c, alpha=1.0, beta=1.0)
    einsum_(mycc, 'mnfe,nifa->maei', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t2, out=W_ovvo_c, alpha=-0.5, beta=1.0)
    einsum_(mycc, 'nmef,infa->maei', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=W_ovvo_c, alpha=-0.5, beta=1.0)
    #
    einsum_(mycc, 'maei,jkbe->abmijk', W_ovvo_c, t2, out=W_vvoooo, alpha=1.0, beta=1.0)
    einsum_(mycc, 'make,jibe->abmijk', mycc.W_ovov_tc, t2, out=W_vvoooo, alpha=1.0, beta=1.0)
    einsum_(mycc, 'mnki,njab->abmijk', mycc.W_oooo, t2, out=W_vvoooo, alpha=-0.5, beta=1.0)

    W_vvvvoo = np.empty((nvir,) * 4 + (nocc,) * 2)
    einsum_(mycc, 'abef,jkfc->abcejk', mycc.W_vvvv_tc, t2, out=W_vvvvoo, alpha=0.5, beta=0.0)

    t4_tmp = np.empty((blksize,) * 4 + (nvir,) * 4, dtype=t4.dtype)
    for n0, n1 in lib.prange(0, nocc, blksize):
        bn = n1 - n0
        for i0, i1 in lib.prange(0, nocc, blksize):
            bi = i1 - i0
            for j0, j1 in lib.prange(0, nocc, blksize):
                bj = j1 - j0
                for k0, k1 in lib.prange(0, nocc, blksize):
                    bk = k1 - k0
                    _unpack_t4_(mycc, t4, t4_tmp, n0, n1, i0, i1, j0, j1, k0, k1)
                    t4_spin_summation_inplace_c(t4_tmp, blksize**4, nvir, "P4_201", 1.0, 0.0)
                    einsum_(mycc, 'mnef,nijkfabe->abmijk', t1_eris[:nocc, n0:n1, nocc:, nocc:],
                        t4_tmp[:bn, :bi, :bj, :bk], out=W_vvoooo[..., i0:i1, j0:j1, k0:k1], alpha=0.5, beta=1.0)
                    einsum_(mycc, 'inef,nijkfabc->abcejk', t1_eris[i0:i1, n0:n1, nocc:, nocc:],
                        t4_tmp[:bn, :bi, :bj, :bk], out=W_vvvvoo[..., j0:j1, k0:k1], alpha=-0.5, beta=1.0)
    t4_tmp = None

    W_vvoooo += W_vvoooo.transpose(1, 0, 2, 4, 3, 5)
    W_vvvvoo += W_vvvvoo.transpose(0, 2, 1, 3, 5, 4)

    mycc.W_vvoooo = W_vvoooo
    mycc.W_vvvvoo = W_vvvvoo
    return mycc

def compute_r4_tril(mycc, t2, t3, t4):
    '''Compute r4 with triangular-stored T4 amplitudes; r4 is returned in triangular form as well.
    r4 will require a symmetry restoration step afterward.
    '''
    time1 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    nocc, nmo, blksize = mycc.nocc, mycc.nmo, mycc.blksize
    nvir = nmo - nocc
    c_t3 = np.empty_like(t3)
    t3_spin_summation(t3, c_t3, nocc**3, nvir, "P3_201", 1.0, 0.0)

    # r4 = np.empty_like(t4)
    r4 = np.zeros_like(t4)

    time2 = logger.process_clock(), logger.perf_counter()
    t4_tmp = np.empty((blksize,) * 4 + (nvir,) * 4, dtype=t4.dtype)
    r4_tmp = np.empty((blksize,) * 4 + (nvir,) * 4, dtype=t4.dtype)
    for l0, l1 in lib.prange(0, nocc, blksize):
        bl = l1 - l0
        for k0, k1 in lib.prange(0, l1, blksize):
            bk = k1 - k0
            for j0, j1 in lib.prange(0, k1, blksize):
                bj = j1 - j0
                for i0, i1 in lib.prange(0, j1, blksize):
                    bi = i1 - i0

                    # R4: P0 (o4v5 * 12)
                    einsum_(mycc, "abej,iklecd->ijklabcd", mycc.W_vvvo_tc[..., j0:j1], t3[i0:i1, k0:k1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=0.0)
                    einsum_(mycc, "acek,ijlebd->ijklabcd", mycc.W_vvvo_tc[..., k0:k1], t3[i0:i1, j0:j1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "adel,ijkebc->ijklabcd", mycc.W_vvvo_tc[..., l0:l1], t3[i0:i1, j0:j1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "baei,jklecd->ijklabcd", mycc.W_vvvo_tc[..., i0:i1], t3[j0:j1, k0:k1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "caei,kjlebd->ijklabcd", mycc.W_vvvo_tc[..., i0:i1], t3[k0:k1, j0:j1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "daei,ljkebc->ijklabcd", mycc.W_vvvo_tc[..., i0:i1], t3[l0:l1, j0:j1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "bcek,jilead->ijklabcd", mycc.W_vvvo_tc[..., k0:k1], t3[j0:j1, i0:i1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "bdel,jikeac->ijklabcd", mycc.W_vvvo_tc[..., l0:l1], t3[j0:j1, i0:i1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "cbej,kilead->ijklabcd", mycc.W_vvvo_tc[..., j0:j1], t3[k0:k1, i0:i1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "dbej,likeac->ijklabcd", mycc.W_vvvo_tc[..., j0:j1], t3[l0:l1, i0:i1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "cdel,kijeab->ijklabcd", mycc.W_vvvo_tc[..., l0:l1], t3[k0:k1, i0:i1, j0:j1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "dcek,lijeab->ijklabcd", mycc.W_vvvo_tc[..., k0:k1], t3[l0:l1, i0:i1, j0:j1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)

                    # R4: P1 (o5v4 * 12)
                    einsum_(mycc, "amij,mklbcd->ijklabcd", mycc.W_vooo_tc[:, :, i0:i1, j0:j1],
                        t3[:, k0:k1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "amik,mjlcbd->ijklabcd", mycc.W_vooo_tc[:, :, i0:i1, k0:k1],
                        t3[:, j0:j1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "amil,mjkdbc->ijklabcd", mycc.W_vooo_tc[:, :, i0:i1, l0:l1],
                        t3[:, j0:j1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "bmji,mklacd->ijklabcd", mycc.W_vooo_tc[:, :, j0:j1, i0:i1],
                        t3[:, k0:k1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "cmki,mjlabd->ijklabcd", mycc.W_vooo_tc[:, :, k0:k1, i0:i1],
                        t3[:, j0:j1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "dmli,mjkabc->ijklabcd", mycc.W_vooo_tc[:, :, l0:l1, i0:i1],
                        t3[:, j0:j1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "bmjk,milcad->ijklabcd", mycc.W_vooo_tc[:, :, j0:j1, k0:k1],
                        t3[:, i0:i1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "bmjl,mikdac->ijklabcd", mycc.W_vooo_tc[:, :, j0:j1, l0:l1],
                        t3[:, i0:i1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "cmkj,milbad->ijklabcd", mycc.W_vooo_tc[:, :, k0:k1, j0:j1],
                        t3[:, i0:i1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "dmlj,mikbac->ijklabcd", mycc.W_vooo_tc[:, :, l0:l1, j0:j1],
                        t3[:, i0:i1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "cmkl,mijdab->ijklabcd", mycc.W_vooo_tc[:, :, k0:k1, l0:l1],
                        t3[:, i0:i1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "dmlk,mijcab->ijklabcd", mycc.W_vooo_tc[:, :, l0:l1, k0:k1],
                        t3[:, i0:i1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)

                    # R4: P9 (o5v5 * 12)
                    einsum_(mycc, "mabeij,mklecd->ijklabcd", mycc.W_ovvvoo[..., i0:i1, j0:j1],
                        c_t3[:, k0:k1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum_(mycc, "maceik,mjlebd->ijklabcd", mycc.W_ovvvoo[..., i0:i1, k0:k1],
                        c_t3[:, j0:j1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum_(mycc, "madeil,mjkebc->ijklabcd", mycc.W_ovvvoo[..., i0:i1, l0:l1],
                        c_t3[:, j0:j1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum_(mycc, "mbaeji,mklecd->ijklabcd", mycc.W_ovvvoo[..., j0:j1, i0:i1],
                        c_t3[:, k0:k1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum_(mycc, "mcaeki,mjlebd->ijklabcd", mycc.W_ovvvoo[..., k0:k1, i0:i1],
                        c_t3[:, j0:j1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum_(mycc, "mdaeli,mjkebc->ijklabcd", mycc.W_ovvvoo[..., l0:l1, i0:i1],
                        c_t3[:, j0:j1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum_(mycc, "mbcejk,milead->ijklabcd", mycc.W_ovvvoo[..., j0:j1, k0:k1],
                        c_t3[:, i0:i1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum_(mycc, "mbdejl,mikeac->ijklabcd", mycc.W_ovvvoo[..., j0:j1, l0:l1],
                        c_t3[:, i0:i1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum_(mycc, "mcbekj,milead->ijklabcd", mycc.W_ovvvoo[..., k0:k1, j0:j1],
                        c_t3[:, i0:i1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum_(mycc, "mdbelj,mikeac->ijklabcd", mycc.W_ovvvoo[..., l0:l1, j0:j1],
                        c_t3[:, i0:i1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum_(mycc, "mcdekl,mijeab->ijklabcd", mycc.W_ovvvoo[..., k0:k1, l0:l1],
                        c_t3[:, i0:i1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum_(mycc, "mdcelk,mijeab->ijklabcd", mycc.W_ovvvoo[..., l0:l1, k0:k1],
                        c_t3[:, i0:i1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)

                    # R4: P10 (o5v5 * 24)
                    einsum_(mycc, "mabiej,mklced->ijklabcd", mycc.W_ovvovo[..., i0:i1, :, j0:j1],
                        t3[:, k0:k1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_(mycc, "mabiej,mlkdec->ijklabcd", mycc.W_ovvovo[..., i0:i1, :, j0:j1],
                        t3[:, l0:l1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_(mycc, "maciek,mjlbed->ijklabcd", mycc.W_ovvovo[..., i0:i1, :, k0:k1],
                        t3[:, j0:j1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_(mycc, "madiel,mjkbec->ijklabcd", mycc.W_ovvovo[..., i0:i1, :, l0:l1],
                        t3[:, j0:j1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_(mycc, "maciek,mljdeb->ijklabcd", mycc.W_ovvovo[..., i0:i1, :, k0:k1],
                        t3[:, l0:l1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_(mycc, "madiel,mkjceb->ijklabcd", mycc.W_ovvovo[..., i0:i1, :, l0:l1],
                        t3[:, k0:k1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_(mycc, "mbajei,mklced->ijklabcd", mycc.W_ovvovo[..., j0:j1, :, i0:i1],
                        t3[:, k0:k1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_(mycc, "mbajei,mlkdec->ijklabcd", mycc.W_ovvovo[..., j0:j1, :, i0:i1],
                        t3[:, l0:l1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_(mycc, "mcakei,mjlbed->ijklabcd", mycc.W_ovvovo[..., k0:k1, :, i0:i1],
                        t3[:, j0:j1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_(mycc, "mdalei,mjkbec->ijklabcd", mycc.W_ovvovo[..., l0:l1, :, i0:i1],
                        t3[:, j0:j1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_(mycc, "mcakei,mljdeb->ijklabcd", mycc.W_ovvovo[..., k0:k1, :, i0:i1],
                        t3[:, l0:l1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_(mycc, "mdalei,mkjceb->ijklabcd", mycc.W_ovvovo[..., l0:l1, :, i0:i1],
                        t3[:, k0:k1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_(mycc, "mbcjek,milaed->ijklabcd", mycc.W_ovvovo[..., j0:j1, :, k0:k1],
                        t3[:, i0:i1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_(mycc, "mbdjel,mikaec->ijklabcd", mycc.W_ovvovo[..., j0:j1, :, l0:l1],
                        t3[:, i0:i1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_(mycc, "mcbkej,milaed->ijklabcd", mycc.W_ovvovo[..., k0:k1, :, j0:j1],
                        t3[:, i0:i1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_(mycc, "mdblej,mikaec->ijklabcd", mycc.W_ovvovo[..., l0:l1, :, j0:j1],
                        t3[:, i0:i1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_(mycc, "mcdkel,mijaeb->ijklabcd", mycc.W_ovvovo[..., k0:k1, :, l0:l1],
                        t3[:, i0:i1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_(mycc, "mdclek,mijaeb->ijklabcd", mycc.W_ovvovo[..., l0:l1, :, k0:k1],
                        t3[:, i0:i1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_(mycc, "mbcjek,mlidea->ijklabcd", mycc.W_ovvovo[..., j0:j1, :, k0:k1],
                        t3[:, l0:l1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_(mycc, "mbdjel,mkicea->ijklabcd", mycc.W_ovvovo[..., j0:j1, :, l0:l1],
                        t3[:, k0:k1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_(mycc, "mcbkej,mlidea->ijklabcd", mycc.W_ovvovo[..., k0:k1, :, j0:j1],
                        t3[:, l0:l1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_(mycc, "mdblej,mkicea->ijklabcd", mycc.W_ovvovo[..., l0:l1, :, j0:j1],
                        t3[:, k0:k1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_(mycc, "mcdkel,mjibea->ijklabcd", mycc.W_ovvovo[..., k0:k1, :, l0:l1],
                        t3[:, j0:j1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_(mycc, "mdclek,mjibea->ijklabcd", mycc.W_ovvovo[..., l0:l1, :, k0:k1],
                        t3[:, j0:j1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)

                    # R4: P11 (o5v5 * 24)
                    einsum_(mycc, "mcbiej,mklaed->ijklabcd", mycc.W_ovvovo[..., i0:i1, :, j0:j1],
                        t3[:, k0:k1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mdbiej,mlkaec->ijklabcd", mycc.W_ovvovo[..., i0:i1, :, j0:j1],
                        t3[:, l0:l1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mbciek,mjlaed->ijklabcd", mycc.W_ovvovo[..., i0:i1, :, k0:k1],
                        t3[:, j0:j1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mbdiel,mjkaec->ijklabcd", mycc.W_ovvovo[..., i0:i1, :, l0:l1],
                        t3[:, j0:j1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mdciek,mljaeb->ijklabcd", mycc.W_ovvovo[..., i0:i1, :, k0:k1],
                        t3[:, l0:l1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mcdiel,mkjaeb->ijklabcd", mycc.W_ovvovo[..., i0:i1, :, l0:l1],
                        t3[:, k0:k1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mcajei,mklbed->ijklabcd", mycc.W_ovvovo[..., j0:j1, :, i0:i1],
                        t3[:, k0:k1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mdajei,mlkbec->ijklabcd", mycc.W_ovvovo[..., j0:j1, :, i0:i1],
                        t3[:, l0:l1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mbakei,mjlced->ijklabcd", mycc.W_ovvovo[..., k0:k1, :, i0:i1],
                        t3[:, j0:j1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mbalei,mjkdec->ijklabcd", mycc.W_ovvovo[..., l0:l1, :, i0:i1],
                        t3[:, j0:j1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mdakei,mljceb->ijklabcd", mycc.W_ovvovo[..., k0:k1, :, i0:i1],
                        t3[:, l0:l1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mcalei,mkjdeb->ijklabcd", mycc.W_ovvovo[..., l0:l1, :, i0:i1],
                        t3[:, k0:k1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "macjek,milbed->ijklabcd", mycc.W_ovvovo[..., j0:j1, :, k0:k1],
                        t3[:, i0:i1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "madjel,mikbec->ijklabcd", mycc.W_ovvovo[..., j0:j1, :, l0:l1],
                        t3[:, i0:i1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mabkej,milced->ijklabcd", mycc.W_ovvovo[..., k0:k1, :, j0:j1],
                        t3[:, i0:i1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mablej,mikdec->ijklabcd", mycc.W_ovvovo[..., l0:l1, :, j0:j1],
                        t3[:, i0:i1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "madkel,mijceb->ijklabcd", mycc.W_ovvovo[..., k0:k1, :, l0:l1],
                        t3[:, i0:i1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "maclek,mijdeb->ijklabcd", mycc.W_ovvovo[..., l0:l1, :, k0:k1],
                        t3[:, i0:i1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mdcjek,mlibea->ijklabcd", mycc.W_ovvovo[..., j0:j1, :, k0:k1],
                        t3[:, l0:l1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mcdjel,mkibea->ijklabcd", mycc.W_ovvovo[..., j0:j1, :, l0:l1],
                        t3[:, k0:k1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mdbkej,mlicea->ijklabcd", mycc.W_ovvovo[..., k0:k1, :, j0:j1],
                        t3[:, l0:l1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mcblej,mkidea->ijklabcd", mycc.W_ovvovo[..., l0:l1, :, j0:j1],
                        t3[:, k0:k1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mbdkel,mjicea->ijklabcd", mycc.W_ovvovo[..., k0:k1, :, l0:l1],
                        t3[:, j0:j1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mbclek,mjidea->ijklabcd", mycc.W_ovvovo[..., l0:l1, :, k0:k1],
                        t3[:, j0:j1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)

                    # R4: P12 (o6v4 * 12)
                    einsum_(mycc, "amnijk,mnlbcd->ijklabcd", mycc.W_vooooo[..., i0:i1, j0:j1, k0:k1],
                        t3[:, :, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "amnijl,mnkbdc->ijklabcd", mycc.W_vooooo[..., i0:i1, j0:j1, l0:l1],
                        t3[:, :, k0:k1,], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "amnikl,mnjcdb->ijklabcd", mycc.W_vooooo[..., i0:i1, k0:k1, l0:l1],
                        t3[:, :, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "bmnjik,mnlacd->ijklabcd", mycc.W_vooooo[..., j0:j1, i0:i1, k0:k1],
                        t3[:, :, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "bmnjil,mnkadc->ijklabcd", mycc.W_vooooo[..., j0:j1, i0:i1, l0:l1],
                        t3[:, :, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "bmnjkl,mnicda->ijklabcd", mycc.W_vooooo[..., j0:j1, k0:k1, l0:l1],
                        t3[:, :, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "cmnkij,mnlabd->ijklabcd", mycc.W_vooooo[..., k0:k1, i0:i1, j0:j1],
                        t3[:, :, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "cmnkil,mnjadb->ijklabcd", mycc.W_vooooo[..., k0:k1, i0:i1, l0:l1],
                        t3[:, :, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "cmnkjl,mnibda->ijklabcd", mycc.W_vooooo[..., k0:k1, j0:j1, l0:l1],
                        t3[:, :, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "dmnlij,mnkabc->ijklabcd", mycc.W_vooooo[..., l0:l1, i0:i1, j0:j1],
                        t3[:, :, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "dmnlik,mnjacb->ijklabcd", mycc.W_vooooo[..., l0:l1, i0:i1, k0:k1],
                        t3[:, :, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "dmnljk,mnibca->ijklabcd", mycc.W_vooooo[..., l0:l1, j0:j1, k0:k1],
                        t3[:, :, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)

                    # R4: P13 (o5v4 * 12)
                    einsum_(mycc, "mlcd,abmijk->ijklabcd", t2[:, l0:l1], mycc.W_vvoooo[..., i0:i1, j0:j1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mkdc,abmijl->ijklabcd", t2[:, k0:k1], mycc.W_vvoooo[..., i0:i1, j0:j1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mlbd,acmikj->ijklabcd", t2[:, l0:l1], mycc.W_vvoooo[..., i0:i1, k0:k1, j0:j1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mkbc,admilj->ijklabcd", t2[:, k0:k1], mycc.W_vvoooo[..., i0:i1, l0:l1, j0:j1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mjdb,acmikl->ijklabcd", t2[:, j0:j1], mycc.W_vvoooo[..., i0:i1, k0:k1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mjcb,admilk->ijklabcd", t2[:, j0:j1], mycc.W_vvoooo[..., i0:i1, l0:l1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mlad,bcmjki->ijklabcd", t2[:, l0:l1], mycc.W_vvoooo[..., j0:j1, k0:k1, i0:i1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mkac,bdmjli->ijklabcd", t2[:, k0:k1], mycc.W_vvoooo[..., j0:j1, l0:l1, i0:i1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mjab,cdmkli->ijklabcd", t2[:, j0:j1], mycc.W_vvoooo[..., k0:k1, l0:l1, i0:i1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mida,bcmjkl->ijklabcd", t2[:, i0:i1], mycc.W_vvoooo[..., j0:j1, k0:k1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "mica,bdmjlk->ijklabcd", t2[:, i0:i1], mycc.W_vvoooo[..., j0:j1, l0:l1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_(mycc, "miba,cdmklj->ijklabcd", t2[:, i0:i1], mycc.W_vvoooo[..., k0:k1, l0:l1, j0:j1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)

                    # R4: P14 (o4v5 * 12)
                    einsum_(mycc, "iled,abcejk->ijklabcd", t2[i0:i1, l0:l1], mycc.W_vvvvoo[..., j0:j1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "ikec,abdejl->ijklabcd", t2[i0:i1, k0:k1], mycc.W_vvvvoo[..., j0:j1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "ijeb,acdekl->ijklabcd", t2[i0:i1, j0:j1], mycc.W_vvvvoo[..., k0:k1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "jled,baceik->ijklabcd", t2[j0:j1, l0:l1], mycc.W_vvvvoo[..., i0:i1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "jkec,badeil->ijklabcd", t2[j0:j1, k0:k1], mycc.W_vvvvoo[..., i0:i1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "kled,cabeij->ijklabcd", t2[k0:k1, l0:l1], mycc.W_vvvvoo[..., i0:i1, j0:j1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "lkec,dabeij->ijklabcd", t2[l0:l1, k0:k1], mycc.W_vvvvoo[..., i0:i1, j0:j1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "kjeb,cadeil->ijklabcd", t2[k0:k1, j0:j1], mycc.W_vvvvoo[..., i0:i1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "ljeb,daceik->ijklabcd", t2[l0:l1, j0:j1], mycc.W_vvvvoo[..., i0:i1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "jiea,bcdekl->ijklabcd", t2[j0:j1, i0:i1], mycc.W_vvvvoo[..., k0:k1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "kiea,cbdejl->ijklabcd", t2[k0:k1, i0:i1], mycc.W_vvvvoo[..., j0:j1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_(mycc, "liea,dbcejk->ijklabcd", t2[l0:l1, i0:i1], mycc.W_vvvvoo[..., j0:j1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)

                    # R4: P2 (o4v5 * 4)
                    _unpack_t4_(mycc, t4, t4_tmp, i0, i1, j0, j1, k0, k1, l0, l1)
                    einsum_(mycc, "ae,ijklebcd->ijklabcd", mycc.tf_vv, t4_tmp[:bi, :bj, :bk, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    _unpack_t4_(mycc, t4, t4_tmp, j0, j1, i0, i1, k0, k1, l0, l1)
                    einsum_(mycc, "be,jikleacd->ijklabcd", mycc.tf_vv, t4_tmp[:bj, :bi, :bk, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    _unpack_t4_(mycc, t4, t4_tmp, k0, k1, i0, i1, j0, j1, l0, l1)
                    einsum_(mycc, "ce,kijleabd->ijklabcd", mycc.tf_vv, t4_tmp[:bk, :bi, :bj, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    _unpack_t4_(mycc, t4, t4_tmp, l0, l1, i0, i1, j0, j1, k0, k1)
                    einsum_(mycc, "de,lijkeabc->ijklabcd", mycc.tf_vv, t4_tmp[:bl, :bi, :bj, :bk],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)

                    # R4: P8 (o4v6 * 6)
                    _unpack_t4_(mycc, t4, t4_tmp, i0, i1, j0, j1, k0, k1, l0, l1)
                    einsum_(mycc, "abef,ijklefcd->ijklabcd", mycc.W_vvvv_tc, t4_tmp[:bi, :bj, :bk, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    _unpack_t4_(mycc, t4, t4_tmp, i0, i1, k0, k1, j0, j1, l0, l1)
                    einsum_(mycc, "acef,ikjlefbd->ijklabcd", mycc.W_vvvv_tc, t4_tmp[:bi, :bk, :bj, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    _unpack_t4_(mycc, t4, t4_tmp, i0, i1, l0, l1, j0, j1, k0, k1)
                    einsum_(mycc, "adef,iljkefbc->ijklabcd", mycc.W_vvvv_tc, t4_tmp[:bi, :bl, :bj, :bk],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    _unpack_t4_(mycc, t4, t4_tmp, j0, j1, k0, k1, i0, i1, l0, l1)
                    einsum_(mycc, "bcef,jkilefad->ijklabcd", mycc.W_vvvv_tc, t4_tmp[:bj, :bk, :bi, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    _unpack_t4_(mycc, t4, t4_tmp, j0, j1, l0, l1, i0, i1, k0, k1)
                    einsum_(mycc, "bdef,jlikefac->ijklabcd", mycc.W_vvvv_tc, t4_tmp[:bj, :bl, :bi, :bk],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    _unpack_t4_(mycc, t4, t4_tmp, k0, k1, l0, l1, i0, i1, j0, j1)
                    einsum_(mycc, "cdef,klijefab->ijklabcd", mycc.W_vvvv_tc, t4_tmp[:bk, :bl, :bi, :bj],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)

                    _accumulate_t4_(mycc, r4, r4_tmp, i0, i1, j0, j1, k0, k1, l0, l1)
        time2 = log.timer_debug1('t4: iter: P0, P1, P9 - P14, P2, P8 [%3d, %3d]:' % (l0, l1), *time2)
    t4_tmp = None
    r4_tmp = None
    c_t3 = None
    time1 = log.timer_debug1('t4: P0, P1, P9 - P14, P2, P8', *time1)

    time2 = logger.process_clock(), logger.perf_counter()
    t4_tmp = np.empty((blksize,) * 4 + (nvir,) * 4, dtype=t4.dtype)
    r4_tmp = np.empty((blksize,) * 4 + (nvir,) * 4, dtype=t4.dtype)
    for l0, l1 in lib.prange(0, nocc, blksize):
        bl = l1 - l0
        for k0, k1 in lib.prange(0, l1, blksize):
            bk = k1 - k0
            for j0, j1 in lib.prange(0, k1, blksize):
                bj = j1 - j0
                for i0, i1 in lib.prange(0, j1, blksize):
                    bi = i1 - i0

                    r4_tmp[:] = 0.0
                    for m0, m1 in lib.prange(0, nocc, blksize):
                        bm = m1 - m0

                        # R4: P3 (o5v4 * 4) & P4 (o5v5 * 4)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, j0, j1, k0, k1, l0, l1)
                        einsum_(mycc, "mi,mjklabcd->ijklabcd", mycc.tf_oo[m0:m1, i0:i1], t4_tmp[:bm, :bj, :bk, :bl],
                            out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        t4_spin_summation_inplace_c(t4_tmp, blksize**4, nvir, "P4_201", 1.0, 0.0)
                        einsum_(mycc, "maei,mjklebcd->ijklabcd", mycc.W_ovvo_tc[m0:m1, :, :, i0:i1],
                            t4_tmp[:bm, :bj, :bk, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.5, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, i0, i1, k0, k1, l0, l1)
                        einsum_(mycc, "mj,miklbacd->ijklabcd", mycc.tf_oo[m0:m1, j0:j1], t4_tmp[:bm, :bi, :bk, :bl],
                            out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        t4_spin_summation_inplace_c(t4_tmp, blksize**4, nvir, "P4_201", 1.0, 0.0)
                        einsum_(mycc, "mbej,mikleacd->ijklabcd", mycc.W_ovvo_tc[m0:m1, :, :, j0:j1],
                            t4_tmp[:bm, :bi, :bk, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.5, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, i0, i1, j0, j1, l0, l1)
                        einsum_(mycc, "mk,mijlcabd->ijklabcd", mycc.tf_oo[m0:m1, k0:k1], t4_tmp[:bm, :bi, :bj, :bl],
                            out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        t4_spin_summation_inplace_c(t4_tmp, blksize**4, nvir, "P4_201", 1.0, 0.0)
                        einsum_(mycc, "mcek,mijleabd->ijklabcd", mycc.W_ovvo_tc[m0:m1, :, :, k0:k1],
                            t4_tmp[:bm, :bi, :bj, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.5, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, i0, i1, j0, j1, k0, k1)
                        einsum_(mycc, "ml,mijkdabc->ijklabcd", mycc.tf_oo[m0:m1, l0:l1], t4_tmp[:bm, :bi, :bj, :bk],
                            out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        t4_spin_summation_inplace_c(t4_tmp, blksize**4, nvir, "P4_201", 1.0, 0.0)
                        einsum_(mycc, "mdel,mijkeabc->ijklabcd", mycc.W_ovvo_tc[m0:m1, :, :, l0:l1],
                            t4_tmp[:bm, :bi, :bj, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.5, beta=1.0)

                        # R4: P5 (o5v5 * 12) & P6 (o5v5 * 12)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, j0, j1, k0, k1, l0, l1)
                        einsum_(mycc, "maie,mjklbecd->ijklabcd", mycc.W_ovov_tc[m0:m1, :, i0:i1, :],
                            t4_tmp[:bm, :bj, :bk, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum_(mycc, "mbie,mjklaecd->ijklabcd", mycc.W_ovov_tc[m0:m1, :, i0:i1, :],
                            t4_tmp[:bm, :bj, :bk, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, k0, k1, j0, j1, l0, l1)
                        einsum_(mycc, "maie,mkjlcebd->ijklabcd", mycc.W_ovov_tc[m0:m1, :, i0:i1, :],
                            t4_tmp[:bm, :bk, :bj, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum_(mycc, "mcie,mkjlaebd->ijklabcd", mycc.W_ovov_tc[m0:m1, :, i0:i1, :],
                            t4_tmp[:bm, :bk, :bj, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, l0, l1, j0, j1, k0, k1)
                        einsum_(mycc, "maie,mljkdebc->ijklabcd", mycc.W_ovov_tc[m0:m1, :, i0:i1, :],
                            t4_tmp[:bm, :bl, :bj, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum_(mycc, "mdie,mljkaebc->ijklabcd", mycc.W_ovov_tc[m0:m1, :, i0:i1, :],
                            t4_tmp[:bm, :bl, :bj, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, i0, i1, k0, k1, l0, l1)
                        einsum_(mycc, "mbje,miklaecd->ijklabcd", mycc.W_ovov_tc[m0:m1, :, j0:j1, :],
                            t4_tmp[:bm, :bi, :bk, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum_(mycc, "maje,miklbecd->ijklabcd", mycc.W_ovov_tc[m0:m1, :, j0:j1, :],
                            t4_tmp[:bm, :bi, :bk, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, i0, i1, j0, j1, l0, l1)
                        einsum_(mycc, "mcke,mijlaebd->ijklabcd", mycc.W_ovov_tc[m0:m1, :, k0:k1, :],
                            t4_tmp[:bm, :bi, :bj, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum_(mycc, "make,mijlcebd->ijklabcd", mycc.W_ovov_tc[m0:m1, :, k0:k1, :],
                            t4_tmp[:bm, :bi, :bj, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, i0, i1, j0, j1, k0, k1)
                        einsum_(mycc, "mdle,mijkaebc->ijklabcd", mycc.W_ovov_tc[m0:m1, :, l0:l1, :],
                            t4_tmp[:bm, :bi, :bj, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum_(mycc, "male,mijkdebc->ijklabcd", mycc.W_ovov_tc[m0:m1, :, l0:l1, :],
                            t4_tmp[:bm, :bi, :bj, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, k0, k1, i0, i1, l0, l1)
                        einsum_(mycc, "mbje,mkilcead->ijklabcd", mycc.W_ovov_tc[m0:m1, :, j0:j1, :],
                            t4_tmp[:bm, :bk, :bi, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum_(mycc, "mcje,mkilbead->ijklabcd", mycc.W_ovov_tc[m0:m1, :, j0:j1, :],
                            t4_tmp[:bm, :bk, :bi, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, l0, l1, i0, i1, k0, k1)
                        einsum_(mycc, "mbje,mlikdeac->ijklabcd", mycc.W_ovov_tc[m0:m1, :, j0:j1, :],
                            t4_tmp[:bm, :bl, :bi, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum_(mycc, "mdje,mlikbeac->ijklabcd", mycc.W_ovov_tc[m0:m1, :, j0:j1, :],
                            t4_tmp[:bm, :bl, :bi, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, j0, j1, i0, i1, l0, l1)
                        einsum_(mycc, "mcke,mjilbead->ijklabcd", mycc.W_ovov_tc[m0:m1, :, k0:k1, :],
                            t4_tmp[:bm, :bj, :bi, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum_(mycc, "mbke,mjilcead->ijklabcd", mycc.W_ovov_tc[m0:m1, :, k0:k1, :],
                            t4_tmp[:bm, :bj, :bi, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, j0, j1, i0, i1, k0, k1)
                        einsum_(mycc, "mdle,mjikbeac->ijklabcd", mycc.W_ovov_tc[m0:m1, :, l0:l1, :],
                            t4_tmp[:bm, :bj, :bi, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum_(mycc, "mble,mjikdeac->ijklabcd", mycc.W_ovov_tc[m0:m1, :, l0:l1, :],
                            t4_tmp[:bm, :bj, :bi, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, l0, l1, i0, i1, j0, j1)
                        einsum_(mycc, "mcke,mlijdeab->ijklabcd", mycc.W_ovov_tc[m0:m1, :, k0:k1, :],
                            t4_tmp[:bm, :bl, :bi, :bj], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum_(mycc, "mdke,mlijceab->ijklabcd", mycc.W_ovov_tc[m0:m1, :, k0:k1, :],
                            t4_tmp[:bm, :bl, :bi, :bj], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, k0, k1, i0, i1, j0, j1)
                        einsum_(mycc, "mdle,mkijceab->ijklabcd", mycc.W_ovov_tc[m0:m1, :, l0:l1, :],
                            t4_tmp[:bm, :bk, :bi, :bj], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum_(mycc, "mcle,mkijdeab->ijklabcd", mycc.W_ovov_tc[m0:m1, :, l0:l1, :],
                            t4_tmp[:bm, :bk, :bi, :bj], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)

                    _accumulate_t4_(mycc, r4, r4_tmp, i0, i1, j0, j1, k0, k1, l0, l1, beta=1.0)
        time2 = log.timer_debug1('t4: iter: P3, P4, P5, P6 [%3d, %3d]:'%(l0, l1), *time2)
    t4_tmp = None
    r4_tmp = None
    time1 = log.timer_debug1('t4: P3, P4, P5, P6', *time1)

    time2 = logger.process_clock(), logger.perf_counter()
    t4_tmp = np.empty((blksize,) * 4 + (nvir,) * 4, dtype=t4.dtype)
    r4_tmp = np.empty((blksize,) * 4 + (nvir,) * 4, dtype=t4.dtype)
    for l0, l1 in lib.prange(0, nocc, blksize):
        bl = l1 - l0
        for k0, k1 in lib.prange(0, l1, blksize):
            bk = k1 - k0
            for j0, j1 in lib.prange(0, k1, blksize):
                bj = j1 - j0
                for i0, i1 in lib.prange(0, j1, blksize):
                    bi = i1 - i0

                    r4_tmp[:] = 0.0
                    for m0, m1 in lib.prange(0, nocc, blksize):
                        bm = m1 - m0
                        for n0, n1 in lib.prange(0, nocc, blksize):
                            bn = n1 - n0

                            # R4: P7 (o6v4 * 6)
                            _unpack_t4_(mycc, t4, t4_tmp, m0, m1, n0, n1, k0, k1, l0, l1)
                            einsum_(mycc, "mnij,mnklabcd->ijklabcd", mycc.W_oooo[m0:m1, n0:n1, i0:i1, j0:j1],
                                t4_tmp[:bm, :bn, :bk, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                            _unpack_t4_(mycc, t4, t4_tmp, m0, m1, n0, n1, j0, j1, l0, l1)
                            einsum_(mycc, "mnik,mnjlacbd->ijklabcd", mycc.W_oooo[m0:m1, n0:n1, i0:i1, k0:k1],
                                t4_tmp[:bm, :bn, :bj, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                            _unpack_t4_(mycc, t4, t4_tmp, m0, m1, n0, n1, j0, j1, k0, k1)
                            einsum_(mycc, "mnil,mnjkadbc->ijklabcd", mycc.W_oooo[m0:m1, n0:n1, i0:i1, l0:l1],
                                t4_tmp[:bm, :bn, :bj, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                            _unpack_t4_(mycc, t4, t4_tmp, m0, m1, n0, n1, i0, i1, l0, l1)
                            einsum_(mycc, "mnjk,mnilbcad->ijklabcd", mycc.W_oooo[m0:m1, n0:n1, j0:j1, k0:k1],
                                t4_tmp[:bm, :bn, :bi, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                            _unpack_t4_(mycc, t4, t4_tmp, m0, m1, n0, n1, i0, i1, k0, k1)
                            einsum_(mycc, "mnjl,mnikbdac->ijklabcd", mycc.W_oooo[m0:m1, n0:n1, j0:j1, l0:l1],
                                t4_tmp[:bm, :bn, :bi, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                            _unpack_t4_(mycc, t4, t4_tmp, m0, m1, n0, n1, i0, i1, j0, j1)
                            einsum_(mycc, "mnkl,mnijcdab->ijklabcd", mycc.W_oooo[m0:m1, n0:n1, k0:k1, l0:l1],
                                t4_tmp[:bm, :bn, :bi, :bj], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)

                    _accumulate_t4_(mycc, r4, r4_tmp, i0, i1, j0, j1, k0, k1, l0, l1, beta=1.0)
        time2 = log.timer_debug1('t4: iter: P7 [%3d, %3d]:'%(l0, l1), *time2)
    t4_tmp = None
    r4_tmp = None
    time1 = log.timer_debug1('t4: P7', *time1)
    return r4

def r4_tril_divide_e_(mycc, r4, mo_energy):
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    blksize = mycc.blksize

    eia = mo_energy[: nocc, None] - mo_energy[None, nocc :] - mycc.level_shift
    eijklabcd_blk = np.empty((blksize,) * 4 + (nvir,) * 4, dtype=r4.dtype)
    r4_tmp = np.empty((blksize,) * 4 + (nvir,) * 4, dtype=r4.dtype)
    for l0, l1 in lib.prange(0, nocc, blksize):
        bl = l1 - l0
        for k0, k1 in lib.prange(0, l1, blksize):
            bk = k1 - k0
            for j0, j1 in lib.prange(0, k1, blksize):
                bj = j1 - j0
                for i0, i1 in lib.prange(0, j1, blksize):
                    bi = i1 - i0
                    eijklabcd_blk = (eia[i0:i1, None, None, None, :, None, None, None]
                                + eia[None, j0:j1, None, None, None, :, None, None]
                                + eia[None, None, k0:k1, None, None, None, :, None]
                                + eia[None, None, None, l0:l1, None, None, None, :])
                    _unpack_t4_(mycc, r4, r4_tmp, i0, i1, j0, j1, k0, k1, l0, l1)
                    r4_tmp[:bi, :bj, :bk, :bl] /= eijklabcd_blk
                    _accumulate_t4_(mycc, r4, r4_tmp, i0, i1, j0, j1, k0, k1, l0, l1)
    eijklabcd_blk = None
    r4_tmp = None

def update_amps_rccsdtq_tril_(mycc, tamps, eris):
    '''Update RCCSDTQ amplitudes in place, with T4 amplitudes stored in triangular form.'''
    assert (isinstance(eris, _PhysicistsERIs))
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    t1, t2, t3, t4 = tamps
    nocc4 = t4.shape[0]
    mo_energy = eris.mo_energy.copy()

    # t1, t2
    update_t1_fock_eris_(mycc, t1, eris)
    time1 = log.timer_debug1('update fock and eris', *time0)
    intermediates_t1t2_(mycc, t2)
    time1 = log.timer_debug1('t1t2: update intermediates', *time1)
    r1, r2 = compute_r1r2(mycc, t2)
    r1r2_add_t3_(mycc, t3, r1, r2)
    r2_add_t4_tril_(mycc, t4, r2)
    time1 = log.timer_debug1('t1t2: compute r1 & r2', *time1)
    # symmetrize R2
    r2 += r2.transpose(1, 0, 3, 2)
    time1 = log.timer_debug1('t1t2: symmetrize r2', *time1)
    # divide by eijkabc
    r1r2_divide_e_(mycc, r1, r2, mo_energy)
    time1 = log.timer_debug1('t1t2: divide r1 & r2 by eia & eijab', *time1)

    res_norm = [np.linalg.norm(r1), np.linalg.norm(r2)]

    t1 += r1
    t2 += r2
    time1 = log.timer_debug1('t1t2: update t1 & t2', *time1)
    time0 = log.timer_debug1('t1t2 total', *time0)

    # t3
    intermediates_t3_(mycc, t2)
    intermediates_t3_add_t3_(mycc, t3)
    time1 = log.timer_debug1('t3: update intermediates', *time0)
    r3 = compute_r3(mycc, t2, t3)
    r3_add_t4_tril_(mycc, t4, r3)
    time1 = log.timer_debug1('t3: compute r3', *time1)
    # symmetrization
    t3_perm_symmetrize_inplace_(r3, nocc, nvir, 1.0, 0.0)
    t3_spin_summation_inplace_(r3, nocc**3, nvir, "P3_full", -1.0 / 6.0, 1.0)
    purify_tamps_(r3)
    time1 = log.timer_debug1('t3: symmetrize r3', *time1)
    # divide by eijkabc
    r3_divide_e_(mycc, r3, mo_energy)
    time1 = log.timer_debug1('t3: divide r3 by eijkabc', *time1)

    res_norm.append(np.linalg.norm(r3))

    t3 += r3
    r3 = None
    time1 = log.timer_debug1('t3: update t3', *time1)
    time0 = log.timer_debug1('t3 total', *time0)

    # t4
    intermediates_t4_tril_(mycc, t2, t3, t4)
    mycc.t1_eris = None
    time1 = log.timer_debug1('t4: update intermediates', *time0)
    r4 = compute_r4_tril(mycc, t2, t3, t4)
    time1 = log.timer_debug1('t4: compute r4', *time1)
    # symmetrize r4
    symmetrize_tamps_tril_(r4, nocc)
    t4_spin_summation_inplace_c(r4, nocc4, nvir, "P4_full", -1.0 / 24.0, 1.0)
    purify_tamps_tril_(r4, nocc)
    time1 = log.timer_debug1('t4: symmetrize r4', *time1)
    # divide by eijkabc
    r4_tril_divide_e_(mycc, r4, mo_energy)
    time1 = log.timer_debug1('t4: divide r4 by eijklabcd', *time1)

    res_norm.append(np.linalg.norm(r4))

    # t4 += r4
    t4_add_(t4, r4, nocc4, nvir)
    r4 = None
    time1 = log.timer_debug1('t4: update t4', *time1)
    time0 = log.timer_debug1('t4 total', *time0)
    return res_norm

def memory_estimate_log_rccsdtq(mycc):
    '''Estimate the memory cost (in MB).'''
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc

    log.info('Approximate memory usage estimate')
    if mycc.do_tril_maxT:
        nocc4 = nocc * (nocc + 1) * (nocc + 2) * (nocc + 3) // 24
        t4_memory = nocc4 * nvir**4 * 8 / 1024**2
    else:
        t4_memory = nocc**4 * nvir**4 * 8 / 1024**2
    log.info('T4 memory               %9.5e MB', t4_memory)
    log.info('R4 memory               %9.5e MB', t4_memory)
    if not mycc.do_tril_maxT:
        log.info('Symmetrized T4 memory   %9.5e MB', t4_memory)
    if mycc.einsum_backend in ['numpy', 'pyscf']:
        log.info("T4 einsum buffer        %9.5e MB", t4_memory)
    eris_memory = nmo**4 * 8 / 1024**2
    log.info('ERIs memory             %9.5e MB', eris_memory)
    log.info('T1-ERIs memory          %9.5e MB', eris_memory)
    intermediates_memory = nocc**2 * nvir**4 * 8 / 1024**2 * 2
    log.info('Intermediates memory    %9.5e MB', intermediates_memory)
    if mycc.do_tril_maxT:
        blk_memory = mycc.blksize**4 * nvir**4 * 8 / 1024**2 * 2
        log.info("Block workspace         %9.5e MB", blk_memory)
    if mycc.incore_complete:
        if mycc.do_diis_maxT:
            diis_memory = (nocc * (nocc + 1) * (nocc + 2) * (nocc + 3) // 6 * nvir**4 * 8 / 1024**2
                            * mycc.diis_space * 2)
        else:
            diis_memory = nocc * (nocc + 1) * (nocc + 2) // 6 * nvir**3 * 8 / 1024**2 * mycc.diis_space * 2
        log.info('DIIS memory             %9.5e MB', diis_memory)
    else:
        diis_memory = 0.0
    if mycc.do_tril_maxT:
        total_memory = 2 * t4_memory + 3 * eris_memory + diis_memory + blk_memory
    else:
        total_memory = 3 * t4_memory + 3 * eris_memory + diis_memory
    log.info('Total estimated memory  %9.5e MB', total_memory)
    max_memory = mycc.max_memory - lib.current_memory()[0]
    if total_memory > max_memory:
        logger.warn(mycc, 'Estimated memory usage exceeds the allowed limit for %s', mycc.__class__.__name__)
        logger.warn(mycc, 'The calculation may run out of memory')
        if mycc.incore_complete:
            if mycc.do_diis_maxT:
                logger.warn(mycc, 'Consider setting `do_diis_maxT = False` to reduce memory usage')
            else:
                logger.warn(mycc, 'Consider setting `incore_complete = False` to reduce memory usage')
        if not mycc.do_tril_maxT:
            logger.warn(mycc, 'Consider using %s in `pyscf.cc.rccsdtq` which stores the triangular T amplitudes',
                        mycc.__class__.__name__)
        else:
            logger.warn(mycc, 'Consider reducing `blksize` to reduce memory usage')
    return mycc

def dump_chk(mycc, tamps=None, frozen=None, mo_coeff=None, mo_occ=None):
    if not mycc.chkfile:
        return mycc
    if tamps is None: tamps = mycc.tamps
    if frozen is None: frozen = mycc.frozen
    # "None" cannot be serialized by the chkfile module
    if frozen is None:
        frozen = 0
    cc_chk = {'e_corr': mycc.e_corr, 'tamps': tamps, 'frozen': frozen}
    if mo_coeff is not None: cc_chk['mo_coeff'] = mo_coeff
    if mo_occ is not None: cc_chk['mo_occ'] = mo_occ
    if mycc._nmo is not None: cc_chk['_nmo'] = mycc._nmo
    if mycc._nocc is not None: cc_chk['_nocc'] = mycc._nocc
    if mycc.do_tril_maxT:
        lib.chkfile.save(mycc.chkfile, 'rccsdtq', cc_chk)
    else:
        lib.chkfile.save(mycc.chkfile, 'rccsdtq_highm', cc_chk)


class RCCSDTQ(rccsdt.RCCSDT):

    conv_tol = getattr(__config__, 'cc_rccsdtq_RCCSDTQ_conv_tol', 1e-7)
    conv_tol_normt = getattr(__config__, 'cc_rccsdtq_RCCSDTQ_conv_tol_normt', 1e-6)
    cc_order = getattr(__config__, 'cc_rccsdtq_RCCSDTQ_cc_order', 4)
    tamps = getattr(__config__, 'cc_rccsdtq_RCCSDTQ_tamps', [None, None, None, None])
    unique_tamps_map = getattr(__config__, 'cc_rccsdtq_RCCSDTQ_unique_tamps_map', None)
    do_tril_maxT = getattr(__config__, 'cc_rccsdtq_RCCSDTQ_do_tril_maxT', True)
    do_diis_maxT = getattr(__config__, 'cc_rccsdtq_RCCSDTQ_do_diis_maxT', True)
    blksize = getattr(__config__, 'cc_rccsdtq_RCCSDTQ_blksize', 4)

    @property
    def t4(self):
        return self.tamps[3]

    @t4.setter
    def t4(self, val):
        self.tamps[3] = val

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        rccsdt.RCCSDT.__init__(self, mf, frozen, mo_coeff, mo_occ)

    memory_estimate_log = memory_estimate_log_rccsdtq
    update_amps_ = update_amps_rccsdtq_tril_
    dump_chk = dump_chk

    def kernel(self, tamps=None, eris=None):
        return self.ccsdtq(tamps, eris)

    def ccsdtq(self, tamps=None, eris=None):
        log = logger.Logger(self.stdout, self.verbose)

        assert (self.mo_coeff is not None)
        assert (self.mo_occ is not None)

        assert self.mo_coeff.dtype == np.float64, "`mo_coeff` must be float64"

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        self.e_hf = self.get_e_hf()

        if eris is None:
            eris = self.ao2mo(self.mo_coeff)

        nocc = eris.nocc

        if self.do_tril_maxT:
            self.setup_tril2cube_()

            self.blksize = min(self.blksize, (nocc + 1) // 2)
            log.info('blksize %2d' % (self.blksize))

        self.memory_estimate_log()
        self.build_unique_tamps_map_()

        self.converged, self.e_corr, self.tamps = \
                kernel(self, eris, tamps, max_cycle=self.max_cycle,
                       tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                       verbose=self.verbose, callback=self.callback)
        self._finalize()
        return self.e_corr, self.tamps


if __name__ == "__main__":

    from pyscf import gto, scf

    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto3g", verbose=3)
    mf = scf.RHF(mol)
    mf.level_shift = 0.0
    mf.conv_tol = 1e-14
    mf.max_cycle = 1000
    mf.kernel()
    print()
    ref_ecorr = -0.157579406507473
    frozen = 0
    mycc = RCCSDTQ(mf, frozen=frozen)
    mycc.set_einsum_backend('numpy')
    mycc.conv_tol = 1e-12
    mycc.conv_tol_normt = 1e-10
    mycc.max_cycle = 100
    mycc.verbose = 5
    mycc.do_diis_maxT = True
    mycc.incore_complete = True
    mycc.kernel()
    print("E_corr: % .10f    Ref: % .10f    Diff: % .10e"%(mycc.e_corr, ref_ecorr, mycc.e_corr - ref_ecorr))
    print('\n' * 5)

    # comparison with the high-memory version
    from rccsdtq_highm import RCCSDTQ as RCCSDTQhm
    mycc2 = RCCSDTQhm(mf, frozen=frozen)
    mycc2.set_einsum_backend('numpy')
    mycc2.conv_tol = 1e-12
    mycc2.conv_tol_normt = 1e-10
    mycc2.max_cycle = 100
    mycc2.verbose = 5
    mycc2.do_diis_maxT = True
    mycc2.incore_complete = True
    mycc2.kernel()
    print("E_corr: % .10f    Ref: % .10f    Diff: % .10e"%(mycc2.e_corr, ref_ecorr, mycc2.e_corr - ref_ecorr))
    print()

    t4_tril = mycc.t4
    t4_full = mycc2.t4
    t4_tril_from_t4_full = mycc2.tamp_full2tril(t4_full)
    t4_full_from_t4_tril = mycc.tamp_tril2full(t4_tril)

    print('energy difference                          % .10e' % (mycc.e_tot - mycc2.e_tot))
    print('max(abs(t1 difference))                    % .10e' % np.max(np.abs(mycc.t1 - mycc2.t1)))
    print('max(abs(t2 difference))                    % .10e' % np.max(np.abs(mycc.t2 - mycc2.t2)))
    print('max(abs(t3 difference))                    % .10e' % np.max(np.abs(mycc.t3 - mycc2.t3)))
    print('max(abs(t4_tril - t4_tril_from_t4_full))   % .10e' % np.max(np.abs(t4_tril - t4_tril_from_t4_full)))
    print('max(abs(t4_full - t4_full_from_t4_tril))   % .10e' % np.max(np.abs(t4_full - t4_full_from_t4_tril)))
