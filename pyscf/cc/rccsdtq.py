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
RHF-CCSDTQ with T4 amplitudes stored only for the i <= j <= k <= l index combinations

Ref:
JCP 142, 064108 (2015); DOI:10.1063/1.4907278
'''

import numpy as np
import numpy
import ctypes
from pyscf import lib
from pyscf.lib import logger
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask, get_e_hf, _mo_without_core
from pyscf.cc import ccsd, _ccsd
from pyscf.cc.rccsdt import (einsum_, _make_eris_incore_rccsdt, _make_df_eris_incore_rccsdt, t3_symm_ip,
                            update_t1_fock_eris, init_amps, energy, intermediates_t1t2, compute_r1r2, r1r2_divide_e,
                            intermediates_t3, run_diis, amplitudes_to_vector_rhf, vector_to_amplitudes_rhf,
                            ao2mo_rccsdt, _finalize)
from pyscf.cc.rccsdt_highm import (t3_symm, t3_p_sum_ip, rt_purify, r1r2_add_t3, intermediates_t3_add_t3, compute_r3,
                                    r3_divide_e)


def t4_symm_ip(A, nocc4, nvir, pattern, alpha=1.0, beta=0.0):
    assert A.dtype == np.float64 and A.flags['C_CONTIGUOUS'], "A must be a contiguous float64 array"

    pattern_c = pattern.encode('utf-8')

    drv = _ccsd.libcc.t4_symm_ip_c
    drv(
        A.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int64(nocc4),
        ctypes.c_int64(nvir),
        ctypes.c_char_p(pattern_c),
        ctypes.c_double(alpha),
        ctypes.c_double(beta)
    )
    return A

def t4_add(t4, r4, nocc4, nvir):
    assert t4.dtype == np.float64 and t4.flags['C_CONTIGUOUS'], "t4 must be a contiguous float64 array"
    assert r4.dtype == np.float64 and r4.flags['C_CONTIGUOUS'], "r4 must be a contiguous float64 array"

    drv = _ccsd.libcc.t4_add_c
    drv(
        t4.ctypes.data_as(ctypes.c_void_p),
        r4.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int64(nocc4),
        ctypes.c_int64(nvir),
    )
    return t4

def call_unpack_24fold_c(t4, t4_blk, map_, mask, i0, i1, j0, j1, k0, k1, l0, l1,
                            nocc, nvir, blk_i, blk_j, blk_k, blk_l):
    assert t4.dtype == np.float64 and t4_blk.dtype == np.float64
    assert map_.dtype == np.int64 and mask.dtype == np.bool_

    # Ensure arrays are contiguous
    t4_c = np.ascontiguousarray(t4)
    t4_blk_c = np.ascontiguousarray(t4_blk)
    map_c = np.ascontiguousarray(map_)
    mask_c = np.ascontiguousarray(mask)

    drv = _ccsd.libcc.unpack_24fold_c
    drv(
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

def call_update_packed_24fold_c(t4, t4_blk, map_, i0, i1, j0, j1, k0, k1, l0, l1,
                                nocc, nvir, blk_i, blk_j, blk_k, blk_l, alpha, beta):
    assert t4.dtype == np.float64 and t4_blk.dtype == np.float64
    assert map_.dtype == np.int64

    # Ensure arrays are contiguous
    t4_c = np.ascontiguousarray(t4)
    t4_blk_c = np.ascontiguousarray(t4_blk)
    map_c = np.ascontiguousarray(map_)

    drv = _ccsd.libcc.update_packed_24fold_c
    drv(
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

def setup_tril2cube_t4(mycc):
    nocc = mycc.nocc
    nocc4 = nocc * (nocc + 1) * (nocc + 2) * (nocc + 3) // 24
    tril2cube_map = np.zeros((24, nocc, nocc, nocc, nocc), dtype=np.int64)
    tril2cube_mask = np.zeros((24, nocc, nocc, nocc, nocc), dtype=np.bool_)
    tril2cube_tp = []

    i, j, k, l = np.meshgrid(np.arange(nocc), np.arange(nocc), np.arange(nocc), np.arange(nocc), indexing='ij')
    t4_map = np.where((i <= j) & (j <= k) & (k <= l))

    import itertools
    perms = list(itertools.permutations([0, 1, 2, 3]))
    for idx, perm in enumerate(perms):
        tril2cube_map[idx, t4_map[perm[0]], t4_map[perm[1]], t4_map[perm[2]], t4_map[perm[3]]] = np.arange(nocc4)

    labels = ('i', 'j', 'k', 'l')
    collect_relation = {('i', 'j'), ('i', 'k'), ('i', 'l'), ('j', 'k'), ('j', 'l'), ('k', 'l')}
    var_map = {'i': i, 'j': j, 'k': k, 'l': l}
    for idx, perm in enumerate(perms):
        indices = np.argsort(perm)
        vars_sorted = [var_map[labels[indices[i]]] for i in range(4)]
        comparisons = []
        for comparison_idx in range(3):
            left_label = labels[indices[comparison_idx]]
            right_label = labels[indices[comparison_idx + 1]]
            if (right_label, left_label) in collect_relation:
                comparisons.append(vars_sorted[comparison_idx] < vars_sorted[comparison_idx + 1])
            else:
                comparisons.append(vars_sorted[comparison_idx] <= vars_sorted[comparison_idx + 1])
        tril2cube_mask[idx] = comparisons[0] & comparisons[1] & comparisons[2]

    for idx, perm in enumerate(perms):
        tril2cube_tp.append((0,) + tuple([p + 1 for p in perm]))

    mycc.tril2cube_map = tril2cube_map
    mycc.tril2cube_mask = tril2cube_mask
    mycc.tril2cube_tp = tril2cube_tp
    return mycc

def _unpack_24fold(mycc, t4, t4_blk, i0, i1, j0, j1, k0, k1, l0, l1,
                    blksize0=None, blksize1=None, blksize2=None, blksize3=None):
    if blksize0 is None: blksize0 = mycc.blksize
    if blksize1 is None: blksize1 = mycc.blksize
    if blksize2 is None: blksize2 = mycc.blksize
    if blksize3 is None: blksize3 = mycc.blksize
    call_unpack_24fold_c(t4, t4_blk, mycc.tril2cube_map, mycc.tril2cube_mask, i0, i1, j0, j1, k0, k1, l0, l1,
                        mycc.nocc, mycc.nvir, blksize0, blksize1, blksize2, blksize3)

def _update_packed_24fold(mycc, t4, t4_blk, i0, i1, j0, j1, k0, k1, l0, l1,
                    blksize0=None, blksize1=None, blksize2=None, blksize3=None, alpha=1.0, beta=0.0):
    if blksize0 is None: blksize0 = mycc.blksize
    if blksize1 is None: blksize1 = mycc.blksize
    if blksize2 is None: blksize2 = mycc.blksize
    if blksize3 is None: blksize3 = mycc.blksize
    call_update_packed_24fold_c(t4, t4_blk, mycc.tril2cube_map, i0, i1, j0, j1, k0, k1, l0, l1, mycc.nocc, mycc.nvir,
                                blksize0, blksize1, blksize2, blksize3, alpha=alpha, beta=beta)

def rt_symmetrize_t4_tril(r):
    """
    Enforce permutation symmetry of r in the special cases:

    - i = j < k <= l : symmetrize over (a, b)
    - i < j = k < l : symmetrize over (b, c)
    - i <= j < k = l : symmetrize over (c, d)

    The cases where three or more indices equal are excluded
    since those entries are set to zero elsewhere.
    """
    import itertools, numpy as np
    nocc4 = r.shape[0]
    nocc = int((24 * nocc4) ** (1 / 4))
    if nocc * (nocc + 1) * (nocc + 2) * (nocc + 3) // 24 < nocc4: nocc += 1
    if nocc * (nocc + 1) * (nocc + 2) * (nocc + 3) // 24 > nocc4: nocc -= 1
    assert nocc * (nocc + 1) * (nocc + 2) * (nocc + 3) // 24 == nocc4, "the size of r might be incorrect"

    quadruples = np.array([(i, j, k, l) for i in range(nocc) for j in range(i, nocc)
                                        for k in range(j, nocc) for l in range(k, nocc)])
    assert quadruples.shape[0] == nocc4, "the size of r might be incorrect"

    case_1_idx = np.where((quadruples[:, 0] == quadruples[:, 1]) & (quadruples[:, 1] < quadruples[:, 2])
                            & (quadruples[:, 2] <= quadruples[:, 3]))[0]
    r[case_1_idx, ...] = (r[case_1_idx, ...] + r[case_1_idx, ...].transpose(0, 2, 1, 3, 4)) / 2

    case_2_idx = np.where((quadruples[:, 0] < quadruples[:, 1]) & (quadruples[:, 1] == quadruples[:, 2])
                            & (quadruples[:, 2] < quadruples[:, 3]))[0]
    r[case_2_idx, ...] = (r[case_2_idx, ...] + r[case_2_idx, ...].transpose(0, 1, 3, 2, 4)) / 2

    case_3_idx = np.where((quadruples[:, 0] <= quadruples[:, 1]) & (quadruples[:, 1] < quadruples[:, 2])
                            & (quadruples[:, 2] == quadruples[:, 3]))[0]
    r[case_3_idx, ...] = (r[case_3_idx, ...] + r[case_3_idx, ...].transpose(0, 1, 2, 4, 3)) / 2

def rt_purify_t4_tril(r):
    """
    Set all entries with three or more of (i, j, k, l) or (a, b, c, d) equal to zero
    """
    import itertools, numpy as np
    n = 4
    nocc4 = r.shape[0]
    nocc = int((24 * nocc4) ** (1 / 4))
    if nocc * (nocc + 1) * (nocc + 2) * (nocc + 3) // 24 < nocc4: nocc += 1
    if nocc * (nocc + 1) * (nocc + 2) * (nocc + 3) // 24 > nocc4: nocc -= 1
    assert nocc * (nocc + 1) * (nocc + 2) * (nocc + 3) // 24 == nocc4, "the size of r might be incorrect"

    quadruples = np.array([(i, j, k, l) for i in range(nocc) for j in range(i, nocc)
                            for k in range(j, nocc) for l in range(k, nocc)])
    assert quadruples.shape[0] == nocc4, "the size of r might be incorrect"
    # ijk
    diag_idx = np.where((quadruples[:, 0] == quadruples[:, 1]) & (quadruples[:, 1] == quadruples[:, 2]))[0]
    r[diag_idx, ...] = 0.0
    # jkl
    diag_idx = np.where((quadruples[:, 1] == quadruples[:, 2]) & (quadruples[:, 2] == quadruples[:, 3]))[0]
    r[diag_idx, ...] = 0.0

    # Set all entries with a=b=c=d to zero
    for perm in itertools.combinations(range(n), 3):
        idxr = [slice(None)] * n
        for p in perm:
            idxr[p] = np.mgrid[:r.shape[p + 1]]
        r[(slice(None), ) * 1 + tuple(idxr)] = 0.0

def r2_add_t4_tril(mycc, t4, r2):
    nocc, nvir = mycc.nocc, mycc.nvir
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
                    _unpack_24fold(mycc, t4, t4_tmp, m0, m1, n0, n1, i0, i1, j0, j1)
                    t4_symm_ip(t4_tmp, blksize**4, nvir, "ccnn", 1.0, 0.0)
                    einsum_('mnef,mnijefab->ijab', mycc.t1_eris[m0:m1, n0:n1, nocc:, nocc:],
                        t4_tmp[:bm, :bn, :bi, :bj], out=r2[i0:i1, j0:j1, :, :], alpha=0.25, beta=1.0)
    t4_tmp = None
    return r2

def update_amps_t1t2_with_t3t4_tril(mycc, tamps):
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    t1, t2, t3, t4 = tamps

    update_t1_fock_eris(mycc, t1)

    intermediates_t1t2(mycc, t2)
    time1 = log.timer_debug1('t1t2: update intermediates', *time0)

    r1, r2 = compute_r1r2(mycc, t2)
    r1, r2 = r1r2_add_t3(mycc, t3, r1, r2)
    r2 = r2_add_t4_tril(mycc, t4, r2)

    # divide by eijkabc
    r1, r2 = r1r2_divide_e(mycc, r1, r2)
    # symmetrize R2
    r2 += r2.transpose(1, 0, 3, 2)

    mycc.r_norm[0] = np.linalg.norm(r1)
    mycc.r_norm[1] = np.linalg.norm(r2)

    t1 += r1
    t2 += r2
    time1 = log.timer_debug1('t1t2: update t1 & t2', *time1)
    time0 = log.timer_debug1('t1t2 total', *time0)
    return t1, t2

def r3_add_t4_tril(mycc, t4, r3):
    '''Add t4_tril contributions to r3'''
    nocc, nvir = mycc.nocc, mycc.nvir
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
                    _unpack_24fold(mycc, t4, t4_tmp, m0, m1, i0, i1, j0, j1, k0, k1)
                    t4_symm_ip(t4_tmp, blksize**4, nvir, "cnnn", 1.0, 0.0)
                    # R3: P9
                    einsum_('me,mijkeabc->ijkabc', t1_fock[m0:m1, nocc:], t4_tmp[:bm, :bi, :bj, :bk],
                            out=r3[i0:i1, j0:j1, k0:k1, ...], alpha=1.0 / 6.0, beta=1.0)
                    # R3: P10
                    einsum_('amef,mijkfebc->ijkabc', t1_eris[nocc:, m0:m1, nocc:, nocc:],
                        t4_tmp[:bm, :bi, :bj, :bk], out=r3[i0:i1, j0:j1, k0:k1, ...], alpha=0.5, beta=1.0)
                    # R3: P11
                    einsum_('mjen,mijkeabc->inkabc', t1_eris[m0:m1, j0:j1, nocc:, :nocc],
                        t4_tmp[:bm, :bi, :bj, :bk], out=r3[i0:i1, :, k0:k1, ...], alpha=-0.5, beta=1.0)
    t4_tmp = None
    return r3

def update_amps_t3_with_t4_tril(mycc, tamps):
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    nocc, nvir = mycc.nocc, mycc.nvir

    _, t2, t3, t4 = tamps

    intermediates_t3(mycc, t2)
    intermediates_t3_add_t3(mycc, t3)
    time1 = log.timer_debug1('t3: update intermediates', *time0)

    r3 = compute_r3(mycc, t2, t3)
    r3 = r3_add_t4_tril(mycc, t4, r3)
    time1 = log.timer_debug1('t3: compute r3', *time1)
    # divide by eijkabc
    r3 = r3_divide_e(mycc, r3)
    time1 = log.timer_debug1('t3: divide r3 by eijkabc', *time1)
    # symmetrization
    t3_p_sum_ip(r3, nocc, nvir, 1.0, 0.0)
    t3_symm_ip(r3, nocc**3, nvir, "111111", -1.0 / 6.0, 1.0)
    rt_purify(r3)
    time1 = log.timer_debug1('t3: symmetrize r3', *time1)

    mycc.r_norm[2] = np.linalg.norm(r3)

    t3 += r3
    r3 = None
    time1 = log.timer_debug1('t3: update t3', *time1)
    time0 = log.timer_debug1('t3 total', *time0)
    return t3

def intermediates_t4_tril(mycc, t2, t3, t4):
    nocc, nvir = mycc.nocc, mycc.nvir
    blksize = mycc.blksize
    t1_fock, t1_eris = mycc.t1_fock, mycc.t1_eris

    # FIXME
    c_t2 = 2.0 * t2 - t2.transpose(0, 1, 3, 2)
    c_t3 = np.empty_like(t3)
    t3_symm(t3, c_t3, nocc**3, nvir, "20-100-1", 1.0, 0.0)

    einsum_('me,mjab->abej', t1_fock[:nocc, nocc:], t2, out=mycc.W_vvvo_tc, alpha=-1.0, beta=1.0)

    W_ovvvoo = np.empty((nocc,) + (nvir,) * 3 + (nocc,) * 2)
    einsum_('maef,jibf->mabeij', t1_eris[:nocc, nocc:, nocc:, nocc:], t2, out=W_ovvvoo, alpha=2.0, beta=0.0)
    einsum_('mafe,jibf->mabeij', t1_eris[:nocc, nocc:, nocc:, nocc:], t2, out=W_ovvvoo, alpha=-1.0, beta=1.0)
    einsum_('mnei,njab->mabeij', t1_eris[:nocc, :nocc, nocc:, :nocc], t2, out=W_ovvvoo, alpha=-2.0, beta=1.0)
    einsum_('nmei,njab->mabeij', t1_eris[:nocc, :nocc, nocc:, :nocc], t2, out=W_ovvvoo, alpha=1.0, beta=1.0)
    einsum_('nmfe,nijfab->mabeij', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t3, out=W_ovvvoo, alpha=0.5, beta=1.0)
    einsum_('mnfe,nijfab->mabeij', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t3, out=W_ovvvoo, alpha=-0.25, beta=1.0)
    W_ovvvoo += W_ovvvoo.transpose(0, 2, 1, 3, 5, 4)
    mycc.W_ovvvoo = W_ovvvoo
    c_t3 = None

    W_ovvovo = np.empty((nocc, nvir, nvir, nocc, nvir, nocc))
    einsum_('mafe,jibf->mabiej', t1_eris[:nocc, nocc:, nocc:, nocc:], t2, out=W_ovvovo, alpha=1.0, beta=0.0)
    einsum_('mnie,njab->mabiej', t1_eris[:nocc, :nocc, :nocc, nocc:], t2, out=W_ovvovo, alpha=-1.0, beta=1.0)
    einsum_('nmef,injfab->mabiej', t1_eris[:nocc, :nocc, nocc:, nocc:], t3, out=W_ovvovo, alpha=-0.5, beta=1.0)
    mycc.W_ovvovo = W_ovvovo

    W_vooooo = np.empty((nvir,) + (nocc,) * 5)
    einsum_('mnek,ijae->amnijk', t1_eris[:nocc, :nocc, nocc:, :nocc], t2, out=W_vooooo, alpha=1.0, beta=0.0)
    einsum_('mnef,ijkaef->amnijk', t1_eris[:nocc, :nocc, nocc:, nocc:], t3, out=W_vooooo, alpha=0.5, beta=1.0)
    W_vooooo += W_vooooo.transpose(0, 2, 1, 3, 5, 4)
    mycc.W_vooooo = W_vooooo

    W_vvoooo = np.empty((nvir,) * 2 + (nocc,) * 4)
    einsum_('amef,ijkebf->abmijk', t1_eris[nocc:, :nocc, nocc:, nocc:], t3, out=W_vvoooo, alpha=1.0, beta=0.0)
    # FIXME: Find an alternative way of calculating this term
    W_ovvo_c = t1_eris[:nocc, nocc:, nocc:, :nocc].copy()
    einsum_('nmfe,nifa->maei', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t2, out=W_ovvo_c, alpha=1.0, beta=1.0)
    einsum_('mnfe,nifa->maei', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t2, out=W_ovvo_c, alpha=-0.5, beta=1.0)
    einsum_('nmef,infa->maei', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=W_ovvo_c, alpha=-0.5, beta=1.0)
    #
    einsum_('maei,jkbe->abmijk', W_ovvo_c, t2, out=W_vvoooo, alpha=1.0, beta=1.0)
    einsum_('make,jibe->abmijk', mycc.W_ovov_tc, t2, out=W_vvoooo, alpha=1.0, beta=1.0)
    einsum_('mnki,njab->abmijk', mycc.W_oooo, t2, out=W_vvoooo, alpha=-0.5, beta=1.0)

    W_vvvvoo = np.empty((nvir,) * 4 + (nocc,) * 2)
    einsum_('abef,jkfc->abcejk', mycc.W_vvvv_tc, t2, out=W_vvvvoo, alpha=0.5, beta=0.0)

    t4_tmp = np.empty((blksize,) * 4 + (nvir,) * 4, dtype=t4.dtype)
    for n0, n1 in lib.prange(0, nocc, blksize):
        bn = n1 - n0
        for i0, i1 in lib.prange(0, nocc, blksize):
            bi = i1 - i0
            for j0, j1 in lib.prange(0, nocc, blksize):
                bj = j1 - j0
                for k0, k1 in lib.prange(0, nocc, blksize):
                    bk = k1 - k0
                    _unpack_24fold(mycc, t4, t4_tmp, n0, n1, i0, i1, j0, j1, k0, k1)
                    t4_symm_ip(t4_tmp, blksize**4, nvir, "cnnn", 1.0, 0.0)
                    einsum_('mnef,nijkfabe->abmijk', t1_eris[:nocc, n0:n1, nocc:, nocc:],
                        t4_tmp[:bn, :bi, :bj, :bk], out=W_vvoooo[..., i0:i1, j0:j1, k0:k1], alpha=0.5, beta=1.0)
                    einsum_('inef,nijkfabc->abcejk', t1_eris[i0:i1, n0:n1, nocc:, nocc:],
                        t4_tmp[:bn, :bi, :bj, :bk], out=W_vvvvoo[..., j0:j1, k0:k1], alpha=-0.5, beta=1.0)
    t4_tmp = None

    W_vvoooo += W_vvoooo.transpose(1, 0, 2, 4, 3, 5)
    W_vvvvoo += W_vvvvoo.transpose(0, 2, 1, 3, 5, 4)

    mycc.W_vvoooo = W_vvoooo
    mycc.W_vvvvoo = W_vvvvoo
    return mycc

def compute_r4_tril(mycc, t2, t3, t4):
    time1 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    nocc, nvir, blksize = mycc.nocc, mycc.nvir, mycc.blksize
    # FIXME
    c_t3 = np.empty_like(t3)
    t3_symm(t3, c_t3, nocc**3, nvir, "20-100-1", 1.0, 0.0)

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
                    einsum_("abej,iklecd->ijklabcd", mycc.W_vvvo_tc[..., j0:j1], t3[i0:i1, k0:k1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=0.0)
                    einsum_("acek,ijlebd->ijklabcd", mycc.W_vvvo_tc[..., k0:k1], t3[i0:i1, j0:j1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("adel,ijkebc->ijklabcd", mycc.W_vvvo_tc[..., l0:l1], t3[i0:i1, j0:j1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("baei,jklecd->ijklabcd", mycc.W_vvvo_tc[..., i0:i1], t3[j0:j1, k0:k1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("caei,kjlebd->ijklabcd", mycc.W_vvvo_tc[..., i0:i1], t3[k0:k1, j0:j1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("daei,ljkebc->ijklabcd", mycc.W_vvvo_tc[..., i0:i1], t3[l0:l1, j0:j1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("bcek,jilead->ijklabcd", mycc.W_vvvo_tc[..., k0:k1], t3[j0:j1, i0:i1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("bdel,jikeac->ijklabcd", mycc.W_vvvo_tc[..., l0:l1], t3[j0:j1, i0:i1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("cbej,kilead->ijklabcd", mycc.W_vvvo_tc[..., j0:j1], t3[k0:k1, i0:i1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("dbej,likeac->ijklabcd", mycc.W_vvvo_tc[..., j0:j1], t3[l0:l1, i0:i1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("cdel,kijeab->ijklabcd", mycc.W_vvvo_tc[..., l0:l1], t3[k0:k1, i0:i1, j0:j1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("dcek,lijeab->ijklabcd", mycc.W_vvvo_tc[..., k0:k1], t3[l0:l1, i0:i1, j0:j1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)

                    # R4: P1 (o5v4 * 12)
                    einsum_("amij,mklbcd->ijklabcd", mycc.W_vooo_tc[:, :, i0:i1, j0:j1],
                        t3[:, k0:k1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("amik,mjlcbd->ijklabcd", mycc.W_vooo_tc[:, :, i0:i1, k0:k1],
                        t3[:, j0:j1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("amil,mjkdbc->ijklabcd", mycc.W_vooo_tc[:, :, i0:i1, l0:l1],
                        t3[:, j0:j1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("bmji,mklacd->ijklabcd", mycc.W_vooo_tc[:, :, j0:j1, i0:i1],
                        t3[:, k0:k1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("cmki,mjlabd->ijklabcd", mycc.W_vooo_tc[:, :, k0:k1, i0:i1],
                        t3[:, j0:j1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("dmli,mjkabc->ijklabcd", mycc.W_vooo_tc[:, :, l0:l1, i0:i1],
                        t3[:, j0:j1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("bmjk,milcad->ijklabcd", mycc.W_vooo_tc[:, :, j0:j1, k0:k1],
                        t3[:, i0:i1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("bmjl,mikdac->ijklabcd", mycc.W_vooo_tc[:, :, j0:j1, l0:l1],
                        t3[:, i0:i1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("cmkj,milbad->ijklabcd", mycc.W_vooo_tc[:, :, k0:k1, j0:j1],
                        t3[:, i0:i1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("dmlj,mikbac->ijklabcd", mycc.W_vooo_tc[:, :, l0:l1, j0:j1],
                        t3[:, i0:i1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("cmkl,mijdab->ijklabcd", mycc.W_vooo_tc[:, :, k0:k1, l0:l1],
                        t3[:, i0:i1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("dmlk,mijcab->ijklabcd", mycc.W_vooo_tc[:, :, l0:l1, k0:k1],
                        t3[:, i0:i1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)

                    # R4: P9 (o5v5 * 12)
                    einsum_("mabeij,mklecd->ijklabcd", mycc.W_ovvvoo[..., i0:i1, j0:j1],
                        c_t3[:, k0:k1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum_("maceik,mjlebd->ijklabcd", mycc.W_ovvvoo[..., i0:i1, k0:k1],
                        c_t3[:, j0:j1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum_("madeil,mjkebc->ijklabcd", mycc.W_ovvvoo[..., i0:i1, l0:l1],
                        c_t3[:, j0:j1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum_("mbaeji,mklecd->ijklabcd", mycc.W_ovvvoo[..., j0:j1, i0:i1],
                        c_t3[:, k0:k1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum_("mcaeki,mjlebd->ijklabcd", mycc.W_ovvvoo[..., k0:k1, i0:i1],
                        c_t3[:, j0:j1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum_("mdaeli,mjkebc->ijklabcd", mycc.W_ovvvoo[..., l0:l1, i0:i1],
                        c_t3[:, j0:j1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum_("mbcejk,milead->ijklabcd", mycc.W_ovvvoo[..., j0:j1, k0:k1],
                        c_t3[:, i0:i1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum_("mbdejl,mikeac->ijklabcd", mycc.W_ovvvoo[..., j0:j1, l0:l1],
                        c_t3[:, i0:i1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum_("mcbekj,milead->ijklabcd", mycc.W_ovvvoo[..., k0:k1, j0:j1],
                        c_t3[:, i0:i1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum_("mdbelj,mikeac->ijklabcd", mycc.W_ovvvoo[..., l0:l1, j0:j1],
                        c_t3[:, i0:i1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum_("mcdekl,mijeab->ijklabcd", mycc.W_ovvvoo[..., k0:k1, l0:l1],
                        c_t3[:, i0:i1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum_("mdcelk,mijeab->ijklabcd", mycc.W_ovvvoo[..., l0:l1, k0:k1],
                        c_t3[:, i0:i1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)

                    # R4: P10 (o5v5 * 24)
                    einsum_("mabiej,mklced->ijklabcd", mycc.W_ovvovo[..., i0:i1, :, j0:j1],
                        t3[:, k0:k1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_("mabiej,mlkdec->ijklabcd", mycc.W_ovvovo[..., i0:i1, :, j0:j1],
                        t3[:, l0:l1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_("maciek,mjlbed->ijklabcd", mycc.W_ovvovo[..., i0:i1, :, k0:k1],
                        t3[:, j0:j1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_("madiel,mjkbec->ijklabcd", mycc.W_ovvovo[..., i0:i1, :, l0:l1],
                        t3[:, j0:j1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_("maciek,mljdeb->ijklabcd", mycc.W_ovvovo[..., i0:i1, :, k0:k1],
                        t3[:, l0:l1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_("madiel,mkjceb->ijklabcd", mycc.W_ovvovo[..., i0:i1, :, l0:l1],
                        t3[:, k0:k1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_("mbajei,mklced->ijklabcd", mycc.W_ovvovo[..., j0:j1, :, i0:i1],
                        t3[:, k0:k1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_("mbajei,mlkdec->ijklabcd", mycc.W_ovvovo[..., j0:j1, :, i0:i1],
                        t3[:, l0:l1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_("mcakei,mjlbed->ijklabcd", mycc.W_ovvovo[..., k0:k1, :, i0:i1],
                        t3[:, j0:j1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_("mdalei,mjkbec->ijklabcd", mycc.W_ovvovo[..., l0:l1, :, i0:i1],
                        t3[:, j0:j1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_("mcakei,mljdeb->ijklabcd", mycc.W_ovvovo[..., k0:k1, :, i0:i1],
                        t3[:, l0:l1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_("mdalei,mkjceb->ijklabcd", mycc.W_ovvovo[..., l0:l1, :, i0:i1],
                        t3[:, k0:k1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_("mbcjek,milaed->ijklabcd", mycc.W_ovvovo[..., j0:j1, :, k0:k1],
                        t3[:, i0:i1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_("mbdjel,mikaec->ijklabcd", mycc.W_ovvovo[..., j0:j1, :, l0:l1],
                        t3[:, i0:i1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_("mcbkej,milaed->ijklabcd", mycc.W_ovvovo[..., k0:k1, :, j0:j1],
                        t3[:, i0:i1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_("mdblej,mikaec->ijklabcd", mycc.W_ovvovo[..., l0:l1, :, j0:j1],
                        t3[:, i0:i1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_("mcdkel,mijaeb->ijklabcd", mycc.W_ovvovo[..., k0:k1, :, l0:l1],
                        t3[:, i0:i1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_("mdclek,mijaeb->ijklabcd", mycc.W_ovvovo[..., l0:l1, :, k0:k1],
                        t3[:, i0:i1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_("mbcjek,mlidea->ijklabcd", mycc.W_ovvovo[..., j0:j1, :, k0:k1],
                        t3[:, l0:l1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_("mbdjel,mkicea->ijklabcd", mycc.W_ovvovo[..., j0:j1, :, l0:l1],
                        t3[:, k0:k1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_("mcbkej,mlidea->ijklabcd", mycc.W_ovvovo[..., k0:k1, :, j0:j1],
                        t3[:, l0:l1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_("mdblej,mkicea->ijklabcd", mycc.W_ovvovo[..., l0:l1, :, j0:j1],
                        t3[:, k0:k1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_("mcdkel,mjibea->ijklabcd", mycc.W_ovvovo[..., k0:k1, :, l0:l1],
                        t3[:, j0:j1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum_("mdclek,mjibea->ijklabcd", mycc.W_ovvovo[..., l0:l1, :, k0:k1],
                        t3[:, j0:j1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)

                    # R4: P11 (o5v5 * 24)
                    einsum_("mcbiej,mklaed->ijklabcd", mycc.W_ovvovo[..., i0:i1, :, j0:j1],
                        t3[:, k0:k1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mdbiej,mlkaec->ijklabcd", mycc.W_ovvovo[..., i0:i1, :, j0:j1],
                        t3[:, l0:l1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mbciek,mjlaed->ijklabcd", mycc.W_ovvovo[..., i0:i1, :, k0:k1],
                        t3[:, j0:j1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mbdiel,mjkaec->ijklabcd", mycc.W_ovvovo[..., i0:i1, :, l0:l1],
                        t3[:, j0:j1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mdciek,mljaeb->ijklabcd", mycc.W_ovvovo[..., i0:i1, :, k0:k1],
                        t3[:, l0:l1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mcdiel,mkjaeb->ijklabcd", mycc.W_ovvovo[..., i0:i1, :, l0:l1],
                        t3[:, k0:k1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mcajei,mklbed->ijklabcd", mycc.W_ovvovo[..., j0:j1, :, i0:i1],
                        t3[:, k0:k1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mdajei,mlkbec->ijklabcd", mycc.W_ovvovo[..., j0:j1, :, i0:i1],
                        t3[:, l0:l1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mbakei,mjlced->ijklabcd", mycc.W_ovvovo[..., k0:k1, :, i0:i1],
                        t3[:, j0:j1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mbalei,mjkdec->ijklabcd", mycc.W_ovvovo[..., l0:l1, :, i0:i1],
                        t3[:, j0:j1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mdakei,mljceb->ijklabcd", mycc.W_ovvovo[..., k0:k1, :, i0:i1],
                        t3[:, l0:l1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mcalei,mkjdeb->ijklabcd", mycc.W_ovvovo[..., l0:l1, :, i0:i1],
                        t3[:, k0:k1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("macjek,milbed->ijklabcd", mycc.W_ovvovo[..., j0:j1, :, k0:k1],
                        t3[:, i0:i1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("madjel,mikbec->ijklabcd", mycc.W_ovvovo[..., j0:j1, :, l0:l1],
                        t3[:, i0:i1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mabkej,milced->ijklabcd", mycc.W_ovvovo[..., k0:k1, :, j0:j1],
                        t3[:, i0:i1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mablej,mikdec->ijklabcd", mycc.W_ovvovo[..., l0:l1, :, j0:j1],
                        t3[:, i0:i1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("madkel,mijceb->ijklabcd", mycc.W_ovvovo[..., k0:k1, :, l0:l1],
                        t3[:, i0:i1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("maclek,mijdeb->ijklabcd", mycc.W_ovvovo[..., l0:l1, :, k0:k1],
                        t3[:, i0:i1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mdcjek,mlibea->ijklabcd", mycc.W_ovvovo[..., j0:j1, :, k0:k1],
                        t3[:, l0:l1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mcdjel,mkibea->ijklabcd", mycc.W_ovvovo[..., j0:j1, :, l0:l1],
                        t3[:, k0:k1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mdbkej,mlicea->ijklabcd", mycc.W_ovvovo[..., k0:k1, :, j0:j1],
                        t3[:, l0:l1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mcblej,mkidea->ijklabcd", mycc.W_ovvovo[..., l0:l1, :, j0:j1],
                        t3[:, k0:k1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mbdkel,mjicea->ijklabcd", mycc.W_ovvovo[..., k0:k1, :, l0:l1],
                        t3[:, j0:j1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mbclek,mjidea->ijklabcd", mycc.W_ovvovo[..., l0:l1, :, k0:k1],
                        t3[:, j0:j1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)

                    # R4: P12 (o6v4 * 12)
                    einsum_("amnijk,mnlbcd->ijklabcd", mycc.W_vooooo[..., i0:i1, j0:j1, k0:k1],
                        t3[:, :, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("amnijl,mnkbdc->ijklabcd", mycc.W_vooooo[..., i0:i1, j0:j1, l0:l1],
                        t3[:, :, k0:k1,], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("amnikl,mnjcdb->ijklabcd", mycc.W_vooooo[..., i0:i1, k0:k1, l0:l1],
                        t3[:, :, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("bmnjik,mnlacd->ijklabcd", mycc.W_vooooo[..., j0:j1, i0:i1, k0:k1],
                        t3[:, :, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("bmnjil,mnkadc->ijklabcd", mycc.W_vooooo[..., j0:j1, i0:i1, l0:l1],
                        t3[:, :, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("bmnjkl,mnicda->ijklabcd", mycc.W_vooooo[..., j0:j1, k0:k1, l0:l1],
                        t3[:, :, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("cmnkij,mnlabd->ijklabcd", mycc.W_vooooo[..., k0:k1, i0:i1, j0:j1],
                        t3[:, :, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("cmnkil,mnjadb->ijklabcd", mycc.W_vooooo[..., k0:k1, i0:i1, l0:l1],
                        t3[:, :, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("cmnkjl,mnibda->ijklabcd", mycc.W_vooooo[..., k0:k1, j0:j1, l0:l1],
                        t3[:, :, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("dmnlij,mnkabc->ijklabcd", mycc.W_vooooo[..., l0:l1, i0:i1, j0:j1],
                        t3[:, :, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("dmnlik,mnjacb->ijklabcd", mycc.W_vooooo[..., l0:l1, i0:i1, k0:k1],
                        t3[:, :, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("dmnljk,mnibca->ijklabcd", mycc.W_vooooo[..., l0:l1, j0:j1, k0:k1],
                        t3[:, :, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)

                    # R4: P13 (o5v4 * 12)
                    einsum_("mlcd,abmijk->ijklabcd", t2[:, l0:l1], mycc.W_vvoooo[..., i0:i1, j0:j1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mkdc,abmijl->ijklabcd", t2[:, k0:k1], mycc.W_vvoooo[..., i0:i1, j0:j1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mlbd,acmikj->ijklabcd", t2[:, l0:l1], mycc.W_vvoooo[..., i0:i1, k0:k1, j0:j1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mkbc,admilj->ijklabcd", t2[:, k0:k1], mycc.W_vvoooo[..., i0:i1, l0:l1, j0:j1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mjdb,acmikl->ijklabcd", t2[:, j0:j1], mycc.W_vvoooo[..., i0:i1, k0:k1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mjcb,admilk->ijklabcd", t2[:, j0:j1], mycc.W_vvoooo[..., i0:i1, l0:l1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mlad,bcmjki->ijklabcd", t2[:, l0:l1], mycc.W_vvoooo[..., j0:j1, k0:k1, i0:i1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mkac,bdmjli->ijklabcd", t2[:, k0:k1], mycc.W_vvoooo[..., j0:j1, l0:l1, i0:i1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mjab,cdmkli->ijklabcd", t2[:, j0:j1], mycc.W_vvoooo[..., k0:k1, l0:l1, i0:i1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mida,bcmjkl->ijklabcd", t2[:, i0:i1], mycc.W_vvoooo[..., j0:j1, k0:k1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("mica,bdmjlk->ijklabcd", t2[:, i0:i1], mycc.W_vvoooo[..., j0:j1, l0:l1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum_("miba,cdmklj->ijklabcd", t2[:, i0:i1], mycc.W_vvoooo[..., k0:k1, l0:l1, j0:j1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)

                    # R4: P14 (o4v5 * 12)
                    einsum_("iled,abcejk->ijklabcd", t2[i0:i1, l0:l1], mycc.W_vvvvoo[..., j0:j1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("ikec,abdejl->ijklabcd", t2[i0:i1, k0:k1], mycc.W_vvvvoo[..., j0:j1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("ijeb,acdekl->ijklabcd", t2[i0:i1, j0:j1], mycc.W_vvvvoo[..., k0:k1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("jled,baceik->ijklabcd", t2[j0:j1, l0:l1], mycc.W_vvvvoo[..., i0:i1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("jkec,badeil->ijklabcd", t2[j0:j1, k0:k1], mycc.W_vvvvoo[..., i0:i1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("kled,cabeij->ijklabcd", t2[k0:k1, l0:l1], mycc.W_vvvvoo[..., i0:i1, j0:j1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("lkec,dabeij->ijklabcd", t2[l0:l1, k0:k1], mycc.W_vvvvoo[..., i0:i1, j0:j1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("kjeb,cadeil->ijklabcd", t2[k0:k1, j0:j1], mycc.W_vvvvoo[..., i0:i1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("ljeb,daceik->ijklabcd", t2[l0:l1, j0:j1], mycc.W_vvvvoo[..., i0:i1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("jiea,bcdekl->ijklabcd", t2[j0:j1, i0:i1], mycc.W_vvvvoo[..., k0:k1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("kiea,cbdejl->ijklabcd", t2[k0:k1, i0:i1], mycc.W_vvvvoo[..., j0:j1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum_("liea,dbcejk->ijklabcd", t2[l0:l1, i0:i1], mycc.W_vvvvoo[..., j0:j1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)

                    # R4: P2 (o4v5 * 4)
                    _unpack_24fold(mycc, t4, t4_tmp, i0, i1, j0, j1, k0, k1, l0, l1)
                    einsum_("ae,ijklebcd->ijklabcd", mycc.tf_vv, t4_tmp[:bi, :bj, :bk, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    _unpack_24fold(mycc, t4, t4_tmp, j0, j1, i0, i1, k0, k1, l0, l1)
                    einsum_("be,jikleacd->ijklabcd", mycc.tf_vv, t4_tmp[:bj, :bi, :bk, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    _unpack_24fold(mycc, t4, t4_tmp, k0, k1, i0, i1, j0, j1, l0, l1)
                    einsum_("ce,kijleabd->ijklabcd", mycc.tf_vv, t4_tmp[:bk, :bi, :bj, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    _unpack_24fold(mycc, t4, t4_tmp, l0, l1, i0, i1, j0, j1, k0, k1)
                    einsum_("de,lijkeabc->ijklabcd", mycc.tf_vv, t4_tmp[:bl, :bi, :bj, :bk],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)

                    # R4: P8 (o4v6 * 6)
                    _unpack_24fold(mycc, t4, t4_tmp, i0, i1, j0, j1, k0, k1, l0, l1)
                    einsum_("abef,ijklefcd->ijklabcd", mycc.W_vvvv_tc, t4_tmp[:bi, :bj, :bk, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    _unpack_24fold(mycc, t4, t4_tmp, i0, i1, k0, k1, j0, j1, l0, l1)
                    einsum_("acef,ikjlefbd->ijklabcd", mycc.W_vvvv_tc, t4_tmp[:bi, :bk, :bj, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    _unpack_24fold(mycc, t4, t4_tmp, i0, i1, l0, l1, j0, j1, k0, k1)
                    einsum_("adef,iljkefbc->ijklabcd", mycc.W_vvvv_tc, t4_tmp[:bi, :bl, :bj, :bk],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    _unpack_24fold(mycc, t4, t4_tmp, j0, j1, k0, k1, i0, i1, l0, l1)
                    einsum_("bcef,jkilefad->ijklabcd", mycc.W_vvvv_tc, t4_tmp[:bj, :bk, :bi, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    _unpack_24fold(mycc, t4, t4_tmp, j0, j1, l0, l1, i0, i1, k0, k1)
                    einsum_("bdef,jlikefac->ijklabcd", mycc.W_vvvv_tc, t4_tmp[:bj, :bl, :bi, :bk],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    _unpack_24fold(mycc, t4, t4_tmp, k0, k1, l0, l1, i0, i1, j0, j1)
                    einsum_("cdef,klijefab->ijklabcd", mycc.W_vvvv_tc, t4_tmp[:bk, :bl, :bi, :bj],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)

                    _update_packed_24fold(mycc, r4, r4_tmp, i0, i1, j0, j1, k0, k1, l0, l1)
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
                        _unpack_24fold(mycc, t4, t4_tmp, m0, m1, j0, j1, k0, k1, l0, l1)
                        einsum_("mi,mjklabcd->ijklabcd", mycc.tf_oo[m0:m1, i0:i1], t4_tmp[:bm, :bj, :bk, :bl],
                            out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        t4_symm_ip(t4_tmp, blksize**4, nvir, "cnnn", 1.0, 0.0)
                        einsum_("maei,mjklebcd->ijklabcd", mycc.W_ovvo_tc[m0:m1, :, :, i0:i1],
                            t4_tmp[:bm, :bj, :bk, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.5, beta=1.0)
                        _unpack_24fold(mycc, t4, t4_tmp, m0, m1, i0, i1, k0, k1, l0, l1)
                        einsum_("mj,miklbacd->ijklabcd", mycc.tf_oo[m0:m1, j0:j1], t4_tmp[:bm, :bi, :bk, :bl],
                            out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        t4_symm_ip(t4_tmp, blksize**4, nvir, "cnnn", 1.0, 0.0)
                        einsum_("mbej,mikleacd->ijklabcd", mycc.W_ovvo_tc[m0:m1, :, :, j0:j1],
                            t4_tmp[:bm, :bi, :bk, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.5, beta=1.0)
                        _unpack_24fold(mycc, t4, t4_tmp, m0, m1, i0, i1, j0, j1, l0, l1)
                        einsum_("mk,mijlcabd->ijklabcd", mycc.tf_oo[m0:m1, k0:k1], t4_tmp[:bm, :bi, :bj, :bl],
                            out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        t4_symm_ip(t4_tmp, blksize**4, nvir, "cnnn", 1.0, 0.0)
                        einsum_("mcek,mijleabd->ijklabcd", mycc.W_ovvo_tc[m0:m1, :, :, k0:k1],
                            t4_tmp[:bm, :bi, :bj, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.5, beta=1.0)
                        _unpack_24fold(mycc, t4, t4_tmp, m0, m1, i0, i1, j0, j1, k0, k1)
                        einsum_("ml,mijkdabc->ijklabcd", mycc.tf_oo[m0:m1, l0:l1], t4_tmp[:bm, :bi, :bj, :bk],
                            out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        t4_symm_ip(t4_tmp, blksize**4, nvir, "cnnn", 1.0, 0.0)
                        einsum_("mdel,mijkeabc->ijklabcd", mycc.W_ovvo_tc[m0:m1, :, :, l0:l1],
                            t4_tmp[:bm, :bi, :bj, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.5, beta=1.0)

                        # R4: P5 (o5v5 * 12) & P6 (o5v5 * 12)
                        _unpack_24fold(mycc, t4, t4_tmp, m0, m1, j0, j1, k0, k1, l0, l1)
                        einsum_("maie,mjklbecd->ijklabcd", mycc.W_ovov_tc[m0:m1, :, i0:i1, :],
                            t4_tmp[:bm, :bj, :bk, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum_("mbie,mjklaecd->ijklabcd", mycc.W_ovov_tc[m0:m1, :, i0:i1, :],
                            t4_tmp[:bm, :bj, :bk, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_24fold(mycc, t4, t4_tmp, m0, m1, k0, k1, j0, j1, l0, l1)
                        einsum_("maie,mkjlcebd->ijklabcd", mycc.W_ovov_tc[m0:m1, :, i0:i1, :],
                            t4_tmp[:bm, :bk, :bj, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum_("mcie,mkjlaebd->ijklabcd", mycc.W_ovov_tc[m0:m1, :, i0:i1, :],
                            t4_tmp[:bm, :bk, :bj, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_24fold(mycc, t4, t4_tmp, m0, m1, l0, l1, j0, j1, k0, k1)
                        einsum_("maie,mljkdebc->ijklabcd", mycc.W_ovov_tc[m0:m1, :, i0:i1, :],
                            t4_tmp[:bm, :bl, :bj, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum_("mdie,mljkaebc->ijklabcd", mycc.W_ovov_tc[m0:m1, :, i0:i1, :],
                            t4_tmp[:bm, :bl, :bj, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_24fold(mycc, t4, t4_tmp, m0, m1, i0, i1, k0, k1, l0, l1)
                        einsum_("mbje,miklaecd->ijklabcd", mycc.W_ovov_tc[m0:m1, :, j0:j1, :],
                            t4_tmp[:bm, :bi, :bk, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum_("maje,miklbecd->ijklabcd", mycc.W_ovov_tc[m0:m1, :, j0:j1, :],
                            t4_tmp[:bm, :bi, :bk, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_24fold(mycc, t4, t4_tmp, m0, m1, i0, i1, j0, j1, l0, l1)
                        einsum_("mcke,mijlaebd->ijklabcd", mycc.W_ovov_tc[m0:m1, :, k0:k1, :],
                            t4_tmp[:bm, :bi, :bj, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum_("make,mijlcebd->ijklabcd", mycc.W_ovov_tc[m0:m1, :, k0:k1, :],
                            t4_tmp[:bm, :bi, :bj, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_24fold(mycc, t4, t4_tmp, m0, m1, i0, i1, j0, j1, k0, k1)
                        einsum_("mdle,mijkaebc->ijklabcd", mycc.W_ovov_tc[m0:m1, :, l0:l1, :],
                            t4_tmp[:bm, :bi, :bj, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum_("male,mijkdebc->ijklabcd", mycc.W_ovov_tc[m0:m1, :, l0:l1, :],
                            t4_tmp[:bm, :bi, :bj, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_24fold(mycc, t4, t4_tmp, m0, m1, k0, k1, i0, i1, l0, l1)
                        einsum_("mbje,mkilcead->ijklabcd", mycc.W_ovov_tc[m0:m1, :, j0:j1, :],
                            t4_tmp[:bm, :bk, :bi, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum_("mcje,mkilbead->ijklabcd", mycc.W_ovov_tc[m0:m1, :, j0:j1, :],
                            t4_tmp[:bm, :bk, :bi, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_24fold(mycc, t4, t4_tmp, m0, m1, l0, l1, i0, i1, k0, k1)
                        einsum_("mbje,mlikdeac->ijklabcd", mycc.W_ovov_tc[m0:m1, :, j0:j1, :],
                            t4_tmp[:bm, :bl, :bi, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum_("mdje,mlikbeac->ijklabcd", mycc.W_ovov_tc[m0:m1, :, j0:j1, :],
                            t4_tmp[:bm, :bl, :bi, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_24fold(mycc, t4, t4_tmp, m0, m1, j0, j1, i0, i1, l0, l1)
                        einsum_("mcke,mjilbead->ijklabcd", mycc.W_ovov_tc[m0:m1, :, k0:k1, :],
                            t4_tmp[:bm, :bj, :bi, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum_("mbke,mjilcead->ijklabcd", mycc.W_ovov_tc[m0:m1, :, k0:k1, :],
                            t4_tmp[:bm, :bj, :bi, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_24fold(mycc, t4, t4_tmp, m0, m1, j0, j1, i0, i1, k0, k1)
                        einsum_("mdle,mjikbeac->ijklabcd", mycc.W_ovov_tc[m0:m1, :, l0:l1, :],
                            t4_tmp[:bm, :bj, :bi, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum_("mble,mjikdeac->ijklabcd", mycc.W_ovov_tc[m0:m1, :, l0:l1, :],
                            t4_tmp[:bm, :bj, :bi, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_24fold(mycc, t4, t4_tmp, m0, m1, l0, l1, i0, i1, j0, j1)
                        einsum_("mcke,mlijdeab->ijklabcd", mycc.W_ovov_tc[m0:m1, :, k0:k1, :],
                            t4_tmp[:bm, :bl, :bi, :bj], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum_("mdke,mlijceab->ijklabcd", mycc.W_ovov_tc[m0:m1, :, k0:k1, :],
                            t4_tmp[:bm, :bl, :bi, :bj], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_24fold(mycc, t4, t4_tmp, m0, m1, k0, k1, i0, i1, j0, j1)
                        einsum_("mdle,mkijceab->ijklabcd", mycc.W_ovov_tc[m0:m1, :, l0:l1, :],
                            t4_tmp[:bm, :bk, :bi, :bj], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum_("mcle,mkijdeab->ijklabcd", mycc.W_ovov_tc[m0:m1, :, l0:l1, :],
                            t4_tmp[:bm, :bk, :bi, :bj], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)

                    _update_packed_24fold(mycc, r4, r4_tmp, i0, i1, j0, j1, k0, k1, l0, l1, beta=1.0)
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
                            _unpack_24fold(mycc, t4, t4_tmp, m0, m1, n0, n1, k0, k1, l0, l1)
                            einsum_("mnij,mnklabcd->ijklabcd", mycc.W_oooo[m0:m1, n0:n1, i0:i1, j0:j1],
                                t4_tmp[:bm, :bn, :bk, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                            _unpack_24fold(mycc, t4, t4_tmp, m0, m1, n0, n1, j0, j1, l0, l1)
                            einsum_("mnik,mnjlacbd->ijklabcd", mycc.W_oooo[m0:m1, n0:n1, i0:i1, k0:k1],
                                t4_tmp[:bm, :bn, :bj, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                            _unpack_24fold(mycc, t4, t4_tmp, m0, m1, n0, n1, j0, j1, k0, k1)
                            einsum_("mnil,mnjkadbc->ijklabcd", mycc.W_oooo[m0:m1, n0:n1, i0:i1, l0:l1],
                                t4_tmp[:bm, :bn, :bj, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                            _unpack_24fold(mycc, t4, t4_tmp, m0, m1, n0, n1, i0, i1, l0, l1)
                            einsum_("mnjk,mnilbcad->ijklabcd", mycc.W_oooo[m0:m1, n0:n1, j0:j1, k0:k1],
                                t4_tmp[:bm, :bn, :bi, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                            _unpack_24fold(mycc, t4, t4_tmp, m0, m1, n0, n1, i0, i1, k0, k1)
                            einsum_("mnjl,mnikbdac->ijklabcd", mycc.W_oooo[m0:m1, n0:n1, j0:j1, l0:l1],
                                t4_tmp[:bm, :bn, :bi, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                            _unpack_24fold(mycc, t4, t4_tmp, m0, m1, n0, n1, i0, i1, j0, j1)
                            einsum_("mnkl,mnijcdab->ijklabcd", mycc.W_oooo[m0:m1, n0:n1, k0:k1, l0:l1],
                                t4_tmp[:bm, :bn, :bi, :bj], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)

                    _update_packed_24fold(mycc, r4, r4_tmp, i0, i1, j0, j1, k0, k1, l0, l1, beta=1.0)
        time2 = log.timer_debug1('t4: iter: P7 [%3d, %3d]:'%(l0, l1), *time2)
    t4_tmp = None
    r4_tmp = None
    time1 = log.timer_debug1('t4: P7', *time1)

    return r4

def r4_tril_divide_e(mycc, r4):
    nocc, nvir = mycc.nocc, mycc.nvir
    blksize = mycc.blksize

    eia = mycc.mo_energy[: nocc, None] - mycc.mo_energy[None, nocc :] - mycc.level_shift
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

                    _unpack_24fold(mycc, r4, r4_tmp, i0, i1, j0, j1, k0, k1, l0, l1)
                    r4_tmp[:bi, :bj, :bk, :bl] /= eijklabcd_blk

                    _update_packed_24fold(mycc, r4, r4_tmp, i0, i1, j0, j1, k0, k1, l0, l1)

    eijklabcd_blk = None
    r4_tmp = None
    return r4

def update_amps_t4_tril(mycc, tamps):
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    nocc4, nvir = mycc.nocc4, mycc.nvir

    _, t2, t3, t4 = tamps

    intermediates_t4_tril(mycc, t2, t3, t4)
    mycc.t1_eris = None
    time1 = log.timer_debug1('t4: update intermediates', *time0)

    r4 = compute_r4_tril(mycc, t2, t3, t4)
    time1 = log.timer_debug1('t4: compute r4', *time0)
    # symmetrize r4
    t4_symm_ip(r4, nocc4, nvir, "11111111", -1.0 / 24.0, 1.0)
    rt_symmetrize_t4_tril(r4)
    rt_purify_t4_tril(r4)
    time1 = log.timer_debug1('t4: symmetrize r4', *time1)
    # divide by eijkabc
    r4 = r4_tril_divide_e(mycc, r4)
    time1 = log.timer_debug1('t4: divide r4 by eijklabcd', *time1)

    mycc.r_norm[3] = np.linalg.norm(r4)

    # t4_tril += r4_tril
    t4_add(t4, r4, nocc4, nvir)
    r4 = None
    time1 = log.timer_debug1('t4: update t4', *time1)
    time0 = log.timer_debug1('t4 total', *time0)
    return t4

def kernel(mycc, eris=None, t1=None, t2=None, t3=None, t4=None, tol=1e-8, tolnormt=1e-6, max_cycle=50,
            verbose=5, callback=None, diis_with_t4=False, num_of_subiters=1):
    log = logger.new_logger(mycc, verbose)

    nocc, nocc4, nvir = mycc.nocc, mycc.nocc4, mycc.nvir
    ccdtype = mycc.mo_coeff.dtype

    if eris is None:
        eris = ao2mo_rccsdt(mycc, mycc.mo_coeff)
    if t3 is None:
        t3 = np.zeros((nocc,) * 3 + (nvir,) * 3, dtype=ccdtype)
    else:
        t3 = np.asarray(t3, dtype=ccdtype)
    if t4 is None:
        shape = (nocc4 if isinstance(nocc4, tuple) else (nocc4,)) + (nvir,) * 4
        t4 = np.zeros(shape, dtype=ccdtype)
    else:
        t4 = np.asarray(t4, dtype=ccdtype)
    if t1 is None and t2 is None:
        t1, t2 = mycc.init_amps()[1:3]
    elif t1 is not None and t2 is not None:
        t1, t2 = np.asarray(t1, dtype=ccdtype), np.asarray(t2, dtype=ccdtype)
    else:
        raise ValueError('Input tamps do not satisfy the expected conditions')

    name = mycc.__class__.__name__
    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    e_corr_old = 0.0
    e_corr = mycc.energy(t1, t2)
    log.info('Init E_corr(%s) = %.15g', name, mycc.e_corr)

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
            t1, t2 = mycc.update_amps_t1t2_with_t3t4((t1, t2, t3, t4))
            t3 = mycc.update_amps_t3_with_t4((t1, t2, t3, t4))
        t4 = mycc.update_amps_t4((t1, t2, t3, t4))

        # NOTE: What does this stand for?
        if callback is not None:
            callback(locals())

        normt = np.linalg.norm(mycc.r_norm)

        if mycc.iterative_damping < 1.0:
            raise NotImplementedError("Damping is not implemented")

        if diis_with_t4:
            (t1, t2, t3, t4) = mycc.run_diis((t1, t2, t3, t4), istep, normt, e_corr - e_corr_old, adiis)
        else:
            (t1, t2, t3) = mycc.run_diis((t1, t2, t3), istep, normt, e_corr - e_corr_old, adiis)

        e_corr_old, e_corr = e_corr, mycc.energy(t1, t2)
        mycc.e_corr_ss = getattr(e_corr, 'e_corr_ss', 0)
        mycc.e_corr_os = getattr(e_corr, 'e_corr_os', 0)

        mycc.cycles = istep + 1
        log.info("cycle = %2d  E_corr(RCCSDTQ) = % .12f  dE = % .12e  norm(t1,t2,t3,t4) = %.8e" % (
            istep + 1, e_corr, e_corr - e_corr_old, normt))
        cput1 = log.timer(f'{name} iter', *cput1)

        if abs(e_corr - e_corr_old) < tol and normt < tolnormt:
            converged = True
            break
    log.timer(name, *cput0)
    return converged, e_corr, t1, t2, t3, t4

def restore_from_diis_(mycc, diis_file, inplace=True, diis_with_t4=True):
    adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
    adiis.restore(diis_file, inplace=inplace)

    ccvec = adiis.extrapolate()
    tamps = mycc.vector_to_amplitudes(ccvec)
    if diis_with_t4:
        mycc.t1, mycc.t2, mycc.t3, mycc.t4 = tamps
    else:
        mycc.t1, mycc.t2, mycc.t3 = tamps
        shape = (mycc.nocc4 if isinstance(mycc.nocc4, tuple) else (mycc.nocc4,)) + (mycc.nvir,) * 4
        mycc.t4 = np.zeros(shape, dtype=ccvec.dtype)
    if inplace:
        mycc.diis = adiis
    return mycc


class RCCSDTQ(ccsd.CCSDBase):

    # conv_tol = getattr(__config__, 'cc_rccsdtq_RCCSDTQ_conv_tol', 1e-7)
    # conv_tol_normt = getattr(__config__, 'cc_rccsdtq_RCCSDTQ_conv_tol_normt', 1e-6)

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        # ccsd.CCSDBase.__init__(self, mf, frozen, mo_coeff, mo_occ)
        super().__init__(mf, frozen, mo_coeff, mo_occ)

        self.cc_order = 4
        self.do_tril = [False, False, False, True]

        self.t3 = None
        self.t4 = None
        self.diis_with_t4 = True
        self.num_of_subiters = 1

        self.blksize = 4

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask
    get_e_hf = get_e_hf
    ao2mo = ao2mo_rccsdt
    init_amps = init_amps
    energy = energy
    restore_from_diis_ = restore_from_diis_
    update_amps_t1t2_with_t3t4 = update_amps_t1t2_with_t3t4_tril
    update_amps_t3_with_t4 = update_amps_t3_with_t4_tril
    update_amps_t4 = update_amps_t4_tril
    amplitudes_to_vector = amplitudes_to_vector_rhf
    vector_to_amplitudes = vector_to_amplitudes_rhf
    run_diis = run_diis
    _finalize = _finalize

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        # log.info('CC2 = %g', self.cc2)
        log.info('CCSDTQ nocc = %s, nmo = %s', self.nocc, self.nmo)
        if self.do_tril[-1]:
            log.info("Allocating only the i <= j <= k <= l part of the T4 amplitudes in memory")
        else:
            log.info("Allocating the entire T4 amplitudes in memory")
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

    def kernel(self, t1=None, t2=None, t3=None, t4=None, eris=None):
        return self.ccsdtq(t1, t2, t3, t4, eris)

    def ccsdtq(self, t1=None, t2=None, t3=None, t4=None, eris=None):
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
        self.eris = np.asarray(eris.pqrs.transpose(0, 2, 1, 3), dtype=ccdtype)
        self.eris_ovov = np.asarray(eris.ovov, dtype=ccdtype)
        if not self.eris.flags['C_CONTIGUOUS']:
            self.eris = np.ascontiguousarray(self.eris)
        self.fock = np.asarray(eris.fock, dtype=ccdtype)
        self.mo_energy = np.asarray(eris.mo_energy, dtype=ccdtype)
        self.mo_coeff = np.asarray(eris.mo_coeff, dtype=ccdtype)

        nocc = self.nocc = eris.nocc
        nmo = self.nmo = eris.fock.shape[0]
        nvir = self.nvir = nmo - nocc
        log.info('nocc %5d    nvir %5d    nmo %5d' % (nocc, nvir, nmo))

        if self.do_tril[-1]:
            nocc4 = self.nocc4 = nocc * (nocc + 1) * (nocc + 2) * (nocc + 3) // 24
        else:
            nocc4 = self.nocc4 = (nocc,) * 4

        # estimate the memory cost
        if self.do_tril[-1]:
            t4_memory = nocc4 * nvir**4 * 8 / 1024**2
        else:
            t4_memory = nocc**4 * nvir**4 * 8 / 1024**2
        log.info('T4 memory             %8.5e MB' % (t4_memory))
        eris_memory = nmo**4 * 8 / 1024**2
        log.info('eris memory           %8.5e MB' % (eris_memory))
        if self.diis_with_t4:
            diis_memory = (nocc * (nocc + 1) * (nocc + 2) * (nocc + 3) // 6 * nvir**4 * 8 / 1024**2
                            * self.diis_space * 2)
        else:
            diis_memory = nocc * (nocc + 1) * (nocc + 2) // 6 * nvir**3 * 8 / 1024**2 * self.diis_space * 2
        log.info('diis memory           %8.5e MB' % (diis_memory))
        if self.do_tril[-1]:
            total_memory = 2 * t4_memory + 3 * eris_memory + diis_memory
        else:
            total_memory = 3 * t4_memory + 3 * eris_memory + diis_memory
        log.info('total estimate memory %8.5e MB' % (total_memory))
        max_memory = self.max_memory - lib.current_memory()[0]
        if total_memory > max_memory:
            logger.warn(self, 'There may not be enough memory for the %s calculation' % self.__class__.__name__)

        # norm of residual vectors
        self.r_norm = np.zeros(self.cc_order, dtype=ccdtype)

        # map of unique part of t-amplitudes
        self.unique_tamps_map = []
        # t1
        self.unique_tamps_map.append([[slice(None)]])
        # t2
        self.unique_tamps_map.append([np.tril_indices(nocc), np.diag_indices(nocc)])
        # t3
        i, j, k = np.meshgrid(np.arange(nocc), np.arange(nocc), np.arange(nocc), indexing='ij')
        mask_all = (i <= j) & (j <= k)
        mask_three = (i == j) & (j == k)
        mask_two = ((i == j) | (j == k) | (i == k)) & (~mask_three) & mask_all
        self.unique_tamps_map.append([np.where(mask_all), np.where(mask_two), np.where(mask_three)])
        # t4
        if self.diis_with_t4:
            if self.do_tril[-1]:
                self.unique_tamps_map.append([[slice(None)]])
            else:
                i, j, k, l = np.meshgrid(np.arange(nocc), np.arange(nocc), np.arange(nocc),
                                        np.arange(nocc), indexing='ij')
                mask_all = (i <= j) & (j <= k) & (k <= l)
                mask_four = (i == j) & (j == k) & (k == l)
                mask_three = (((i == j) & (j == k) & (k < l)) | ((i < j) & (j == k) & (k == l))) & mask_all
                mask_three_2 = ((i == j) & (j < k) & (k == l)) & mask_all
                mask_two = (((i == j) & (j < k) & (k < l)) | ((i < j) & (j == k) & (k < l))
                            | ((i < j) & (j < k) & (k == l))) & mask_all
                self.unique_tamps_map.append([np.where(mask_all), np.where(mask_two), np.where(mask_three),
                                        np.where(mask_three_2), np.where(mask_four)])

        if self.do_tril[-1]:
            # setup the map for (un)packing
            setup_tril2cube_t4(self)

            # setup the blksize for (un)packing and contraction
            self.blksize = min(self.blksize, (nocc + 1) // 2)
            log.info('blksize %2d' % (self.blksize))

        self.converged, self.e_corr, self.t1, self.t2, self.t3, self.t4 = \
                kernel(self, eris, t1, t2, t3, t4, max_cycle=self.max_cycle,
                       tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                       verbose=self.verbose, callback=self.callback,
                       diis_with_t4=self.diis_with_t4, num_of_subiters=self.num_of_subiters)
        self._finalize()
        return self.e_corr, self.t1, self.t2, self.t3, self.t4

    def ccsdtq_5(self, t1=None, t2=None, t3=None, t4=None, eris=None):
        raise NotImplementedError

    def ipccsdtq(self, nroots=1, left=False, koopmans=False, guess=None, partition=None, eris=None):
        raise NotImplementedError

    def eaccsdtq(self, nroots=1, left=False, koopmans=False, guess=None, partition=None, eris=None):
        raise NotImplementedError

    def eeccsdtq(self, nroots=1, koopmans=False, guess=None, eris=None):
        raise NotImplementedError

    def eomee_ccsdtq_singlet(self, nroots=1, koopmans=False, guess=None, eris=None):
        raise NotImplementedError

    def eomee_ccsdtq_triplet(self, nroots=1, koopmans=False, guess=None, eris=None):
        raise NotImplementedError

    def eomsf_ccsdtq(self, nroots=1, koopmans=False, guess=None, eris=None):
        raise NotImplementedError


if __name__ == "__main__":

    from pyscf import gto, scf, df

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
    mycc.conv_tol = 1e-12
    mycc.conv_tol_normt = 1e-10
    mycc.max_cycle = 100
    mycc.verbose = 5
    mycc.diis_with_t4 = True
    mycc.num_of_subiters = 2
    ecorr, t1, t2, t3, t4 = mycc.kernel()
    print("My E_corr: % .10f    Ref E_corr: % .10f    Diff: % .10e"%(ecorr, ref_ecorr, ecorr - ref_ecorr))
    print()

    # mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="631g", verbose=3)
    # mf = scf.RHF(mol)
    # mf.level_shift = 0.0
    # mf.conv_tol = 1e-14
    # mf.max_cycle = 1000
    # mf.kernel()
    # print()
    # ref_ecorr = -0.2375471406644931
    # frozen = 0
    # mycc = RCCSDTQ(mf, frozen=frozen)
    # mycc.conv_tol = 1e-12
    # mycc.conv_tol_normt = 1e-10
    # mycc.max_cycle = 100
    # mycc.verbose = 5
    # mycc.diis_with_t4 = True
    # mycc.num_of_subiters = 2
    # ecorr, t1, t2, t3, t4 = mycc.kernel()
    # print("My E_corr: % .10f    Ref E_corr: % .10f    Diff: % .10e"%(ecorr, ref_ecorr, ecorr - ref_ecorr))
    # print()

    # mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="ccpvdz", verbose=3)
    # mf = scf.RHF(mol)
    # mf.level_shift = 0.0
    # mf.conv_tol = 1e-14
    # mf.max_cycle = 1000
    # mf.kernel()
    # print()
    # ref_ecorr = -0.3271440696774139
    # frozen = 0
    # mycc = RCCSDTQ(mf, frozen=frozen)
    # mycc.conv_tol = 1e-12
    # mycc.conv_tol_normt = 1e-10
    # mycc.max_cycle = 100
    # mycc.verbose = 5
    # mycc.diis_with_t4 = True
    # mycc.num_of_subiters = 2
    # ecorr, t1, t2, t3 = mycc.kernel()
    # print("My E_corr: % .10f    Ref E_corr: % .10f    Diff: % .10e"%(ecorr, ref_ecorr, ecorr - ref_ecorr))
    # print()
