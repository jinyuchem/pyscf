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
RHF-CCSDT with T3 amplitudes stored only for the i <= j <= k index combinations

Ref:
JCP 142, 064108 (2015); DOI:10.1063/1.4907278
'''

import numpy as np
import numpy
from functools import reduce
import ctypes
from pyscf import ao2mo, lib
from pyscf.ao2mo import _ao2mo
from pyscf.lib import logger
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask, get_e_hf, _mo_without_core
from pyscf.cc import ccsd, _ccsd
from pyscf import __config__


def einsum_(mycc, script, *tensors, out=None, alpha=1.0, beta=0.0, optimize='optimal', einsum_backend=None):
    """
    Wrapper for einsum supporting pytblis, tblis_einsum, hast-ctr, pyscf.lib, or numpy backends.
    Defaults to pytblis if available, else falls back to numpy.
    """
    if mycc is not None:
        einsum_backend = mycc.einsum_backend
    else:
        if einsum_backend is None:
            einsum_backend = 'pytblis'

    if einsum_backend == 'pytblis':
        try:
            import pytblis
        except ImportError:
            import numpy as np
            einsum_backend = 'numpy'
    elif einsum_backend == 'hast-ctr':
        try:
            import sys
            sys.path.insert(0, '/mnt/home/yjin1/code/pyscf/t1_dressed_ccsdt/hast-ctr/hast-ctr')
            import hastctr
        except ImportError:
            import numpy as np
            einsum_backend = 'numpy'
    elif einsum_backend == 'tblis_einsum':
        try:
            import sys
            sys.path.insert(0,
                '/mnt/home/yjin1/code/pyscf/.pyscf_genoa/lib/python3.11/site-packages/pyscf/tblis_einsum/')
            import tblis_einsum
        except ImportError:
            import numpy as np
            einsum_backend = 'numpy'
    elif einsum_backend == 'numpy':
        import numpy as np
    elif einsum_backend == 'pyscf':
        from pyscf import lib
    else:
        raise ValueError(f"Unknown einsum_backend: {einsum_backend}")

    def _fix_strides_for_pytblis(arr):
        '''Fix strides_for pytblis'''
        import itertools, numpy as np
        if arr.strides == (0,) * arr.ndim:
            strides = tuple(itertools.accumulate([x if x != 0 else 1 for x in arr.shape],
                            lambda x, y: x * y, initial=1))[:-1]
            arr = np.lib.stride_tricks.as_strided(arr, strides=strides)
        return arr

    if einsum_backend == 'pytblis':
        if out is None:
            result = pytblis.contract(script, *tensors)
        else:
            tensors = [_fix_strides_for_pytblis(x) for x in tensors]
            out = _fix_strides_for_pytblis(out)
            pytblis.contract(script, *tensors, out=out, alpha=alpha, beta=beta)
            return
    elif einsum_backend == 'hast-ctr':
        if out is None:
            result = hastctr.hast_einsum(script, *tensors)
        else:
            hastctr.hast_einsum(script, *tensors, out=out, alpha=alpha, beta=beta)
            return
    elif einsum_backend == 'tblis_einsum':
        if out is None:
            result = tblis_einsum.contract(script, *tensors)
        else:
            tblis_einsum.contract(script, *tensors, out=out, alpha=alpha, beta=beta)
            return
    elif einsum_backend == 'pyscf':
        result = lib.einsum(script, *tensors, optimize=optimize)
    else:
        result = np.einsum(script, *tensors, optimize=optimize)

    if out is None:
        if alpha != 1.0:
            result = alpha * result
        return result
    else:
        if beta == 0.0:
            out[:] = alpha * result
        else:
            out[:] = alpha * result + beta * out
        return out

def t3_symm_ip_(A, nocc3, nvir, pattern, alpha=1.0, beta=0.0):
    assert A.dtype == np.float64 and A.flags['C_CONTIGUOUS'], "A must be a contiguous float64 array"

    pattern_c = pattern.encode('utf-8')

    _ccsd.libcc.t3_symm_ip_c(
        A.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int64(nocc3),
        ctypes.c_int64(nvir),
        ctypes.c_char_p(pattern_c),
        ctypes.c_double(alpha),
        ctypes.c_double(beta)
    )

def call_unpack_6fold_c(t3, t3_blk, map_, mask, i0, i1, j0, j1, k0, k1, nocc, nvir, blk_i, blk_j, blk_k):
    assert t3.dtype == np.float64 and t3_blk.dtype == np.float64
    assert map_.dtype == np.int64 and mask.dtype == np.bool_

    # Ensure arrays are contiguous
    t3_c = np.ascontiguousarray(t3)
    t3_blk_c = np.ascontiguousarray(t3_blk)
    map_c = np.ascontiguousarray(map_)
    mask_c = np.ascontiguousarray(mask)

    _ccsd.libcc.unpack_6fold_c(
        t3_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        t3_blk_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        map_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        mask_c.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
        ctypes.c_int64(i0), ctypes.c_int64(i1),
        ctypes.c_int64(j0), ctypes.c_int64(j1),
        ctypes.c_int64(k0), ctypes.c_int64(k1),
        ctypes.c_int64(nocc), ctypes.c_int64(nvir),
        ctypes.c_int64(blk_i), ctypes.c_int64(blk_j), ctypes.c_int64(blk_k)
    )
    return t3_blk

def call_unpack_6fold_pair_c(t3, t3_blk, map_, mask, i0, i1, j0, j1, k0, k1, nocc, nvir, blk_i, blk_j, blk_k):
    assert t3.dtype == np.float64 and t3_blk.dtype == np.float64
    assert map_.dtype == np.int64 and mask.dtype == np.bool_

    # Ensure arrays are contiguous
    t3_c = np.ascontiguousarray(t3)
    t3_blk_c = np.ascontiguousarray(t3_blk)
    map_c = np.ascontiguousarray(map_)
    mask_c = np.ascontiguousarray(mask)

    _ccsd.libcc.unpack_6fold_pair_c(
        t3_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        t3_blk_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        map_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        mask_c.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
        ctypes.c_int64(i0), ctypes.c_int64(i1),
        ctypes.c_int64(j0), ctypes.c_int64(j1),
        ctypes.c_int64(k0), ctypes.c_int64(k1),
        ctypes.c_int64(nocc), ctypes.c_int64(nvir),
        ctypes.c_int64(blk_i), ctypes.c_int64(blk_j), ctypes.c_int64(blk_k)
    )
    return t3_blk

def call_unpack_6fold_pair_s_c(t3, t3_blk, map_, mask, i0, j0, k0, nocc, nvir):
    assert t3.dtype == np.float64 and t3_blk.dtype == np.float64
    assert map_.dtype == np.int64 and mask.dtype == np.bool_

    # Ensure arrays are contiguous
    t3_c = np.ascontiguousarray(t3)
    t3_blk_c = np.ascontiguousarray(t3_blk)
    map_c = np.ascontiguousarray(map_)
    mask_c = np.ascontiguousarray(mask)

    _ccsd.libcc.unpack_6fold_pair_s_c(
        t3_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        t3_blk_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        map_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        mask_c.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
        ctypes.c_int64(i0), ctypes.c_int64(j0), ctypes.c_int64(k0),
        ctypes.c_int64(nocc), ctypes.c_int64(nvir),
    )
    return t3_blk

def call_unpack_6fold_pair_2_c(t3, t3_blk, map_, mask, i0, i1, j0, j1, k0, k1, nocc, nvir, blk_i, blk_j, blk_k):
    assert t3.dtype == np.float64 and t3_blk.dtype == np.float64
    assert map_.dtype == np.int64 and mask.dtype == np.bool_

    # Ensure arrays are contiguous
    t3_c = np.ascontiguousarray(t3)
    t3_blk_c = np.ascontiguousarray(t3_blk)
    map_c = np.ascontiguousarray(map_)
    mask_c = np.ascontiguousarray(mask)

    _ccsd.libcc.unpack_6fold_pair_2_c(
        t3_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        t3_blk_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        map_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        mask_c.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
        ctypes.c_int64(i0), ctypes.c_int64(i1),
        ctypes.c_int64(j0), ctypes.c_int64(j1),
        ctypes.c_int64(k0), ctypes.c_int64(k1),
        ctypes.c_int64(nocc), ctypes.c_int64(nvir),
        ctypes.c_int64(blk_i), ctypes.c_int64(blk_j), ctypes.c_int64(blk_k)
    )
    return t3_blk

def call_update_packed_6fold_c(t3, t3_blk, map_, i0, i1, j0, j1, k0, k1,
                                nocc, nvir, blk_i, blk_j, blk_k, alpha, beta):
    assert t3.dtype == np.float64 and t3_blk.dtype == np.float64
    assert map_.dtype == np.int64

    # Ensure arrays are contiguous
    t3_c = np.ascontiguousarray(t3)
    t3_blk_c = np.ascontiguousarray(t3_blk)
    map_c = np.ascontiguousarray(map_)

    _ccsd.libcc.update_packed_6fold_c(
        t3_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        t3_blk_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        map_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(i0), ctypes.c_int64(i1),
        ctypes.c_int64(j0), ctypes.c_int64(j1),
        ctypes.c_int64(k0), ctypes.c_int64(k1),
        ctypes.c_int64(nocc), ctypes.c_int64(nvir),
        ctypes.c_int64(blk_i), ctypes.c_int64(blk_j), ctypes.c_int64(blk_k),
        ctypes.c_double(alpha), ctypes.c_double(beta)
    )
    return t3

def call_update_packed_6fold_s_c(t3, t3_blk, map_, i0, j0, k0, nocc, nvir, alpha, beta):
    assert t3.dtype == np.float64 and t3_blk.dtype == np.float64
    assert map_.dtype == np.int64

    # Ensure arrays are contiguous
    t3_c = np.ascontiguousarray(t3)
    t3_blk_c = np.ascontiguousarray(t3_blk)
    map_c = np.ascontiguousarray(map_)

    _ccsd.libcc.update_packed_6fold_s_c(
        t3_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        t3_blk_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        map_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(i0), ctypes.c_int64(j0), ctypes.c_int64(k0),
        ctypes.c_int64(nocc), ctypes.c_int64(nvir),
        ctypes.c_double(alpha), ctypes.c_double(beta)
    )
    return t3

def _unpack_6fold_(mycc, t3, t3_blk, i0, i1, j0, j1, k0, k1, blksize0=None, blksize1=None, blksize2=None):
    if blksize0 is None: blksize0 = mycc.blksize
    if blksize1 is None: blksize1 = mycc.blksize
    if blksize2 is None: blksize2 = mycc.blksize
    call_unpack_6fold_c(t3, t3_blk, mycc.tril2cube_map, mycc.tril2cube_mask,
                        i0, i1, j0, j1, k0, k1, mycc.nocc, mycc.nvir, blksize0, blksize1, blksize2)

def _unpack_6fold_pair_s_(mycc, t3, t3_blk, i0, j0, k0):
    '''return paris (ijkabc + jikbac), (ikjacb + kijcab), (jkibca + kjicba)'''
    call_unpack_6fold_pair_s_c(t3, t3_blk, mycc.tril2cube_map, mycc.tril2cube_mask,
                                i0, j0, k0, mycc.nocc, mycc.nvir)

def _unpack_6fold_pair_2_(mycc, t3, t3_blk, i0, i1, j0, j1, k0, k1, blksize0=None, blksize1=None, blksize2=None):
    '''return paris (jlkdbc + kljdcb), (ilkdac + klidca), (iljdab + jlidba)'''
    if blksize0 is None: blksize0 = mycc.blksize_oovv
    if blksize1 is None: blksize1 = mycc.nocc
    if blksize2 is None: blksize2 = mycc.blksize_oovv
    call_unpack_6fold_pair_2_c(t3, t3_blk, mycc.tril2cube_map, mycc.tril2cube_mask,
                                i0, i1, j0, j1, k0, k1, mycc.nocc, mycc.nvir, blksize0, blksize1, blksize2)

def _update_packed_6fold_(mycc, t3, t3_blk, i0, i1, j0, j1, k0, k1,
                        blksize0=None, blksize1=None, blksize2=None, alpha=1.0, beta=0.0):
    if blksize0 is None: blksize0 = mycc.blksize
    if blksize1 is None: blksize1 = mycc.blksize
    if blksize2 is None: blksize2 = mycc.blksize
    call_update_packed_6fold_c(t3, t3_blk, mycc.tril2cube_map, i0, i1, j0, j1, k0, k1,
                        mycc.nocc, mycc.nvir, blksize0, blksize1, blksize2, alpha=alpha, beta=beta)

def _update_packed_6fold_s_(mycc, t3, t3_blk, i0, j0, k0, alpha=1.0, beta=0.0):
    call_update_packed_6fold_s_c(t3, t3_blk, mycc.tril2cube_map, i0, j0, k0,
                                mycc.nocc, mycc.nvir, alpha=alpha, beta=beta)

def setup_tril2cube_t3_(mycc):
    nocc = mycc.nocc
    nocc3 = nocc * (nocc + 1) * (nocc + 2) // 6

    tril2cube_map = np.zeros((6,) + (nocc,) * 3, dtype=np.int64)
    tril2cube_mask = np.zeros((6,) + (nocc,) * 3, dtype=np.bool_)
    tril2cube_tp = []

    i, j, k = np.meshgrid(np.arange(nocc), np.arange(nocc), np.arange(nocc), indexing='ij')
    t3_map = np.where((i <= j) & (j <= k))

    import itertools
    perms = list(itertools.permutations([0, 1, 2]))
    for idx, perm in enumerate(perms):
        tril2cube_map[idx, t3_map[perm[0]], t3_map[perm[1]], t3_map[perm[2]]] = np.arange(nocc3)

    labels = ('i', 'j', 'k')
    collect_relation = {('i', 'j'), ('i', 'k'), ('j', 'k')}
    var_map = {'i': i, 'j': j, 'k': k}
    for idx, perm in enumerate(perms):
        indices = np.argsort(perm)
        vars_sorted = [var_map[labels[indices[i]]] for i in range(3)]
        comparisons = []
        for comparison_idx in range(2):
            left_label = labels[indices[comparison_idx]]
            right_label = labels[indices[comparison_idx + 1]]
            if (right_label, left_label) in collect_relation:
                comparisons.append(vars_sorted[comparison_idx] < vars_sorted[comparison_idx + 1])
            else:
                comparisons.append(vars_sorted[comparison_idx] <= vars_sorted[comparison_idx + 1])
        tril2cube_mask[idx] = comparisons[0] & comparisons[1]

    for idx, perm in enumerate(perms):
        tril2cube_tp.append((0,) + tuple([p + 1 for p in perm]))

    mycc.tril2cube_map = tril2cube_map
    mycc.tril2cube_mask = tril2cube_mask
    mycc.tril2cube_tp = tril2cube_tp
    return mycc

def update_xy(mycc, t1):
    nocc, nmo = mycc.nocc, mycc.nmo
    x = np.eye(nmo, dtype=t1.dtype)
    x[nocc:, :nocc] -= t1.T
    y = np.eye(nmo, dtype=t1.dtype)
    y[:nocc, nocc:] += t1
    return x, y

def update_fock(mycc, x, y, t1, eris):
    nocc = mycc.nocc
    t1_fock = eris.fock + einsum_(mycc, 'risa,ia->rs', eris.pqrs[:, :nocc, :, nocc:], t1) * 2.0
    t1_fock -= einsum_(mycc, 'rias,ia->rs', eris.pqrs[:, :nocc, nocc:, :], t1)
    t1_fock = x @ t1_fock @ y.T
    return t1_fock

def update_eris(mycc, x, y, eris):
    t1_eris = einsum_(mycc, 'tvuw,pt->pvuw', eris.pqrs, x)
    t1_eris = einsum_(mycc, 'pvuw,rv->pruw', t1_eris, x)
    t1_eris = t1_eris.transpose(2, 3, 0, 1)
    if not t1_eris.flags['C_CONTIGUOUS']:
        t1_eris = np.ascontiguousarray(t1_eris)
    t1_eris = einsum_(mycc, 'uwpr,qu->qwpr', t1_eris, y)
    t1_eris = einsum_(mycc, 'qwpr,sw->qspr', t1_eris, y)
    t1_eris = t1_eris.transpose(2, 3, 0, 1)
    return t1_eris

def update_t1_fock_eris_(mycc, t1, eris):
    x, y = update_xy(mycc, t1)
    mycc.t1_fock = update_fock(mycc, x, y, t1, eris)
    mycc.t1_eris = update_eris(mycc, x, y, eris)
    return mycc

def symmetrize_t3_tril_(r):
    '''
    Enforce permutation symmetry of r in the special cases:

    - i = j < k : symmetrize over the last two indices (a,b)
    - i < j = k : symmetrize over the last two indices (b,c)

    The fully diagonal case i = j = k is excluded,
    since those entries are set to zero elsewhere.
    '''
    import itertools, numpy as np
    nocc3 = r.shape[0]
    nocc = int((6 * nocc3) ** (1 / 3))
    assert nocc * (nocc + 1) * (nocc + 2) // 6 == nocc3, "the size of r might be incorrect"

    triples = np.array([(i, j, k) for i in range(nocc) for j in range(i, nocc) for k in range(j, nocc)])
    assert triples.shape[0] == nocc3, "the size of r might be incorrect"

    case_1_idx = np.where((triples[:, 0] == triples[:, 1]) & (triples[:, 1] < triples[:, 2]))[0]
    r[case_1_idx, :, :, :] = (r[case_1_idx, :, :, :] + r[case_1_idx, :, :, :].transpose(0, 2, 1, 3)) / 2

    case_2_idx = np.where((triples[:, 0] < triples[:, 1]) & (triples[:, 1] == triples[:, 2]))[0]
    r[case_2_idx, :, :, :] = (r[case_2_idx, :, :, :] + r[case_2_idx, :, :, :].transpose(0, 1, 3, 2)) / 2

def purify_t3_tril_(r):
    '''Set all entries with i=j=k (or a=b=c) to zero'''
    import itertools, numpy as np
    n = 3
    nocc3 = r.shape[0]
    nocc = int((6 * nocc3) ** (1/3))
    assert nocc * (nocc + 1) * (nocc + 2) // 6 == nocc3, "the size of r might be incorrect"

    # Set all entries with i=j=k (i.e. the full diagonal of the 3-tensor) to zero.
    triples = np.array([(i, j, k) for i in range(nocc) for j in range(i, nocc) for k in range(j, nocc)])
    assert triples.shape[0] == nocc3, "the size of r might be incorrect"

    diag_idx = np.where((triples[:, 0] == triples[:, 1]) & (triples[:, 1] == triples[:, 2]))[0]
    r[diag_idx, :, :, :] = 0.0

    # Set all entries with a=b=c to zero
    for perm in itertools.combinations(range(n), 3):
        idxr = [slice(None)] * n
        for p in perm:
            idxr[p] = np.mgrid[:r.shape[p + 1]]
        r[(slice(None), ) * 1 + tuple(idxr)] = 0.0

def init_amps(mycc, eris=None):
    time0 = logger.process_clock(), logger.perf_counter()
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    e_hf = mycc.e_hf
    if e_hf is None: e_hf = mycc.get_e_hf(mo_coeff=mycc.mo_coeff)

    mo_e = eris.mo_energy
    nocc, nvir = mycc.nocc, mycc.nvir
    eia = mo_e[: nocc, None] - mo_e[None, nocc :]
    eijab = eia[:, None, :, None] + eia[None, :, None, :]

    t1 = eris.fock[:nocc, nocc:] / eia
    t2 = eris.oovv / eijab

    tau = t2 + einsum_(mycc, "ia,jb->ijab", t1, t1)
    e_corr = 2.0 * einsum_(mycc, "ijab,ijab->", eris.oovv, tau)
    e_corr -= einsum_(mycc, "ijba,ijab->", eris.oovv, tau)
    e_corr += 2.0 * einsum_(mycc, "ai,ia->", eris.fock[nocc:, :nocc], t1)
    logger.info(mycc, "Init t2, MP2 energy = % .12f  E_corr(MP2) % .12f" % (e_hf + e_corr, e_corr))

    from math import prod, factorial
    nx = lambda n, order: prod(n + i for i in range(order)) // factorial(order)
    cc_order = mycc.cc_order
    tamps = [t1, t2]
    for order in range(2, cc_order - 1):
        tamp = np.zeros((nocc,) * (order + 1) + (nvir,) * (order + 1), dtype=t1.dtype)
        tamps.append(tamp)
    if mycc.do_tril_maxT:
        tamp = np.zeros((nx(nocc, cc_order),) + (nvir,) * cc_order, dtype=t1.dtype)
    else:
        tamp = np.zeros((nocc,) * cc_order + (nvir,) * cc_order, dtype=t1.dtype)
    tamps.append(tamp)
    logger.timer(mycc, 'init mp2', *time0)
    return e_corr, tamps

def energy(mycc, tamps, eris=None):
    '''CC correlation energy'''
    if tamps is None:
        t1, t2 = mycc.tamps[:2]
    if eris is None: eris = mycc.ao2mo()

    t1, t2 = tamps[:2]
    nocc = t1.shape[0]

    tau = t2 + einsum_(mycc, "ia,jb->ijab", t1, t1)
    ed = einsum_(mycc, 'ijab,ijab->', tau, eris.oovv, optimize='optimal') * 2.0
    ex = - einsum_(mycc, 'ijab,ijba->', tau, eris.oovv, optimize='optimal')

    ess = (ed * 0.5 + ex)
    # NOTE: need double check
    ess += einsum_(mycc, "ai,ia->", eris.fock[nocc:, :nocc], t1) * 2.0
    eos = ed * 0.5

    if abs((ess + eos).imag) > 1e-4:
        logger.warn(mycc, 'Non-zero imaginary part found in %s energy %s', mycc.__class__.__name__, ess + eos)

    mycc.e_corr = lib.tag_array((ess + eos).real, e_corr_ss=ess.real, e_corr_os=eos.real)

    return mycc.e_corr

def intermediates_t1t2_(mycc, t2):
    nocc = mycc.nocc
    t1_fock, t1_eris = mycc.t1_fock, mycc.t1_eris

    tf_vv = t1_fock[nocc:, nocc:].copy()
    einsum_(mycc, 'kldc,kldb->bc', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=tf_vv, alpha=-2.0, beta=1.0)
    einsum_(mycc, 'klcd,kldb->bc', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=tf_vv, alpha=1.0, beta=1.0)

    tf_oo = t1_fock[:nocc, :nocc].copy()
    einsum_(mycc, 'lkcd,ljcd->kj', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=tf_oo, alpha=2.0, beta=1.0)
    einsum_(mycc, 'lkdc,ljcd->kj', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=tf_oo, alpha=-1.0, beta=1.0)

    W_oooo = t1_eris[:nocc, :nocc, :nocc, :nocc].copy()
    einsum_(mycc, 'klcd,ijcd->klij', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=W_oooo, alpha=1.0, beta=1.0)

    W_ovvo = - t1_eris[:nocc, nocc:, nocc:, :nocc]
    einsum_(mycc, 'klcd,ilad->kaci', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=W_ovvo, alpha=-1.0, beta=1.0)
    einsum_(mycc, 'kldc,ilad->kaci', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=W_ovvo, alpha=0.5, beta=1.0)
    einsum_(mycc, 'klcd,ilda->kaci', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=W_ovvo, alpha=0.5, beta=1.0)

    W_ovov = - t1_eris[:nocc, nocc:, :nocc, nocc:]
    einsum_(mycc, 'kldc,liad->kaic', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=W_ovov, alpha=0.5, beta=1.0)

    mycc.tf_vv = tf_vv
    mycc.tf_oo = tf_oo
    mycc.W_oooo = W_oooo
    mycc.W_ovvo = W_ovvo
    mycc.W_ovov = W_ovov
    return mycc

def compute_r1r2(mycc, t2):
    '''Compute r1 and r2, without the contributions from t3. r2 still needs to be symmetrized'''
    nocc = mycc.nocc
    t1_fock, t1_eris = mycc.t1_fock, mycc.t1_eris

    c_t2 = 2.0 * t2 - t2.transpose(0, 1, 3, 2)
    # R1
    r1 = t1_fock[nocc:, :nocc].T
    einsum_(mycc, 'kc,ikac->ia', t1_fock[:nocc, nocc:], c_t2, out=r1, alpha=1.0, beta=1.0)
    einsum_(mycc, 'akcd,ikcd->ia', t1_eris[nocc:, :nocc, nocc:, nocc:], c_t2, out=r1, alpha=1.0, beta=1.0)
    einsum_(mycc, 'klic,klac->ia', t1_eris[:nocc, :nocc, :nocc, nocc:], c_t2, out=r1, alpha=-1.0, beta=1.0)
    # R2
    r2 = 0.5 * t1_eris[nocc:, nocc:, :nocc, :nocc].T
    einsum_(mycc, "bc,ijac->ijab", mycc.tf_vv, t2, out=r2, alpha=1.0, beta=1.0)
    einsum_(mycc, "kj,ikab->ijab", mycc.tf_oo, t2, out=r2, alpha=-1.0, beta=1.0)
    einsum_(mycc, "abcd,ijcd->ijab", t1_eris[nocc:, nocc:, nocc:, nocc:], t2, out=r2, alpha=0.5, beta=1.0)
    einsum_(mycc, "klij,klab->ijab", mycc.W_oooo, t2, out=r2, alpha=0.5, beta=1.0)
    einsum_(mycc, "kajc,ikcb->ijab", mycc.W_ovov, t2, out=r2, alpha=1.0, beta=1.0)
    einsum_(mycc, "kaci,kjcb->ijab", mycc.W_ovvo, t2, out=r2, alpha=-2.0, beta=1.0)
    einsum_(mycc, "kaic,kjcb->ijab", mycc.W_ovov, t2, out=r2, alpha=1.0, beta=1.0)
    einsum_(mycc, "kaci,jkcb->ijab", mycc.W_ovvo, t2, out=r2, alpha=1.0, beta=1.0)
    return r1, r2

def r1r2_add_t3_tril_(mycc, t3, r1, r2):
    '''Add t3 contribution to r1 and r2'''
    nocc, nvir = mycc.nocc, mycc.nvir
    blksize = mycc.blksize
    t1_fock, t1_eris = mycc.t1_fock, mycc.t1_eris

    # r1
    t3_tmp = np.empty((blksize,) * 3 + (nvir,) * 3, dtype=t3.dtype)
    for k0, k1 in lib.prange(0, nocc, blksize):
        bk = k1 - k0
        for j0, j1 in lib.prange(0, nocc, blksize):
            bj = j1 - j0
            for i0, i1 in lib.prange(0, nocc, blksize):
                bi = i1 - i0
                _unpack_6fold_(mycc, t3, t3_tmp, i0, i1, j0, j1, k0, k1)
                t3_symm_ip_(t3_tmp, blksize**3, nvir, "4-2-211-2", 1.0, 0.0)
                einsum_(mycc, 'jkbc,ijkabc->ia', t1_eris[j0:j1, k0:k1, nocc:, nocc:],
                    t3_tmp[:bi, :bj, :bk], out=r1[i0:i1, :], alpha=0.5, beta=1.0)
    t3_tmp = None

    # r2
    t3_tmp = np.empty((blksize,) * 3 + (nvir,) * 3, dtype=t3.dtype)
    for k0, k1 in lib.prange(0, nocc, blksize):
        bk = k1 - k0
        for j0, j1 in lib.prange(0, nocc, blksize):
            bj = j1 - j0
            for i0, i1 in lib.prange(0, nocc, blksize):
                bi = i1 - i0
                _unpack_6fold_(mycc, t3, t3_tmp, k0, k1, i0, i1, j0, j1)
                t3_symm_ip_(t3_tmp, blksize**3, nvir, "20-100-1", 1.0, 0.0)
                einsum_(mycc, "kc,kijcab->ijab", t1_fock[k0:k1, nocc:], t3_tmp[:bk, :bi, :bj],
                    out=r2[i0:i1, j0:j1, :, :], alpha=0.5, beta=1.0)
                einsum_(mycc, "bkcd,kijdac->ijab", t1_eris[nocc:, k0:k1, nocc:, nocc:],
                        t3_tmp[:bk, :bi, :bj], out=r2[i0:i1, j0:j1, :, :], alpha=1.0, beta=1.0)
                einsum_(mycc, "jklc,kijcab->ilab", t1_eris[j0:j1, k0:k1, :nocc, nocc:],
                        t3_tmp[:bk, :bi, :bj], out=r2[i0:i1, :, :, :], alpha=-1.0, beta=1.0)
    t3_tmp = None

def r1r2_divide_e_(mycc, r1, r2, eris):
    nocc = mycc.nocc
    eia = eris.mo_energy[:nocc, None] - eris.mo_energy[None, nocc:] - mycc.level_shift
    r1 /= eia
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    r2 /= eijab

def intermediates_t3_(mycc, t2):
    '''intermediates for t3 residual equation, without contribution from t3'''
    nocc = mycc.nocc
    t1_fock, t1_eris = mycc.t1_fock, mycc.t1_eris
    # FIXME: recompute c_t2 here
    c_t2 = 2.0 * t2 - t2.transpose(0, 1, 3, 2)

    W_vvvv_tc = t1_eris[nocc:, nocc:, nocc:, nocc:].copy()
    einsum_(mycc, 'lmde,lmab->abde', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=W_vvvv_tc, alpha=1.0, beta=1.0)

    W_vooo_tc = t1_eris[nocc:, :nocc, :nocc, :nocc].copy()
    einsum_(mycc, 'ld,ijad->alij', t1_fock[:nocc, nocc:], t2, out=W_vooo_tc, alpha=1.0, beta=1.0)
    einsum_(mycc, 'mldj,mida->alij', t1_eris[:nocc, :nocc, nocc:, :nocc], c_t2, out=W_vooo_tc, alpha=1.0, beta=1.0)
    einsum_(mycc, 'mljd,mida->alij', t1_eris[:nocc, :nocc, :nocc, nocc:], c_t2, out=W_vooo_tc, alpha=-0.5, beta=1.0)
    einsum_(mycc, 'mljd,imda->alij', t1_eris[:nocc, :nocc, :nocc, nocc:], t2, out=W_vooo_tc, alpha=-0.5, beta=1.0)
    einsum_(mycc, 'mlid,jmda->alij', t1_eris[:nocc, :nocc, :nocc, nocc:], t2, out=W_vooo_tc, alpha=-1.0, beta=1.0)
    einsum_(mycc, 'alde,ijde->alij', t1_eris[nocc:, :nocc, nocc:, nocc:], t2, out=W_vooo_tc, alpha=1.0, beta=1.0)

    W_vvvo_tc = t1_eris[nocc:, nocc:, nocc:, :nocc].copy()
    einsum_(mycc, 'laed,ljeb->abdj', t1_eris[:nocc, nocc:, nocc:, nocc:], c_t2, out=W_vvvo_tc, alpha=1.0, beta=1.0)
    einsum_(mycc, 'lade,ljeb->abdj', t1_eris[:nocc, nocc:, nocc:, nocc:], c_t2, out=W_vvvo_tc, alpha=-0.5, beta=1.0)
    einsum_(mycc, 'lade,jleb->abdj', t1_eris[:nocc, nocc:, nocc:, nocc:], t2, out=W_vvvo_tc, alpha=-0.5, beta=1.0)
    einsum_(mycc, 'lbde,jlea->abdj', t1_eris[:nocc, nocc:, nocc:, nocc:], t2, out=W_vvvo_tc, alpha=-1.0, beta=1.0)
    einsum_(mycc, 'lmdj,lmab->abdj', t1_eris[:nocc, :nocc, nocc:, :nocc], t2, out=W_vvvo_tc, alpha=1.0, beta=1.0)

    W_ovvo_tc = (2.0 * t1_eris[:nocc, nocc:, nocc:, :nocc] - t1_eris[:nocc, nocc:, :nocc, nocc:].transpose(0, 1, 3, 2))
    einsum_(mycc, 'mled,miea->ladi', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t2, out=W_ovvo_tc, alpha=2.0, beta=1.0)
    einsum_(mycc, 'mlde,miea->ladi', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t2, out=W_ovvo_tc, alpha=-1.0, beta=1.0)

    W_ovov_tc = t1_eris[:nocc, nocc:, :nocc, nocc:].copy()
    einsum_(mycc, 'mlde,imea->laid', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=W_ovov_tc, alpha=-1.0, beta=1.0)

    mycc.W_vvvv_tc = W_vvvv_tc
    mycc.W_vooo_tc = W_vooo_tc
    mycc.W_vvvo_tc = W_vvvo_tc
    mycc.W_ovvo_tc = W_ovvo_tc
    mycc.W_ovov_tc = W_ovov_tc
    return mycc

def intermediates_t3_add_t3_tril_(mycc, t3):
    '''Add the contribution of t3 to t3 intermediates'''
    nocc, nvir = mycc.nocc, mycc.nvir
    blksize = mycc.blksize
    t1_eris = mycc.t1_eris

    t3_tmp = np.empty((blksize,) * 3 + (nvir,) * 3, dtype=t3.dtype)
    for j0, j1 in lib.prange(0, nocc, blksize):
        bj = j1 - j0
        for l0, l1 in lib.prange(0, nocc, blksize):
            bl = l1 - l0
            for m0, m1 in lib.prange(0, nocc, blksize):
                bm = m1 - m0
                _unpack_6fold_(mycc, t3, t3_tmp, m0, m1, j0, j1, l0, l1)
                t3_symm_ip_(t3_tmp, blksize**3, nvir, "20-100-1", 1.0, 0.0)
                einsum_(mycc, 'imde,mjlead->aijl', t1_eris[:nocc, m0:m1, nocc:, nocc:],
                    t3_tmp[:bm, :bj, :bl], out=mycc.W_vooo_tc[:, :, j0:j1, l0:l1], alpha=1.0, beta=1.0)
                einsum_(mycc, 'lmde,mjleba->abdj', t1_eris[l0:l1, m0:m1, nocc:, nocc:],
                    t3_tmp[:bm, :bj, :bl], out=mycc.W_vvvo_tc[:, :, :, j0:j1], alpha=-1.0, beta=1.0)
    t3_tmp = None
    return mycc

def compute_r3_tril(mycc, t2, t3):
    time1 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    nocc, nvir = mycc.nocc, mycc.nvir
    blksize, blksize_oovv, blksize_oooo = mycc.blksize, mycc.blksize_oovv, mycc.blksize_oooo

    r3 = np.zeros_like(t3)

    t3_tmp = np.empty((blksize,) * 3 + (nvir,) * 3, dtype=t3.dtype)
    r3_tmp = np.empty((blksize,) * 3 + (nvir,) * 3, dtype=t3.dtype)
    time2 = logger.process_clock(), logger.perf_counter()
    for k0, k1 in lib.prange(0, nocc, blksize):
        bk = k1 - k0
        for j0, j1 in lib.prange(0, k1, blksize):
            bj = j1 - j0
            for i0, i1 in lib.prange(0, j1, blksize):
                bi = i1 - i0
                # R3: P0
                einsum_(mycc, 'abdj,ikdc->ijkabc', mycc.W_vvvo_tc[..., j0:j1], t2[i0:i1, k0:k1],
                    out=r3_tmp[:bi, :bj, :bk], alpha=1.0, beta=0.0)
                einsum_(mycc, 'acdk,ijdb->ijkabc', mycc.W_vvvo_tc[..., k0:k1], t2[i0:i1, j0:j1],
                    out=r3_tmp[:bi, :bj, :bk], alpha=1.0, beta=1.0)
                einsum_(mycc, 'badi,jkdc->ijkabc', mycc.W_vvvo_tc[..., i0:i1], t2[j0:j1, k0:k1],
                    out=r3_tmp[:bi, :bj, :bk], alpha=1.0, beta=1.0)
                einsum_(mycc, 'bcdk,jida->ijkabc', mycc.W_vvvo_tc[..., k0:k1], t2[j0:j1, i0:i1],
                    out=r3_tmp[:bi, :bj, :bk], alpha=1.0, beta=1.0)
                einsum_(mycc, 'cadi,kjdb->ijkabc', mycc.W_vvvo_tc[..., i0:i1], t2[k0:k1, j0:j1],
                    out=r3_tmp[:bi, :bj, :bk], alpha=1.0, beta=1.0)
                einsum_(mycc, 'cbdj,kida->ijkabc', mycc.W_vvvo_tc[..., j0:j1], t2[k0:k1, i0:i1],
                    out=r3_tmp[:bi, :bj, :bk], alpha=1.0, beta=1.0)
                # R3: P1
                einsum_(mycc, 'alij,lkbc->ijkabc', mycc.W_vooo_tc[:, :, i0:i1, j0:j1], t2[:, k0:k1, :, :],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=1.0)
                einsum_(mycc, 'alik,ljcb->ijkabc', mycc.W_vooo_tc[:, :, i0:i1, k0:k1], t2[:, j0:j1, :, :],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=1.0)
                einsum_(mycc, 'blji,lkac->ijkabc', mycc.W_vooo_tc[:, :, j0:j1, i0:i1], t2[:, k0:k1, :, :],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=1.0)
                einsum_(mycc, 'bljk,lica->ijkabc', mycc.W_vooo_tc[:, :, j0:j1, k0:k1], t2[:, i0:i1, :, :],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=1.0)
                einsum_(mycc, 'clki,ljab->ijkabc', mycc.W_vooo_tc[:, :, k0:k1, i0:i1], t2[:, j0:j1, :, :],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=1.0)
                einsum_(mycc, 'clkj,liba->ijkabc', mycc.W_vooo_tc[:, :, k0:k1, j0:j1], t2[:, i0:i1, :, :],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=1.0)
                # R3: P2
                _unpack_6fold_(mycc, t3, t3_tmp, i0, i1, j0, j1, k0, k1)
                einsum_(mycc, 'ad,ijkdbc->ijkabc', mycc.tf_vv, t3_tmp[:bi, :bj, :bk],
                    out=r3_tmp[:bi, :bj, :bk], alpha=1.0, beta=1.0)
                _unpack_6fold_(mycc, t3, t3_tmp, j0, j1, i0, i1, k0, k1)
                einsum_(mycc, 'bd,jikdac->ijkabc', mycc.tf_vv, t3_tmp[:bj, :bi, :bk],
                    out=r3_tmp[:bi, :bj, :bk], alpha=1.0, beta=1.0)
                _unpack_6fold_(mycc, t3, t3_tmp, k0, k1, j0, j1, i0, i1)
                einsum_(mycc, 'cd,kjidba->ijkabc', mycc.tf_vv, t3_tmp[:bk, :bj, :bi],
                    out=r3_tmp[:bi, :bj, :bk], alpha=1.0, beta=1.0)

                _update_packed_6fold_(mycc, r3, r3_tmp, i0, i1, j0, j1, k0, k1, alpha=1.0, beta=1.0)
        time2 = log.timer_debug1('t3: iter: W_vvvo, W_vooo, f_vv [%3d, %3d]:'%(k0, k1), *time2)
    t3_tmp = None
    r3_tmp = None
    time1 = log.timer_debug1('t3: W_vvvo * t2, W_vooo * t2, f_vv * t3', *time1)

    # R3: P3 and P4
    t3_tmp = np.empty((nocc,) + (blksize_oovv,) * 2 + (nvir,) * 3, dtype=t3.dtype)
    r3_tmp = np.empty((blksize_oovv,) * 3 + (nvir,) * 3, dtype=t3.dtype)
    time2 = logger.process_clock(), logger.perf_counter()
    for k0, k1 in lib.prange(0, nocc, blksize_oovv):
        bk = k1 - k0
        for j0, j1 in lib.prange(0, k1, blksize_oovv):
            bj = j1 - j0
            for i0, i1 in lib.prange(0, j1, blksize_oovv):
                bi = i1 - i0
                # original
                _unpack_6fold_(mycc, t3, t3_tmp, 0, nocc, j0, j1, k0, k1, nocc, blksize_oovv, blksize_oovv)
                einsum_(mycc, 'li,ljkabc->ijkabc', mycc.tf_oo[:, i0:i1], t3_tmp[:, :bj, :bk],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=0.0)
                t3_symm_ip_(t3_tmp, nocc * blksize_oovv**2, nvir, "20-100-1", 1.0, 0.0)
                einsum_(mycc, 'ladi,ljkdbc->ijkabc', mycc.W_ovvo_tc[..., i0:i1], t3_tmp[:, :bj, :bk],
                    out=r3_tmp[:bi, :bj, :bk], alpha=0.5, beta=1.0)
                # ai <-> bj
                _unpack_6fold_(mycc, t3, t3_tmp, 0, nocc, i0, i1, k0, k1, nocc, blksize_oovv, blksize_oovv)
                einsum_(mycc, 'lj,likbac->ijkabc', mycc.tf_oo[:, j0:j1], t3_tmp[:, :bi, :bk],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=1.0)
                t3_symm_ip_(t3_tmp, nocc * blksize_oovv**2, nvir, "20-100-1", 1.0, 0.0)
                einsum_(mycc, 'lbdj,likdac->ijkabc', mycc.W_ovvo_tc[..., j0:j1], t3_tmp[:, :bi, :bk],
                    out=r3_tmp[:bi, :bj, :bk], alpha=0.5, beta=1.0)
                # ai <-> ck
                _unpack_6fold_(mycc, t3, t3_tmp, 0, nocc, j0, j1, i0, i1, nocc, blksize_oovv, blksize_oovv)
                einsum_(mycc, 'lk,ljicba->ijkabc', mycc.tf_oo[:, k0:k1], t3_tmp[:, :bj, :bi],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=1.0)
                t3_symm_ip_(t3_tmp, nocc * blksize_oovv**2, nvir, "20-100-1", 1.0, 0.0)
                einsum_(mycc, 'lcdk,ljidba->ijkabc', mycc.W_ovvo_tc[..., k0:k1], t3_tmp[:, :bj, :bi],
                    out=r3_tmp[:bi, :bj, :bk], alpha=0.5, beta=1.0)

                _update_packed_6fold_(mycc, r3, r3_tmp, i0, i1, j0, j1, k0, k1,
                                    blksize_oovv, blksize_oovv, blksize_oovv, alpha=1.0, beta=1.0)
        time2 = log.timer_debug1('t3: iter: f_oo, W_ovvo [%3d, %3d]:'%(k0, k1), *time2)
    t3_tmp = None
    r3_tmp = None
    time1 = log.timer_debug1('t3: f_oo * t3, W_ovvo * t3', *time1)

    # R3: P5 & P6
    t3_tmp = np.empty((blksize_oovv, nocc, blksize_oovv,) + (nvir,) * 3, dtype=t3.dtype)
    r3_tmp = np.empty((blksize_oovv,) * 3 + (nvir,) * 3, dtype=t3.dtype)
    time2 = logger.process_clock(), logger.perf_counter()
    for k0, k1 in lib.prange(0, nocc, blksize_oovv):
        bk = k1 - k0
        for j0, j1 in lib.prange(0, k1, blksize_oovv):
            bj = j1 - j0
            for i0, i1 in lib.prange(0, j1, blksize_oovv):
                bi = i1 - i0
                #
                _unpack_6fold_(mycc, t3, t3_tmp, j0, j1, 0, nocc, k0, k1, blksize_oovv, nocc, blksize_oovv)
                einsum_(mycc, 'lbid,jlkdac->ijkabc', mycc.W_ovov_tc[:, :, i0:i1, :], t3_tmp[:bj, :, :bk],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=0.0)
                _unpack_6fold_(mycc, t3, t3_tmp, k0, k1, 0, nocc, j0, j1, blksize_oovv, nocc, blksize_oovv)
                einsum_(mycc, 'lcid,kljdab->ijkabc', mycc.W_ovov_tc[:, :, i0:i1, :], t3_tmp[:bk, :, :bj],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=1.0)
                # laid,jlkdbc + laid,kljdcb
                _unpack_6fold_pair_2_(mycc, t3, t3_tmp, j0, j1, 0, nocc, k0, k1)
                einsum_(mycc, 'laid,jlkdbc->ijkabc', mycc.W_ovov_tc[:, :, i0:i1, :], t3_tmp[:bj, :, :bk],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-0.5, beta=1.0)
                #
                _unpack_6fold_(mycc, t3, t3_tmp, i0, i1, 0, nocc, k0, k1, blksize_oovv, nocc, blksize_oovv)
                einsum_(mycc, 'lajd,ilkdbc->ijkabc', mycc.W_ovov_tc[:, :, j0:j1, :], t3_tmp[:bi, :, :bk],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=1.0)
                _unpack_6fold_(mycc, t3, t3_tmp, k0, k1, 0, nocc, i0, i1, blksize_oovv, nocc, blksize_oovv)
                einsum_(mycc, 'lcjd,klidba->ijkabc', mycc.W_ovov_tc[:, :, j0:j1, :], t3_tmp[:bk, :, :bi],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=1.0)
                # lbjd,ilkdac + lbjd,klidca
                _unpack_6fold_pair_2_(mycc, t3, t3_tmp, i0, i1, 0, nocc, k0, k1)
                einsum_(mycc, 'lbjd,ilkdac->ijkabc', mycc.W_ovov_tc[:, :, j0:j1, :], t3_tmp[:bi, :, :bk],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-0.5, beta=1.0)
                #
                _unpack_6fold_(mycc, t3, t3_tmp, i0, i1, 0, nocc, j0, j1, blksize_oovv, nocc, blksize_oovv)
                einsum_(mycc, 'lakd,iljdcb->ijkabc', mycc.W_ovov_tc[:, :, k0:k1, :], t3_tmp[:bi, :, :bj],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=1.0)
                _unpack_6fold_(mycc, t3, t3_tmp, j0, j1, 0, nocc, i0, i1, blksize_oovv, nocc, blksize_oovv)
                einsum_(mycc, 'lbkd,jlidca->ijkabc', mycc.W_ovov_tc[:, :, k0:k1, :], t3_tmp[:bj, :, :bi],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=1.0)
                # lckd,iljdab + lckd,jlidba
                _unpack_6fold_pair_2_(mycc, t3, t3_tmp, i0, i1, 0, nocc, j0, j1)
                einsum_(mycc, 'lckd,iljdab->ijkabc', mycc.W_ovov_tc[:, :, k0:k1, :], t3_tmp[:bi, :, :bj],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-0.5, beta=1.0)

                _update_packed_6fold_(mycc, r3, r3_tmp, i0, i1, j0, j1, k0, k1,
                                    blksize_oovv, blksize_oovv, blksize_oovv, alpha=1.0, beta=1.0)
        time2 = log.timer_debug1('t3: iter: W_ovov [%3d, %3d]:'%(k0, k1), *time2)
    t3_tmp = None
    r3_tmp = None
    time1 = log.timer_debug1('t3: W_ovov * t3', *time1)

    # R3: P7
    t3_tmp = np.empty((blksize_oooo,) * 2 + (nocc,) + (nvir,) * 3, dtype=t3.dtype)
    r3_tmp = np.empty((blksize_oooo,) * 3 + (nvir,) * 3, dtype=t3.dtype)
    time2 = logger.process_clock(), logger.perf_counter()
    for l0, l1 in lib.prange(0, nocc, blksize_oooo):
        bl = l1 - l0
        for m0, m1 in lib.prange(0, nocc, blksize_oooo):
            bm = m1 - m0
            _unpack_6fold_(mycc, t3, t3_tmp, l0, l1, m0, m1, 0, nocc, blksize_oooo, blksize_oooo, nocc)
            for k0, k1 in lib.prange(0, nocc, blksize_oooo):
                bk = k1 - k0
                for j0, j1 in lib.prange(0, k1, blksize_oooo):
                    bj = j1 - j0
                    for i0, i1 in lib.prange(0, j1, blksize_oooo):
                        bi = i1 - i0
                        einsum_(mycc, 'lmij,lmkabc->ijkabc', mycc.W_oooo[l0:l1, m0:m1, i0:i1, j0:j1],
                                t3_tmp[:bl, :bm, k0:k1], out=r3_tmp[:bi, :bj, :bk], alpha=1.0, beta=0.0)
                        einsum_(mycc, 'lmik,lmjacb->ijkabc', mycc.W_oooo[l0:l1, m0:m1, i0:i1, k0:k1],
                                t3_tmp[:bl, :bm, j0:j1], out=r3_tmp[:bi, :bj, :bk], alpha=1.0, beta=1.0)
                        einsum_(mycc, 'lmjk,lmibca->ijkabc', mycc.W_oooo[l0:l1, m0:m1, j0:j1, k0:k1],
                                t3_tmp[:bl, :bm, i0:i1], out=r3_tmp[:bi, :bj, :bk], alpha=1.0, beta=1.0)
                        _update_packed_6fold_(mycc, r3, r3_tmp, i0, i1, j0, j1, k0, k1,
                                            blksize_oooo, blksize_oooo, blksize_oooo, alpha=1.0, beta=1.0)
        time2 = log.timer_debug1('t3: iter: W_oooo [%3d, %3d]:'%(l0, l1), *time2)
    t3_tmp = None
    r3_tmp = None
    time1 = log.timer_debug1('t3: W_oooo * t3', *time1)

    # R3: P8
    t3_tmp_s = np.empty((nvir, nvir, nvir), dtype=t3.dtype)
    r3_tmp_s = np.empty((nvir, nvir, nvir), dtype=t3.dtype)
    time2 = logger.process_clock(), logger.perf_counter()
    for k0 in range(nocc):
        for j0 in range(k0 + 1):
            for i0 in range(j0 + 1):
                # ijk & jik
                _unpack_6fold_pair_s_(mycc, t3, t3_tmp_s, i0, j0, k0)
                einsum_(mycc, 'abde,dec->abc', mycc.W_vvvv_tc, t3_tmp_s, out=r3_tmp_s, alpha=0.5, beta=0.0)
                # ikj and kij
                _unpack_6fold_pair_s_(mycc, t3, t3_tmp_s, i0, k0, j0)
                einsum_(mycc, 'acde,deb->abc', mycc.W_vvvv_tc, t3_tmp_s, out=r3_tmp_s, alpha=0.5, beta=1.0)
                # jki and kji
                _unpack_6fold_pair_s_(mycc, t3, t3_tmp_s, j0, k0, i0)
                einsum_(mycc, 'bcde,dea->abc', mycc.W_vvvv_tc, t3_tmp_s, out=r3_tmp_s, alpha=0.5, beta=1.0)
                _update_packed_6fold_s_(mycc, r3, r3_tmp_s, i0, j0, k0, alpha=1.0, beta=1.0)
        time2 = log.timer_debug1('t3: iter: W_vvvv %3d:'%k0, *time2)
    t3_tmp_s = None
    r3_tmp_s = None
    time1 = log.timer_debug1('t3: W_vvvv * t3', *time1)
    return r3

def r3_tril_divide_e_(mycc, r3, eris):
    nocc, nvir = mycc.nocc, mycc.nvir
    blksize = mycc.blksize
    eia = eris.mo_energy[:nocc, None] - eris.mo_energy[None, nocc:] - mycc.level_shift
    r3_tmp = np.empty((blksize,) * 3 + (nvir,) * 3, dtype=r3.dtype)
    for k0, k1 in lib.prange(0, nocc, blksize):
        bk = k1 - k0
        for j0, j1 in lib.prange(0, k1, blksize):
            bj = j1 - j0
            for i0, i1 in lib.prange(0, j1, blksize):
                bi = i1 - i0
                eijkabc_blk = (eia[i0:i1, None, None, :, None, None] + eia[None, j0:j1, None, None, :, None]
                            + eia[None, None, k0:k1, None, None, :])
                _unpack_6fold_(mycc, r3, r3_tmp, i0, i1, j0, j1, k0, k1)
                r3_tmp[:bi, :bj, :bk] /= eijkabc_blk
                _update_packed_6fold_(mycc, r3, r3_tmp, i0, i1, j0, j1, k0, k1)
    eijkabc_blk = None
    r3_tmp = None

def update_amps_rccsdt_tril_(mycc, tamps, eris):
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    t1, t2, t3 = tamps
    # t1 t2
    update_t1_fock_eris_(mycc, t1, eris)
    time1 = log.timer_debug1('update fock and eris', *time0)
    intermediates_t1t2_(mycc, t2)
    time1 = log.timer_debug1('t1t2: update intermediates', *time1)
    r1, r2 = compute_r1r2(mycc, t2)
    r1r2_add_t3_tril_(mycc, t3, r1, r2)
    time1 = log.timer_debug1('t1t2: compute r1 & r2', *time1)
    # symmetrize r2
    r2 += r2.transpose(1, 0, 3, 2)
    time1 = log.timer_debug1('t1t2: symmetrize r2', *time1)
    # divide by eijkabc
    r1r2_divide_e_(mycc, r1, r2, eris)
    time1 = log.timer_debug1('t1t2: divide r1 & r2 by eia & eijab', *time1)

    res_norm = [np.linalg.norm(r1), np.linalg.norm(r2)]

    t1 += r1
    t2 += r2
    time1 = log.timer_debug1('t1t2: update t1 & t2', *time1)
    time0 = log.timer_debug1('t1t2 total', *time0)

    # t3
    intermediates_t3_(mycc, t2)
    intermediates_t3_add_t3_tril_(mycc, t3)
    mycc.t1_eris = None
    time1 = log.timer_debug1('t3: update intermediates', *time0)
    r3 = compute_r3_tril(mycc, t2, t3)
    time1 = log.timer_debug1('t3: compute r3', *time1)
    # symmetrize r3
    t3_symm_ip_(r3, r3.shape[0], mycc.nvir, "111111", -1.0 / 6.0, 1.0)
    symmetrize_t3_tril_(r3)
    purify_t3_tril_(r3)
    time1 = log.timer_debug1('t3: symmetrize r3', *time1)
    # divide by eijkabc
    r3_tril_divide_e_(mycc, r3, eris)
    time1 = log.timer_debug1('t3: divide r3 by eijkabc', *time1)

    res_norm.append(np.linalg.norm(r3))

    t3 += r3
    r3 = None
    time1 = log.timer_debug1('t3: update t3', *time1)
    time0 = log.timer_debug1('t3 total', *time0)

    tamps = [t1, t2, t3]
    return res_norm

def amplitudes_to_vector_rhf(mycc, tamps):
    from math import prod, factorial
    nx = lambda n, order: prod(n + i for i in range(order)) // factorial(order)

    nocc, nvir = mycc.nocc, mycc.nvir
    tamps_size = [0]
    for i in range(1, len(tamps) + 1):
        tamps_size.append(nx(nocc, i) * nvir ** i)
    cum_sizes = np.cumsum(tamps_size)
    vector = np.zeros(cum_sizes[-1], dtype=tamps[0].dtype)
    for i, t in enumerate(tamps):
        idx = (*mycc.unique_tamps_map[i][0], *[slice(None)] * (i + 1))
        vector[cum_sizes[i] : cum_sizes[i + 1]] = t[idx].ravel()
    return vector

def vector_to_amplitudes_rhf(mycc, vector):
    from math import prod, factorial
    nx = lambda n, order: prod(n + i for i in range(order)) // factorial(order)

    nocc, nvir = mycc.nocc, mycc.nvir
    tamps_size = [0]
    for i in range(1, mycc.cc_order + 1):
        tamps_size.append(nx(nocc, i) * nvir ** i)
    cum_sizes = np.cumsum(tamps_size)

    try:
        endpoint = cum_sizes.tolist().index(vector.shape[0])
    except ValueError:
        raise ValueError("Mismatch between vector size and tamps size")

    tamps = []
    for i in range(endpoint):
        if (i == mycc.cc_order - 1) and mycc.do_tril_maxT:
            t = np.zeros((nx(nocc, i + 1),) + (nvir,) * (i + 1), dtype=vector.dtype)
        else:
            t = np.zeros((nocc,) * (i + 1) + (nvir,) * (i + 1), dtype=vector.dtype)
        idx = (*mycc.unique_tamps_map[i][0], *[slice(None)] * (i + 1))
        t[idx] = vector[cum_sizes[i] : cum_sizes[i + 1]].reshape((-1,) + (nvir,) * (i + 1))
        do_tril = mycc.do_tril_maxT and (i == mycc.cc_order - 1)
        restore_t_(t, order=i + 1, do_tril=do_tril, unique_tamps_map=mycc.unique_tamps_map[i])
        tamps.append(t)
    return tamps

def restore_t_(t, order=1, do_tril=False, unique_tamps_map=None):
    if order >= 5:
        raise NotImplementedError("restore_t function only works up to T4 amplitudes")
    if order == 2:
        if do_tril:
            raise NotImplementedError
        else:
            idx = (*unique_tamps_map[1], *[slice(None)] * order)
            t[idx] *= (1.0 / 2.0)
            t += t.transpose(1, 0, 3, 2)
    if order == 3:
        if do_tril:
            symmetrize_t3_tril_(t)
            purify_t3_tril_(t)
        else:
            idx = (*unique_tamps_map[1], *[slice(None)] * order)
            t[idx] *= (1.0 / 2.0)
            idx = (*unique_tamps_map[2], *[slice(None)] * order)
            t[idx] *= (1.0 / 6.0)
            nocc, nvir = t.shape[0], t.shape[order]
            from pyscf.cc.rccsdt_highm import t3_p_sum_ip_, purify_tamps_
            t3_p_sum_ip_(t, nocc, nvir, 1.0, 0.0)
            purify_tamps_(t)
    elif order == 4:
        if do_tril:
            from pyscf.cc.rccsdtq import symmetrize_t4_tril_, purify_t4_tril_
            symmetrize_t4_tril_(t)
            purify_t4_tril_(t)
        else:
            idx = (*unique_tamps_map[1], *[slice(None)] * order)
            t[idx] *= (1.0 / 2.0)
            idx = (*unique_tamps_map[2], *[slice(None)] * order)
            t[idx] *= (1.0 / 6.0)
            idx = (*unique_tamps_map[3], *[slice(None)] * order)
            t[idx] *= (1.0 / 4.0)
            idx = (*unique_tamps_map[4], *[slice(None)] * order)
            t[idx] *= (1.0 / 24.0)
            nocc, nvir = t.shape[0], t.shape[order]
            from pyscf.cc.rccsdtq_highm import t4_p_sum_ip_
            t4_p_sum_ip_(t, nocc, nvir, 1.0, 0.0)
            from pyscf.cc.rccsdt_highm import purify_tamps_
            purify_tamps_(t)

def run_diis(mycc, tamps, istep, normt, de, adiis):
    if (adiis and istep >= mycc.diis_start_cycle and abs(de) < mycc.diis_start_energy_diff):
        vector = mycc.amplitudes_to_vector(tamps)
        tamps = mycc.vector_to_amplitudes(adiis.update(vector))
        logger.debug1(mycc, 'DIIS for step %d', istep)
    return tamps

def kernel(mycc, eris=None, tamps=None, tol=1e-8, tolnormt=1e-6, max_cycle=50,
            verbose=5, callback=None):
    log = logger.new_logger(mycc, verbose)

    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)

    if tamps is None:
        tamps = mycc.init_amps(eris)[1]
    else:
        if len(tamps) < mycc.cc_order:
            tamps = [*tamps, *mycc.init_amps(eris)[1][len(tamps) : ]]

    name = mycc.__class__.__name__
    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    e_corr_old = 0.0
    e_corr = mycc.energy(tamps, eris)
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
        res_norm = mycc.update_amps_(tamps, eris)

        if callback is not None:
            callback(locals())

        normt = np.linalg.norm(res_norm)

        if mycc.iterative_damping < 1.0:
            raise NotImplementedError("Damping is not implemented")

        if mycc.do_diis_maxT:
            tamps = mycc.run_diis(tamps, istep, normt, e_corr - e_corr_old, adiis)
        else:
            tamps[:mycc.cc_order - 1] = mycc.run_diis(tamps[:mycc.cc_order - 1], istep, normt,
                                                        e_corr - e_corr_old, adiis)

        e_corr_old, e_corr = e_corr, mycc.energy(tamps, eris)
        mycc.e_corr_ss = getattr(e_corr, 'e_corr_ss', 0)
        mycc.e_corr_os = getattr(e_corr, 'e_corr_os', 0)

        mycc.cycles = istep + 1
        log.info("cycle = %2d  E_corr(%s) = % .12f  dE = % .12e  norm(tamps) = %.8e" % (
            istep + 1, mycc.__class__.__name__, e_corr, e_corr - e_corr_old, normt))
        cput1 = log.timer(f'{name} iter', *cput1)

        if abs(e_corr - e_corr_old) < tol and normt < tolnormt:
            converged = True
            break
    log.timer(name, *cput0)
    return converged, e_corr, tamps

def restore_from_diis_(mycc, diis_file, inplace=True):
    from math import prod, factorial
    nx = lambda n, order: prod(n + i for i in range(order)) // factorial(order)

    nocc, nvir, cc_order = mycc.nocc, mycc.nvir, mycc.cc_order

    adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
    adiis.restore(diis_file, inplace=inplace)

    ccvec = adiis.extrapolate()
    tamps = mycc.vector_to_amplitudes(ccvec)
    if mycc.do_diis_maxT:
        mycc.tamps = tamps
    else:
        mycc.tamps[ : cc_order - 1] = tamps
        if mycc.do_tril_maxT:
            mycc.tamp[-1] = np.zeros((nx(nocc, cc_order),) + (nvir,) * cc_order, dtype=ccvec.dtype)
        else:
            mycc.tamp[-1] = np.zeros((nocc,) * cc_order + (nvir,) * cc_order, dtype=ccvec.dtype)
    if inplace:
        mycc.diis = adiis
    return mycc

def ao2mo_rccsdt(mycc, mo_coeff=None):
    if mycc._scf._eri is not None:
        logger.note(mycc, '_make_eris_incore_' + mycc.__class__.__name__)
        return _make_eris_incore_rccsdt(mycc, mo_coeff)
    elif getattr(mycc._scf, 'with_df', None):
        logger.note(mycc, '_make_df_eris_incore_' + mycc.__class__.__name__)
        return _make_df_eris_incore_rccsdt(mycc, mo_coeff)

def _finalize(mycc):
    name = mycc.__class__.__name__

    if mycc.converged:
        logger.note(mycc, '%s converged', name)
    else:
        logger.note(mycc, '%s not converged', name)
    logger.note(mycc, 'E(%s) = %.16g   E_corr = %.16g', name, mycc.e_tot, mycc.e_corr)
    logger.note(mycc, 'E_corr(same-spin) = %.15g', mycc.e_corr_ss)
    logger.note(mycc, 'E_corr(oppo-spin) = %.15g', mycc.e_corr_os)
    return mycc

def dump_flags(mycc, verbose=None):
    log = logger.new_logger(mycc, verbose)
    log.info('')
    log.info('******** %s ********', mycc.__class__)
    # log.info('CC2 = %g', self.cc2)
    log.info('%s nocc = %s, nmo = %s', mycc.__class__.__name__, mycc.nocc, mycc.nmo)
    if mycc.do_tril_maxT:
        text = '<='.join('ijklml'[:mycc.cc_order])
        log.info("Allocating only the %s part of the T%d amplitude in memory", text, mycc.cc_order)
    else:
        log.info("Allocating the entire T%d amplitude in memory", mycc.cc_order)
    if mycc.frozen is not None:
        log.info('frozen orbitals %s', mycc.frozen)
    log.info('max_cycle = %d', mycc.max_cycle)
    # log.info('direct = %d', mycc.direct)
    log.info('conv_tol = %g', mycc.conv_tol)
    log.info('conv_tol_normt = %s', mycc.conv_tol_normt)
    if mycc.do_diis_maxT:
        log.info('diis with the T%d amplitude', mycc.cc_order)
    else:
        log.info('diis without the T%d amplitude', mycc.cc_order)
    log.info('diis_space = %d', mycc.diis_space)
    #log.info('diis_file = %s', self.diis_file)
    log.info('diis_start_cycle = %d', mycc.diis_start_cycle)
    log.info('diis_start_energy_diff = %g', mycc.diis_start_energy_diff)
    log.info('max_memory %d MB (current use %d MB)', mycc.max_memory, lib.current_memory()[0])
    if mycc.einsum_backend is not None:
        log.info('einsum_backend: %s', mycc.einsum_backend)
    return mycc


class RCCSDT(ccsd.CCSDBase):

    conv_tol = getattr(__config__, 'cc_rccsdt_RCCSDT_conv_tol', 1e-7)
    conv_tol_normt = getattr(__config__, 'cc_rccsdt_RCCSDT_conv_tol_normt', 1e-5)
    cc_order = getattr(__config__, 'cc_rccsdt_RCCSDT_cc_order', 3)
    tamps = getattr(__config__, 'cc_rccsdt_RCCSDT_tamps', [None, None, None])
    do_tril_maxT = getattr(__config__, 'cc_rccsdt_RCCSDT_do_tril_maxT', True)
    do_diis_maxT = getattr(__config__, 'cc_rccsdt_RCCSDT_do_diis_maxT', True)
    blksize = getattr(__config__, 'cc_rccsdt_RCCSDT_blksize', 8)
    blksize_oovv = getattr(__config__, 'cc_rccsdt_RCCSDT_blksize_oovv', 4)
    blksize_oooo = getattr(__config__, 'cc_rccsdt_RCCSDT_blksize_oooo', 4)
    einsum_backend = getattr(__config__, 'cc_rccsdt_RCCSDT_einsum_backend', 'numpy')

    @property
    def t1(self):
        return self.tamps[0]

    @t1.setter
    def t1(self, val):
        self.tamps[0] = val

    @property
    def t2(self):
        return self.tamps[1]

    @t2.setter
    def t2(self, val):
        self.tamps[1] = val

    @property
    def t3(self):
        return self.tamps[2]

    @t3.setter
    def t3(self, val):
        self.tamps[2] = val

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        ccsd.CCSDBase.__init__(self, mf, frozen, mo_coeff, mo_occ)

    def set_einsum_backend(self, backend):
        self.einsum_backend = backend

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask
    get_e_hf = get_e_hf
    ao2mo = ao2mo_rccsdt
    init_amps = init_amps
    energy = energy
    restore_from_diis_ = restore_from_diis_
    update_amps_ = update_amps_rccsdt_tril_
    amplitudes_to_vector = amplitudes_to_vector_rhf
    vector_to_amplitudes = vector_to_amplitudes_rhf
    run_diis = run_diis
    _finalize = _finalize
    dump_flags = dump_flags

    def kernel(self, tamps=None, eris=None):
        return self.ccsdt(tamps, eris)

    def ccsdt(self, tamps=None, eris=None):
        log = logger.Logger(self.stdout, self.verbose)

        assert (self.mo_coeff is not None)
        assert (self.mo_occ is not None)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        self.e_hf = self.get_e_hf()

        if eris is None:
            eris = self.ao2mo(self.mo_coeff)

        nocc = self.nocc = eris.nocc
        nmo = self.nmo = eris.fock.shape[0]
        nvir = self.nvir = nmo - nocc
        # log.info('nocc %5d    nvir %5d    nmo %5d'%(nocc, nvir, nmo))

        if self.do_tril_maxT:
            # setup the map for (un)packing
            setup_tril2cube_t3_(self)

            # setup the blksize for (un)packing and contraction
            self.blksize = min(self.blksize, (self.nocc + 1) // 2)
            self.blksize_oovv = min(self.blksize_oovv, (self.nocc + 1) // 2)
            self.blksize_oooo = min(self.blksize_oooo, (self.nocc + 1) // 2)
            self.blksize_vvvv = 1
            log.info('blksize %2d    blksize_oovv %2d    blksize_oooo %2d    blksize_vvvv %2d'%(
                        self.blksize, self.blksize_oovv, self.blksize_oooo, self.blksize_vvvv))

        log.info('Approximate memory usage estimate')
        if self.do_tril_maxT:
            nocc3 = nocc * (nocc + 1) * (nocc + 2) // 6
            t3_memory = nocc3 * nvir**3 * 8 / 1024**2
        else:
            t3_memory = nocc**3 * nvir**3 * 8 / 1024**2
        log.info('T3 memory               %9.5e MB' % (t3_memory))
        log.info('R3 memory               %9.5e MB' % (t3_memory))
        if not self.do_tril_maxT:
            log.info('Spin-summed T3 memory   %9.5e MB' % (t3_memory))
        eris_memory = nmo**4 * 8 / 1024**2
        log.info('eris memory             %9.5e MB' % (eris_memory))
        log.info('t1_eris memory          %9.5e MB' % (eris_memory))
        log.info('intermediates memory    %9.5e MB' % (eris_memory))
        if self.do_diis_maxT:
            diis_memory = nocc * (nocc + 1) * (nocc + 2) // 6 * nvir**3 * 8 / 1024**2 * self.diis_space * 2
        else:
            diis_memory = nocc * (nocc + 1) // 2 * nvir**2 * 8 / 1024**2 * self.diis_space * 2
        log.info('diis memory             %9.5e MB' % (diis_memory))
        if self.do_tril_maxT:
            blk_memory = self.blksize_oovv**2 * nocc * nvir**3 * 8 / 1024**2 * 2
            log.info('blk workspace           %9.5e MB' % (blk_memory))
        if self.do_tril_maxT:
            total_memory = 2 * t3_memory + 3 * eris_memory + diis_memory + blk_memory
        else:
            total_memory = 3 * t3_memory + 3 * eris_memory + diis_memory
        log.info('total estimate memory   %9.5e MB' % (total_memory))
        max_memory = self.max_memory - lib.current_memory()[0]
        if total_memory > max_memory:
            logger.warn(self, 'There may not be enough memory for the %s calculation' % self.__class__.__name__)
            # change blksize

        # map of unique part of t-amplitudes
        self.unique_tamps_map = []
        # t1
        self.unique_tamps_map.append([[slice(None)]])
        # t2
        self.unique_tamps_map.append([np.tril_indices(nocc), np.diag_indices(nocc)])
        # t3
        if self.do_diis_maxT:
            if self.do_tril_maxT:
                self.unique_tamps_map.append([[slice(None)]])
            else:
                i, j, k = np.meshgrid(np.arange(nocc), np.arange(nocc), np.arange(nocc), indexing='ij')
                mask_all = (i <= j) & (j <= k)
                mask_three = (i == j) & (j == k)
                mask_two = ((i == j) | (j == k) | (i == k)) & (~mask_three) & mask_all
                self.unique_tamps_map.append([np.where(mask_all), np.where(mask_two), np.where(mask_three)])

        self.converged, self.e_corr, self.tamps = \
                kernel(self, eris, tamps, max_cycle=self.max_cycle,
                       tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                       verbose=self.verbose, callback=self.callback)
        self._finalize()
        return self.e_corr, self.tamps

    def ccsdt_q(self, tamps, eris=None):
        raise NotImplementedError


class _PhysicistsERIs:
    '''<pq|rs> = (pr|qs)'''
    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nocc = None
        self.fock = None

    def _common_init_(self, mycc, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        self.mo_coeff = mo_coeff = _mo_without_core(mycc, mo_coeff)

        dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        vhf = mycc._scf.get_veff(mycc.mol, dm)
        fockao = mycc._scf.get_fock(vhf=vhf, dm=dm)
        self.fock = reduce(numpy.dot, (mo_coeff.conj().T, fockao, mo_coeff))

        nocc = self.nocc = mycc.nocc
        self.mol = mycc.mol

        mo_e = self.mo_energy = self.fock.diagonal().real
        try:
            gap = abs(mo_e[:nocc,None] - mo_e[None,nocc:]).min()
            if gap < 1e-5:
                logger.warn(mycc, 'HOMO-LUMO gap %s too small for CCSD.\n'
                            'CCSD may be difficult to converge. Increasing '
                            'CCSD Attribute level_shift may improve '
                            'convergence.', gap)
        except ValueError:  # gap.size == 0
            pass
        return self

def _make_eris_incore_rccsdt(mycc, mo_coeff=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    eris = _PhysicistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]

    eri1 = ao2mo.incore.full(mycc._scf._eri, eris.mo_coeff)
    eris.pqrs = ao2mo.restore(1, eri1, nmo).transpose(0, 2, 1, 3)
    if not eris.pqrs.flags['C_CONTIGUOUS']:
        eris.pqrs = np.ascontiguousarray(eris.pqrs)
    eris.oovv = eris.pqrs[:nocc, :nocc, nocc:, nocc:].copy()

    logger.timer(mycc, mycc.__class__.__name__ + ' integral transformation', *cput0)
    return eris

def _make_df_eris_incore_rccsdt(mycc, mo_coeff=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    eris = _PhysicistsERIs()
    eris._common_init_(mycc, mo_coeff)

    nocc = eris.nocc
    mo_coeff = numpy.asarray(eris.mo_coeff, order='F')
    nao, nmo = mo_coeff.shape

    naux = mycc._scf.with_df.get_naoaux()
    ijslice = (0, nmo, 0, nmo)
    Lpq = numpy.empty((naux, nmo, nmo))
    p1 = 0
    Lpq_tmp = None
    for eri1 in mycc._scf.with_df.loop():
        Lpq_tmp = _ao2mo.nr_e2(eri1, mo_coeff, ijslice, aosym='s2', out=Lpq_tmp).reshape(-1, nmo, nmo)
        p0, p1 = p1, p1 + Lpq_tmp.shape[0]
        Lpq[p0:p1, :, :] = Lpq_tmp[:, :, :]
        Lpq_tmp = None
    Lpq = Lpq.reshape(naux, nmo * nmo)

    eris.pqrs = lib.ddot(Lpq.T, Lpq).reshape(nmo, nmo, nmo, nmo).transpose(0, 2, 1, 3)
    if not eris.pqrs.flags['C_CONTIGUOUS']:
        eris.pqrs = np.ascontiguousarray(eris.pqrs)
    eris.oovv = eris.pqrs[:nocc, :nocc, nocc:, nocc:].copy()

    logger.timer(mycc, mycc.__class__.__name__ + ' integral transformation', *cput0)
    return eris


if __name__ == "__main__":

    from pyscf import gto, scf, df

    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="ccpvdz", verbose=3)
    mf = scf.RHF(mol)
    mf.level_shift = 0.0
    mf.conv_tol = 1e-14
    mf.max_cycle = 1000
    mf.kernel()
    print()

    ref_ecorr = -0.3217858674891447
    frozen = [0, 1]
    mycc = RCCSDT(mf, frozen=frozen)
    mycc.set_einsum_backend('pytblis')
    mycc.conv_tol = 1e-12
    mycc.conv_tol_normt = 1e-10
    mycc.max_cycle = 100
    mycc.verbose = 5
    mycc.do_diis_maxT = True
    ecorr, tamps = mycc.kernel()
    print("My E_corr: % .10f    Ref E_corr: % .10f    Diff: % .10e"%(ecorr, ref_ecorr, ecorr - ref_ecorr))
    print()

    # exit()

    # mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="ccpvtz", verbose=3)
    # mf = scf.RHF(mol)
    # mf.level_shift = 0.0
    # mf.conv_tol = 1e-14
    # mf.max_cycle = 1000
    # mf.kernel()
    # print()

    # ref_ecorr = -0.3906059659777624
    # frozen = [0, 1]
    # mycc = RCCSDT(mf, frozen=frozen)
    # mycc.conv_tol = 1e-12
    # mycc.conv_tol_normt = 1e-10
    # mycc.max_cycle = 100
    # mycc.verbose = 5
    # mycc.do_diis_maxT = True
    # ecorr, tamps = mycc.kernel()
    # print("My E_corr: % .10f    Ref E_corr: % .10f    Diff: % .10e"%(ecorr, ref_ecorr, ecorr - ref_ecorr))
    # print()
