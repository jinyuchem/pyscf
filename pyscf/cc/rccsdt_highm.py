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
RHF-CCSDT with full T3 amplitudes stored.
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
from pyscf.cc import _ccsd, rccsdt
from pyscf.cc.rccsdt import (einsum_, t3_spin_summation_inplace_, update_t1_fock_eris_, intermediates_t1t2_,
                            compute_r1r2, r1r2_divide_e_, intermediates_t3_, _PhysicistsERIs)
from pyscf import __config__


def t3_spin_summation(A, B, nocc3, nvir, pattern, alpha=1.0, beta=0.0):
    assert A.dtype == np.float64 and A.flags['C_CONTIGUOUS'], "A must be a contiguous float64 array"
    assert B.dtype == np.float64 and B.flags['C_CONTIGUOUS'], "B must be a contiguous float64 array"
    pattern_c = pattern.encode('utf-8')
    _ccsd.libccsdt.t3_spin_summation_c(
        A.ctypes.data_as(ctypes.c_void_p),
        B.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int64(nocc3),
        ctypes.c_int64(nvir),
        ctypes.c_char_p(pattern_c),
        ctypes.c_double(alpha),
        ctypes.c_double(beta)
    )
    return B

def t3_perm_symmetrize_inplace_(A, nocc, nvir, alpha=1.0, beta=0.0):
    assert A.dtype == np.float64 and A.flags['C_CONTIGUOUS'], "A must be a contiguous float64 array"
    _ccsd.libccsdt.t3_perm_symmetrize_inplace_c(
        A.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int64(nocc),
        ctypes.c_int64(nvir),
        ctypes.c_double(alpha),
        ctypes.c_double(beta)
    )
    return A

def purify_tamps_(r):
    '''Zero out unphysical diagonal elements in CC amplitudes, i.e.,
    enforces T = 0 if three or more occupied/virtual indices are equal
    '''
    import itertools, numpy as np
    order = r.ndim // 2
    for perm in itertools.combinations(range(order), 3):
        idxl, idxr = [slice(None)] * order, [slice(None)] * order
        for p in perm:
            idxl[p] = np.mgrid[ : r.shape[p]]
        for p in perm:
            idxr[p] = np.mgrid[ : r.shape[p + order]]
        r[tuple(idxl) + (slice(None), ) * order] = 0.0
        r[(slice(None), ) * order + tuple(idxr)] = 0.0
    return r

def r1r2_add_t3_(mycc, t3, r1, r2):
    '''Add the T3 contributions to r1 and r2. T3 amplitudes are stored in full form.'''
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    t1_fock, t1_eris = mycc.t1_fock, mycc.t1_eris

    cc_t3 = np.empty_like(t3)
    t3_spin_summation(t3, cc_t3, nocc**3, nvir, "P3_422", 1.0, 0.0)
    einsum_(mycc, 'jkbc,kijcab->ia', t1_eris[:nocc, :nocc, nocc:, nocc:], cc_t3, out=r1, alpha=0.5, beta=1.0)
    cc_t3 = None

    # TODO: Avoid recomputing c_t3; cache or reuse the result
    c_t3 = np.empty_like(t3)
    t3_spin_summation(t3, c_t3, nocc**3, nvir, "P3_201", 1.0, 0.0)
    einsum_(mycc, "kc,kijcab->ijab", t1_fock[:nocc, nocc:], c_t3, out=r2, alpha=0.5, beta=1.0)
    einsum_(mycc, "bkcd,kijdac->ijab", t1_eris[nocc:, :nocc, nocc:, nocc:], c_t3, out=r2, alpha=1.0, beta=1.0)
    einsum_(mycc, "kljc,likcab->ijab", t1_eris[:nocc, :nocc, :nocc, nocc:], c_t3, out=r2, alpha=-1.0, beta=1.0)

def intermediates_t3_add_t3_(mycc, t3):
    '''Add the T3-dependent contributions to the T3 intermediates, with T3 stored in full form.'''
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    # TODO: Avoid recomputing c_t3; cache or reuse the result
    c_t3 = np.empty_like(t3)
    t3_spin_summation(t3, c_t3, nocc**3, nvir, "P3_201", 1.0, 0.0)

    einsum_(mycc, 'lmde,mijead->alij', mycc.t1_eris[:nocc, :nocc, nocc:, nocc:], c_t3,
            out=mycc.W_vooo_tc, alpha=1.0, beta=1.0)
    einsum_(mycc, 'lmde,mjleba->abdj', mycc.t1_eris[:nocc, :nocc, nocc:, nocc:], c_t3,
            out=mycc.W_vvvo_tc, alpha=-1.0, beta=1.0)
    return mycc

def compute_r3(mycc, t2, t3):
    '''Compute r3 with full T3 amplitudes; r3 is returned in full form as well.
    r3 will require a symmetry restoration step afterward.
    '''
    time1 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    # TODO: Avoid recomputing c_t3; cache or reuse the result
    c_t3 = np.empty_like(t3)
    t3_spin_summation(t3, c_t3, nocc**3, nvir, "P3_201", 1.0, 0.0)

    r3 = np.empty_like(t3)
    # R3: P0 & P1 & P2
    einsum_(mycc, 'abdj,ikdc->ijkabc', mycc.W_vvvo_tc, t2, out=r3, alpha=1.0, beta=0.0)
    einsum_(mycc, 'alij,lkbc->ijkabc', mycc.W_vooo_tc, t2, out=r3, alpha=-1.0, beta=1.0)
    einsum_(mycc, 'ad,ijkdbc->ijkabc', mycc.tf_vv, t3, out=r3, alpha=0.5, beta=1.0)
    time1 = log.timer_debug1('t3: W_vvvo * t2, W_vooo * t2, f_vv * t3', *time1)
    # R3: P3 & P4
    einsum_(mycc, 'li,ljkabc->ijkabc', mycc.tf_oo, t3, out=r3, alpha=-0.5, beta=1.0)
    einsum_(mycc, 'ladi,ljkdbc->ijkabc', mycc.W_ovvo_tc, c_t3, out=r3, alpha=0.25, beta=1.0)
    c_t3 = None
    time1 = log.timer_debug1('t3: f_oo * t3, W_ovvo * t3', *time1)
    # R3: P5 & P6
    einsum_(mycc, 'laid,jlkdbc->ijkabc', mycc.W_ovov_tc, t3, out=r3, alpha=-0.5, beta=1.0)
    einsum_(mycc, 'lbid,jlkdac->ijkabc', mycc.W_ovov_tc, t3, out=r3, alpha=-1.0, beta=1.0)
    time1 = log.timer_debug1('t3: W_ovov * t3', *time1)
    # R3: P7
    einsum_(mycc, 'lmij,lmkabc->ijkabc', mycc.W_oooo, t3, out=r3, alpha=0.5, beta=1.0)
    time1 = log.timer_debug1('t3: W_oooo * t3', *time1)
    # R3: P8
    einsum_(mycc, 'abde,ijkdec->ijkabc', mycc.W_vvvv_tc, t3, out=r3, alpha=0.5, beta=1.0)
    time1 = log.timer_debug1('t3: W_vvvv * t3', *time1)
    return r3

def r3_divide_e_(mycc, r3, mo_energy):
    nocc = mycc.nocc
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:] - mycc.level_shift
    eijkabc = (eia[:, None, None, :, None, None] + eia[None, :, None, None, :, None]
                + eia[None, None, :, None, None, :])
    r3 /= eijkabc
    eijkabc = None

def update_amps_rccsdt_(mycc, tamps, eris):
    '''Update RCCSDT amplitudes in place, with T3 amplitudes stored in full form.'''
    assert (isinstance(eris, _PhysicistsERIs))
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    t1, t2, t3 = tamps
    mo_energy = eris.mo_energy.copy()

    update_t1_fock_eris_(mycc, t1, eris)
    time1 = log.timer_debug1('update fock and eris', *time0)
    intermediates_t1t2_(mycc, t2)
    time1 = log.timer_debug1('t1t2: update intermediates', *time1)
    r1, r2 = compute_r1r2(mycc, t2)
    r1r2_add_t3_(mycc, t3, r1, r2)
    time1 = log.timer_debug1('t1t2: compute r1 & r2', *time1)
    # symmetrize r2
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

    intermediates_t3_(mycc, t2)
    intermediates_t3_add_t3_(mycc, t3)
    mycc.t1_eris = None
    time1 = log.timer_debug1('t3: update intermediates', *time0)
    r3 = compute_r3(mycc, t2, t3)
    time1 = log.timer_debug1('t3: compute r3', *time1)
    # symmetrize r3
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
    return res_norm


class RCCSDT(rccsdt.RCCSDT):

    do_tril_maxT = getattr(__config__, 'cc_rccsdt_highm_RCCSDT_do_tril_maxT', False)

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        rccsdt.RCCSDT.__init__(self, mf, frozen, mo_coeff, mo_occ)

    update_amps_ = update_amps_rccsdt_


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
    mycc = RCCSDT(mf, frozen=[0, 1])
    mycc.set_einsum_backend('numpy')
    mycc.conv_tol = 1e-12
    mycc.conv_tol_normt = 1e-10
    mycc.max_cycle = 100
    mycc.verbose = 5
    mycc.do_diis_maxT = True
    mycc.incore_complete = True
    mycc.kernel()
    print("E_corr: % .10f    Ref: % .10f    Diff: % .10e"%(mycc.e_corr, ref_ecorr, mycc.e_corr - ref_ecorr))
    print()
