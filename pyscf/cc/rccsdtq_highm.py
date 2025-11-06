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
RHF-CCSDTQ with full T4 amplitudes stored

Ref:
JCP 142, 064108 (2015); DOI:10.1063/1.4907278
'''

import numpy as np
import numpy
import ctypes
from pyscf import lib
from pyscf.lib import logger
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask, get_e_hf, _mo_without_core
from pyscf.cc import _ccsd, rccsdtq
from pyscf.cc.rccsdt import (einsum_, _make_eris_incore_rccsdt, _make_df_eris_incore_rccsdt, t3_symm_ip_,
                            update_t1_fock_eris_, init_amps, energy, intermediates_t1t2_, compute_r1r2, r1r2_divide_e_,
                            intermediates_t3_, run_diis, amplitudes_to_vector_rhf, vector_to_amplitudes_rhf,
                            ao2mo_rccsdt, _finalize, kernel)
from pyscf.cc.rccsdt_highm import (t3_symm, t3_p_sum_ip_, purify_tamps_, r1r2_add_t3_, intermediates_t3_add_t3_,
                                    compute_r3, r3_divide_e_)
from pyscf.cc.rccsdtq import t4_symm_ip_, t4_add_
from pyscf import __config__


def t4_symm(A, B, nocc4, nvir, pattern, alpha=1.0, beta=0.0):
    assert A.dtype == np.float64 and A.flags['C_CONTIGUOUS'], "A must be a contiguous float64 array"
    assert B.dtype == np.float64 and B.flags['C_CONTIGUOUS'], "B must be a contiguous float64 array"

    pattern_c = pattern.encode('utf-8')

    _ccsd.libcc.t4_symm_c(
        A.ctypes.data_as(ctypes.c_void_p),
        B.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int64(nocc4),
        ctypes.c_int64(nvir),
        ctypes.c_char_p(pattern_c),
        ctypes.c_double(alpha),
        ctypes.c_double(beta)
    )
    return B

def t4_p_sum_ip_(A, nocc, nvir, alpha=1.0, beta=0.0):
    assert A.dtype == np.float64 and A.flags['C_CONTIGUOUS'], "A must be a contiguous float64 array"

    _ccsd.libcc.t4_p_sum_ip_c(
        A.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int64(nocc),
        ctypes.c_int64(nvir),
        ctypes.c_double(alpha),
        ctypes.c_double(beta)
    )
    return A

def eijkl_division_(A, eia, nocc, nvir):
    assert A.dtype == np.float64 and A.flags['C_CONTIGUOUS'], "A must be a contiguous float64 array"
    assert eia.dtype == np.float64 and eia.flags['C_CONTIGUOUS'], "eia must be a contiguous float64 array"

    drv = _ccsd.libcc.eijkl_division_c
    drv(
        A.ctypes.data_as(ctypes.c_void_p),
        eia.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int64(nocc),
        ctypes.c_int64(nvir),
    )

def r2_add_t4_(mycc, t4, r2):
    nocc, nvir = mycc.nocc, mycc.nvir
    cc_t4 = np.empty_like(t4)
    t4_symm(t4, cc_t4, nocc**4, nvir, "ccnn", 1.0, 0.0)
    einsum_(mycc, 'mnef,mnijefab->ijab', mycc.t1_eris[:nocc, :nocc, nocc:, nocc:], cc_t4, out=r2, alpha=0.25, beta=1.0)
    cc_t4 = None

def r3_add_t4_(mycc, t4, r3):
    '''Add t4 contribution to r3'''
    nocc, nvir = mycc.nocc, mycc.nvir
    t1_fock, t1_eris = mycc.t1_fock, mycc.t1_eris

    # FIXME
    c_t4 = np.empty_like(t4)
    t4_symm(t4, c_t4, nocc**4, nvir, "cnnn", 1.0, 0.0)
    # R3: P9
    einsum_(mycc, 'me,mijkeabc->ijkabc', t1_fock[:nocc, nocc:], c_t4, out=r3, alpha=1.0 / 6.0, beta=1.0)
    # R3: P10
    einsum_(mycc, 'amef,mijkfebc->ijkabc', t1_eris[nocc:, :nocc, nocc:, nocc:], c_t4, out=r3, alpha=0.5, beta=1.0)
    # R3: P11
    einsum_(mycc, 'mnej,minkeabc->ijkabc', t1_eris[:nocc, :nocc, nocc:, :nocc], c_t4, out=r3, alpha=-0.5, beta=1.0)
    c_t4 = None

def intermediates_t4_(mycc, t2, t3, t4):
    nocc, nvir = mycc.nocc, mycc.nvir
    t1_fock, t1_eris = mycc.t1_fock, mycc.t1_eris

    # FIXME
    c_t2 = 2.0 * t2 - t2.transpose(0, 1, 3, 2)
    c_t3 = np.empty_like(t3)
    t3_symm(t3, c_t3, nocc**3, nvir, "20-100-1", 1.0, 0.0)
    c_t4 = np.empty_like(t4)
    t4_symm(t4, c_t4, nocc**4, nvir, "cnnn", 1.0, 0.0)

    einsum_(mycc, 'me,mjab->abej', t1_fock[:nocc, nocc:], t2, out=mycc.W_vvvo_tc, alpha=-1.0, beta=1.0)

    W_ovvvoo = np.empty((nocc,) + (nvir,) * 3 + (nocc,) * 2, dtype=t2.dtype)
    einsum_(mycc, 'maef,jibf->mabeij', t1_eris[:nocc, nocc:, nocc:, nocc:], t2, out=W_ovvvoo, alpha=2.0, beta=0.0)
    einsum_(mycc, 'mafe,jibf->mabeij', t1_eris[:nocc, nocc:, nocc:, nocc:], t2, out=W_ovvvoo, alpha=-1.0, beta=1.0)
    einsum_(mycc, 'mnei,njab->mabeij', t1_eris[:nocc, :nocc, nocc:, :nocc], t2, out=W_ovvvoo, alpha=-2.0, beta=1.0)
    einsum_(mycc, 'nmei,njab->mabeij', t1_eris[:nocc, :nocc, nocc:, :nocc], t2, out=W_ovvvoo, alpha=1.0, beta=1.0)
    einsum_(mycc, 'nmfe,nijfab->mabeij', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t3, out=W_ovvvoo, alpha=0.5, beta=1.0)
    einsum_(mycc, 'mnfe,nijfab->mabeij', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t3, out=W_ovvvoo, alpha=-0.25, beta=1.0)
    W_ovvvoo += W_ovvvoo.transpose(0, 2, 1, 3, 5, 4)

    W_ovvovo = np.empty((nocc,) + (nvir,) * 2 + (nocc, nvir, nocc), dtype=t2.dtype)
    einsum_(mycc, 'mafe,jibf->mabiej', t1_eris[:nocc, nocc:, nocc:, nocc:], t2, out=W_ovvovo, alpha=1.0, beta=0.0)
    einsum_(mycc, 'mnie,njab->mabiej', t1_eris[:nocc, :nocc, :nocc, nocc:], t2, out=W_ovvovo, alpha=-1.0, beta=1.0)
    einsum_(mycc, 'nmef,injfab->mabiej', t1_eris[:nocc, :nocc, nocc:, nocc:], t3, out=W_ovvovo, alpha=-0.5, beta=1.0)

    W_vooooo = np.empty((nvir,) + (nocc,) * 5, dtype=t2.dtype)
    einsum_(mycc, 'mnek,ijae->amnijk', t1_eris[:nocc, :nocc, nocc:, :nocc], t2, out=W_vooooo, alpha=1.0, beta=0.0)
    einsum_(mycc, 'mnef,ijkaef->amnijk', t1_eris[:nocc, :nocc, nocc:, nocc:], t3, out=W_vooooo, alpha=0.5, beta=1.0)
    W_vooooo += W_vooooo.transpose(0, 2, 1, 3, 5, 4)

    W_vvoooo = np.empty((nvir,) * 2 + (nocc,) * 4, dtype=t2.dtype)
    einsum_(mycc, 'amef,ijkebf->abmijk', t1_eris[nocc:, :nocc, nocc:, nocc:], t3, out=W_vvoooo, alpha=1.0, beta=0.0)
    # FIXME
    W_ovvo_c = t1_eris[:nocc, nocc:, nocc:, :nocc].copy()
    einsum_(mycc, 'nmfe,nifa->maei', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t2, out=W_ovvo_c, alpha=1.0, beta=1.0)
    einsum_(mycc, 'mnfe,nifa->maei', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t2, out=W_ovvo_c, alpha=-0.5, beta=1.0)
    einsum_(mycc, 'nmef,infa->maei', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=W_ovvo_c, alpha=-0.5, beta=1.0)
    #
    einsum_(mycc, 'maei,jkbe->abmijk', W_ovvo_c, t2, out=W_vvoooo, alpha=1.0, beta=1.0)
    einsum_(mycc, 'make,jibe->abmijk', mycc.W_ovov_tc, t2, out=W_vvoooo, alpha=1.0, beta=1.0)
    einsum_(mycc, 'mnki,njab->abmijk', mycc.W_oooo, t2, out=W_vvoooo, alpha=-0.5, beta=1.0)
    einsum_(mycc, 'mnef,nijkfabe->abmijk', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t4, out=W_vvoooo, alpha=0.5, beta=1.0)
    W_vvoooo += W_vvoooo.transpose(1, 0, 2, 4, 3, 5)

    W_vvvvoo = np.empty((nvir,) * 4 + (nocc,) * 2, dtype=t2.dtype)
    einsum_(mycc, 'abef,jkfc->abcejk', mycc.W_vvvv_tc, t2, out=W_vvvvoo, alpha=0.5, beta=0.0)
    einsum_(mycc, 'mnef,nmjkfabc->abcejk', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t4,
            out=W_vvvvoo, alpha=-0.5, beta=1.0)
    W_vvvvoo += W_vvvvoo.transpose(0, 2, 1, 3, 5, 4)

    mycc.W_ovvvoo = W_ovvvoo
    mycc.W_ovvovo = W_ovvovo
    mycc.W_vooooo = W_vooooo
    mycc.W_vvoooo = W_vvoooo
    mycc.W_vvvvoo = W_vvvvoo

    c_t2, c_t3, c_t4 = None, None, None
    return mycc

def compute_r4(mycc, t2, t3, t4):
    time1 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    nocc, nvir = mycc.nocc, mycc.nvir

    # FIXME
    c_t3 = np.empty_like(t3)
    t3_symm(t3, c_t3, nocc**3, nvir, "20-100-1", 1.0, 0.0)
    c_t4 = np.empty_like(t4)
    t4_symm(t4, c_t4, nocc**4, nvir, "cnnn", 1.0, 0.0)

    r4 = np.empty_like(t4)
    # R4: P0
    einsum_(mycc, 'abej,iklecd->ijklabcd', mycc.W_vvvo_tc, t3, out=r4, alpha=0.5, beta=0.0)
    time1 = log.timer_debug1('t4: W_vvvo * t3', *time1)
    # R4: P1
    einsum_(mycc, 'amij,mklbcd->ijklabcd', mycc.W_vooo_tc, t3, out=r4, alpha=-0.5, beta=1.0)
    time1 = log.timer_debug1('t4: W_vooo * t3', *time1)
    # R4: P2
    einsum_(mycc, 'ae,ijklebcd->ijklabcd', mycc.tf_vv, t4, out=r4, alpha=1.0 / 6.0, beta=1.0)
    time1 = log.timer_debug1('t4: f_vv * t4', *time1)
    # R4: P3
    einsum_(mycc, 'mi,mjklabcd->ijklabcd', mycc.tf_oo, t4, out=r4, alpha=-1.0 / 6.0, beta=1.0)
    time1 = log.timer_debug1('t4: f_oo * t4', *time1)
    # R4: P4
    einsum_(mycc, 'maei,mjklebcd->ijklabcd', mycc.W_ovvo_tc, c_t4, out=r4, alpha=1.0 / 12.0, beta=1.0)
    c_t4 = None
    time1 = log.timer_debug1('t4: W_ovvo * c_t4', *time1)
    # R4: P5
    einsum_(mycc, 'maie,jmklebcd->ijklabcd', mycc.W_ovov_tc, t4, out=r4, alpha=-0.25, beta=1.0)
    time1 = log.timer_debug1('t3: W_ovov * t4', *time1)
    # R4: P6
    einsum_(mycc, 'mbie,jmkleacd->ijklabcd', mycc.W_ovov_tc, t4, out=r4, alpha=-0.5, beta=1.0)
    time1 = log.timer_debug1('t4: W_ovov * t4', *time1)
    # R4: P7
    einsum_(mycc, 'mnij,mnklabcd->ijklabcd', mycc.W_oooo, t4, out=r4, alpha=0.25, beta=1.0)
    time1 = log.timer_debug1('t4: W_oooo * t4', *time1)
    # R4: P8
    einsum_(mycc, 'abef,ijklefcd->ijklabcd', mycc.W_vvvv_tc, t4, out=r4, alpha=0.25, beta=1.0)
    time1 = log.timer_debug1('t4: W_vvvv * t4', *time1)
    # R4: P9
    einsum_(mycc, 'mabeij,mklecd->ijklabcd', mycc.W_ovvvoo, c_t3, out=r4, alpha=0.125, beta=1.0)
    c_t3 = None
    time1 = log.timer_debug1('t4: W_ovvvoo * c_t3', *time1)
    # R4: P10 & P11
    einsum_(mycc, 'mabiej,kmlecd->ijklabcd', mycc.W_ovvovo, t3, out=r4, alpha=-0.5, beta=1.0)
    einsum_(mycc, 'mcbiej,kmlead->ijklabcd', mycc.W_ovvovo, t3, out=r4, alpha=-1.0, beta=1.0)
    time1 = log.timer_debug1('t4: W_ovvovo * t3', *time1)
    # R4: P12
    einsum_(mycc, 'amnijk,mnlbcd->ijklabcd', mycc.W_vooooo, t3, out=r4, alpha=0.5, beta=1.0)
    time1 = log.timer_debug1('t4: W_vooooo * t3', *time1)
    # R4: P13
    einsum_(mycc, 'abmijk,mlcd->ijklabcd', mycc.W_vvoooo, t2, out=r4, alpha=-0.5, beta=1.0)
    time1 = log.timer_debug1('t4: W_vvoooo * t2', *time1)
    # R4: P14
    einsum_(mycc, 'abcejk,iled->ijklabcd', mycc.W_vvvvoo, t2, out=r4, alpha=0.5, beta=1.0)
    time1 = log.timer_debug1('t4: W_vvvvoo * t2', *time1)
    return r4

def r4_divide_e_(mycc, r4, eris):
    nocc, nvir = mycc.nocc, mycc.nvir
    eia = eris.mo_energy[:nocc, None] - eris.mo_energy[None, nocc:] - mycc.level_shift
    eijkl_division_(r4, eia, nocc, nvir)

def update_amps_rccsdtq_(mycc, tamps, eris):
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    nocc, nvir = mycc.nocc, mycc.nvir

    t1, t2, t3, t4 = tamps

    # t1, t2
    update_t1_fock_eris_(mycc, t1, eris)
    time1 = log.timer_debug1('update fock and eris', *time0)
    intermediates_t1t2_(mycc, t2)
    time1 = log.timer_debug1('t1t2: update intermediates', *time1)
    r1, r2 = compute_r1r2(mycc, t2)
    r1r2_add_t3_(mycc, t3, r1, r2)
    r2_add_t4_(mycc, t4, r2)
    time1 = log.timer_debug1('t1t2: compute r1 & r2', *time1)
    # symmetrize R2
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
    intermediates_t3_add_t3_(mycc, t3)
    time1 = log.timer_debug1('t3: update intermediates', *time0)
    r3 = compute_r3(mycc, t2, t3)
    r3_add_t4_(mycc, t4, r3)
    time1 = log.timer_debug1('t3: compute r3', *time1)
    # symmetrization
    t3_p_sum_ip_(r3, nocc, nvir, 1.0, 0.0)
    t3_symm_ip_(r3, nocc**3, nvir, "111111", -1.0 / 6.0, 1.0)
    purify_tamps_(r3)
    time1 = log.timer_debug1('t3: symmetrize r3', *time1)
    # divide by eijkabc
    r3_divide_e_(mycc, r3, eris)
    time1 = log.timer_debug1('t3: divide r3 by eijkabc', *time1)

    res_norm.append(np.linalg.norm(r3))

    t3 += r3
    r3 = None
    time1 = log.timer_debug1('t3: update t3', *time1)
    time0 = log.timer_debug1('t3 total', *time0)

    # t4
    intermediates_t4_(mycc, t2, t3, t4)
    mycc.t1_eris = None
    time1 = log.timer_debug1('t4: update intermediates', *time0)
    r4 = compute_r4(mycc, t2, t3, t4)
    time1 = log.timer_debug1('t4: compute r4', *time1)
    # symmetrize r4
    t4_p_sum_ip_(r4, nocc, nvir, 1.0, 0.0)
    t4_symm_ip_(r4, nocc**4, nvir, "11111111", -1.0 / 24.0, 1.0)
    purify_tamps_(r4)
    time1 = log.timer_debug1('t4: symmetrize r4', *time1)
    # divide by eijkabc
    r4_divide_e_(mycc, r4, eris)
    time1 = log.timer_debug1('t4: divide r4 by eijklabcd', *time1)

    res_norm.append(np.linalg.norm(r4))

    t4_add_(t4, r4, nocc**4, nvir)
    r4 = None
    time1 = log.timer_debug1('t4: update t4', *time1)
    time0 = log.timer_debug1('t4 total', *time0)
    return res_norm


class RCCSDTQ(rccsdtq.RCCSDTQ):

    do_tril_maxT = getattr(__config__, 'cc_rccsdtq_highm_RCCSDTQ_do_tril_maxT', False)

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        rccsdtq.RCCSDTQ.__init__(self, mf, frozen, mo_coeff, mo_occ)

    update_amps_ = update_amps_rccsdtq_


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
    mycc.set_einsum_backend('pytblis')
    mycc.conv_tol = 1e-12
    mycc.conv_tol_normt = 1e-10
    mycc.max_cycle = 100
    mycc.verbose = 5
    mycc.do_diis_maxT = True
    ecorr, tamps = mycc.kernel()
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
    # mycc.do_diis_maxT = True
    # ecorr, tamps = mycc.kernel()
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
    # mycc.do_diis_maxT = True
    # ecorr, tamps = mycc.kernel()
    # print("My E_corr: % .10f    Ref E_corr: % .10f    Diff: % .10e"%(ecorr, ref_ecorr, ecorr - ref_ecorr))
    # print()
