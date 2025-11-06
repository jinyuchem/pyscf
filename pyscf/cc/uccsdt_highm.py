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
UHF-CCSDT with full T3 amplitudes stored
'''

import numpy as np
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.mp.mp2 import get_e_hf
from pyscf.mp.ump2 import get_nocc, get_nmo, get_frozen_mask
from pyscf.cc import uccsdt
from pyscf.cc.rccsdt import einsum_, run_diis, _finalize
from pyscf.cc.uccsdt import (update_t1_fock_eris_uhf_, intermediates_t1t2_uhf_, compute_r1r2_uhf,
                                antisymmetrize_r2_uhf_, r1r2_divide_e_uhf_, intermediates_t3_uhf_)
from pyscf import __config__


def r1r2_add_t3_uhf_(mycc, t3, r1, r2):
    '''add the contributions from t3 amplitudes to r1r2'''
    nocca, noccb = mycc.nocca, mycc.noccb
    t1_focka, t1_fockb = mycc.t1_focka, mycc.t1_fockb
    t1_erisaa, t1_erisab, t1_erisbb = mycc.t1_erisaa, mycc.t1_erisab, mycc.t1_erisbb
    t3aaa, t3aab, t3bba, t3bbb = t3

    (r1a, r1b), (r2aa, r2ab, r2bb) = r1, r2
    # r1a
    einsum_(mycc, 'mnef,imnaef->ia', t1_erisaa[:nocca, :nocca, nocca:, nocca:], t3aaa, out=r1a, alpha=0.25, beta=1.0)
    einsum_(mycc, 'nmfe,inafme->ia', t1_erisab[:nocca, :noccb, nocca:, noccb:], t3aab, out=r1a, alpha=1.0, beta=1.0)
    einsum_(mycc, 'mnef,mnefia->ia', t1_erisbb[:noccb, :noccb, noccb:, noccb:], t3bba, out=r1a, alpha=0.25, beta=1.0)
    # r1b
    einsum_(mycc, 'mnef,imnaef->ia', t1_erisbb[:noccb, :noccb, noccb:, noccb:], t3bbb, out=r1b, alpha=0.25, beta=1.0)
    einsum_(mycc, 'mnef,inafme->ia', t1_erisab[:nocca, :noccb, nocca:, noccb:], t3bba, out=r1b, alpha=1.0, beta=1.0)
    einsum_(mycc, 'mnef,mnefia->ia', t1_erisaa[:nocca, :nocca, nocca:, nocca:], t3aab, out=r1b, alpha=0.25, beta=1.0)
    # r2aa
    einsum_(mycc, "me,ijmabe->ijab", t1_focka[:nocca, nocca:], t3aaa, out=r2aa, alpha=0.25, beta=1.0)
    einsum_(mycc, "me,ijabme->ijab", t1_fockb[:noccb, noccb:], t3aab, out=r2aa, alpha=0.25, beta=1.0)
    einsum_(mycc, "bmef,ijmaef->ijab", t1_erisaa[nocca:, :nocca, nocca:, nocca:], t3aaa, out=r2aa, alpha=0.25, beta=1.0)
    einsum_(mycc, "bmef,ijaemf->ijab", t1_erisab[nocca:, :noccb, nocca:, noccb:], t3aab, out=r2aa, alpha=0.5, beta=1.0)
    einsum_(mycc, "mnje,imnabe->ijab", t1_erisaa[:nocca, :nocca, :nocca, nocca:], t3aaa,
            out=r2aa, alpha=-0.25, beta=1.0)
    einsum_(mycc, "mnje,imabne->ijab", t1_erisab[:nocca, :noccb, :nocca, noccb:], t3aab, out=r2aa, alpha=-0.5, beta=1.0)
    # r2ab
    einsum_(mycc, "me,imaejb->iajb", t1_focka[:nocca, nocca:], t3aab, out=r2ab, alpha=1.0, beta=1.0)
    einsum_(mycc, "me,jmbeia->iajb", t1_fockb[:noccb, noccb:], t3bba, out=r2ab, alpha=1.0, beta=1.0)
    einsum_(mycc, "mbfe,imafje->iajb", t1_erisab[:nocca, noccb:, nocca:, noccb:], t3aab, out=r2ab, alpha=1.0, beta=1.0)
    einsum_(mycc, "bmef,jmefia->iajb", t1_erisbb[noccb:, :noccb, noccb:, noccb:], t3bba, out=r2ab, alpha=0.5, beta=1.0)
    einsum_(mycc, "amef,imefjb->iajb", t1_erisaa[nocca:, :nocca, nocca:, nocca:], t3aab, out=r2ab, alpha=0.5, beta=1.0)
    einsum_(mycc, "amef,jmbfie->iajb", t1_erisab[nocca:, :noccb, nocca:, noccb:], t3bba, out=r2ab, alpha=1.0, beta=1.0)
    einsum_(mycc, "nmej,inaemb->iajb", t1_erisab[:nocca, :noccb, nocca:, :noccb], t3aab, out=r2ab, alpha=-1.0, beta=1.0)
    einsum_(mycc, "mnje,mnbeia->iajb", t1_erisbb[:noccb, :noccb, :noccb, noccb:], t3bba, out=r2ab, alpha=-0.5, beta=1.0)
    einsum_(mycc, "mnie,mnaejb->iajb", t1_erisaa[:nocca, :nocca, :nocca, nocca:], t3aab, out=r2ab, alpha=-0.5, beta=1.0)
    einsum_(mycc, "mnie,jnbema->iajb", t1_erisab[:nocca, :noccb, :nocca, noccb:], t3bba, out=r2ab, alpha=-1.0, beta=1.0)
    # r2bb
    einsum_(mycc, "me,ijmabe->ijab", t1_fockb[:noccb, noccb:], t3bbb, out=r2bb, alpha=0.25, beta=1.0)
    einsum_(mycc, "me,ijabme->ijab", t1_focka[:nocca, nocca:], t3bba, out=r2bb, alpha=0.25, beta=1.0)
    einsum_(mycc, "bmef,ijmaef->ijab", t1_erisbb[noccb:, :noccb, noccb:, noccb:], t3bbb, out=r2bb, alpha=0.25, beta=1.0)
    einsum_(mycc, "mbfe,ijaemf->ijab", t1_erisab[:nocca, noccb:, nocca:, noccb:], t3bba, out=r2bb, alpha=0.5, beta=1.0)
    einsum_(mycc, "mnje,imnabe->ijab", t1_erisbb[:noccb, :noccb, :noccb, noccb:], t3bbb,
            out=r2bb, alpha=-0.25, beta=1.0)
    einsum_(mycc, "nmej,imabne->ijab", t1_erisab[:nocca, :noccb, nocca:, :noccb], t3bba, out=r2bb, alpha=-0.5, beta=1.0)

def intermediates_t3_add_t3_uhf_(mycc, t3):
    '''add the contributions of t3 to t3 intermediates'''
    nocca, noccb = mycc.nocca, mycc.noccb
    t1_erisaa, t1_erisab, t1_erisbb = mycc.t1_erisaa, mycc.t1_erisab, mycc.t1_erisbb
    t3aaa, t3aab, t3bba, t3bbb = t3

    oaoavava = (slice(None, nocca), slice(None, nocca), slice(nocca, None), slice(nocca, None))
    oaobvavb = (slice(None, nocca), slice(None, noccb), slice(nocca, None), slice(noccb, None))
    obobvbvb = (slice(None, noccb), slice(None, noccb), slice(noccb, None), slice(noccb, None))
    einsum_(mycc, 'lmde,lmkbec->bcdk', t1_erisaa[oaoavava], t3aaa, out=mycc.W_vvvo, alpha=-0.5, beta=1.0)
    einsum_(mycc, 'lmde,lkbcme->bcdk', t1_erisab[oaobvavb], t3aab, out=mycc.W_vvvo, alpha=-1.0, beta=1.0)
    einsum_(mycc, 'lmde,jmkdec->lcjk', t1_erisaa[oaoavava], t3aaa, out=mycc.W_ovoo, alpha=0.5, beta=1.0)
    einsum_(mycc, 'lmde,jkdcme->lcjk', t1_erisab[oaobvavb], t3aab, out=mycc.W_ovoo, alpha=1.0, beta=1.0)
    einsum_(mycc, 'lmde,lmkbec->bcdk', t1_erisbb[obobvbvb], t3bbb, out=mycc.W_VVVO, alpha=-0.5, beta=1.0)
    einsum_(mycc, 'mled,lkbcme->bcdk', t1_erisab[oaobvavb], t3bba, out=mycc.W_VVVO, alpha=-1.0, beta=1.0)
    einsum_(mycc, 'lmde,jmkdec->lcjk', t1_erisbb[obobvbvb], t3bbb, out=mycc.W_OVOO, alpha=0.5, beta=1.0)
    einsum_(mycc, 'mled,jkdcme->lcjk', t1_erisab[oaobvavb], t3bba, out=mycc.W_OVOO, alpha=1.0, beta=1.0)
    einsum_(mycc, 'lmde,lmbekc->bcdk', t1_erisaa[oaoavava], t3aab, out=mycc.W_vVvO, alpha=-0.5, beta=1.0)
    einsum_(mycc, 'lmde,mkeclb->bcdk', t1_erisab[oaobvavb], t3bba, out=mycc.W_vVvO, alpha=-1.0, beta=1.0)
    einsum_(mycc, 'lmde,jmdekc->lcjk', t1_erisaa[oaoavava], t3aab, out=mycc.W_oVoO, alpha=0.5, beta=1.0)
    einsum_(mycc, 'lmde,mkecjd->lcjk', t1_erisab[oaobvavb], t3bba, out=mycc.W_oVoO, alpha=1.0, beta=1.0)
    einsum_(mycc, 'mled,jmbelc->bcjd', t1_erisab[oaobvavb], t3aab, out=mycc.W_vVoV, alpha=-1.0, beta=1.0)
    einsum_(mycc, 'lmde,mlecjb->bcjd', t1_erisbb[obobvbvb], t3bba, out=mycc.W_vVoV, alpha=-0.5, beta=1.0)
    einsum_(mycc, 'mled,jmaekd->aljk', t1_erisab[oaobvavb], t3aab, out=mycc.W_vOoO, alpha=1.0, beta=1.0)
    einsum_(mycc, 'lmde,mkedja->aljk', t1_erisbb[obobvbvb], t3bba, out=mycc.W_vOoO, alpha=0.5, beta=1.0)
    return mycc

def compute_r3_uhf(mycc, t2, t3):
    time1 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    t2aa, t2ab, t2bb = t2
    t3aaa, t3aab, t3bba, t3bbb = t3

    # aaa
    r3aaa = np.empty_like(t3aaa)
    einsum_(mycc, "bcdk,ijad->ijkabc", mycc.W_vvvo, t2aa, out=r3aaa, alpha=0.25, beta=0.0)
    einsum_(mycc, "lcjk,ilab->ijkabc", mycc.W_ovoo, t2aa, out=r3aaa, alpha=-0.25, beta=1.0)
    einsum_(mycc, "cd,ijkabd->ijkabc", mycc.tf_vv, t3aaa, out=r3aaa, alpha=1.0 / 12.0, beta=1.0)
    einsum_(mycc, "lk,ijlabc->ijkabc", mycc.tf_oo, t3aaa, out=r3aaa, alpha=-1.0 / 12.0, beta=1.0)
    einsum_(mycc, "abde,ijkdec->ijkabc", mycc.W_vvvv, t3aaa, out=r3aaa, alpha=1.0 / 24.0, beta=1.0)
    einsum_(mycc, "lmij,lmkabc->ijkabc", mycc.W_oooo, t3aaa, out=r3aaa, alpha=1.0 / 24.0, beta=1.0)
    einsum_(mycc, "alid,ljkdbc->ijkabc", mycc.W_voov, t3aaa, out=r3aaa, alpha=0.25, beta=1.0)
    einsum_(mycc, "alid,jkbcld->ijkabc", mycc.W_vOoV, t3aab, out=r3aaa, alpha=0.25, beta=1.0)
    time1 = log.timer_debug1('t3: r3aaa', *time1)

    # bbb
    r3bbb = np.empty_like(t3bbb)
    einsum_(mycc, "bcdk,ijad->ijkabc", mycc.W_VVVO, t2bb, out=r3bbb, alpha=0.25, beta=0.0)
    einsum_(mycc, "lcjk,ilab->ijkabc", mycc.W_OVOO, t2bb, out=r3bbb, alpha=-0.25, beta=1.0)
    einsum_(mycc, "cd,ijkabd->ijkabc", mycc.tf_VV, t3bbb, out=r3bbb, alpha=1.0 / 12.0, beta=1.0)
    einsum_(mycc, "lk,ijlabc->ijkabc", mycc.tf_OO, t3bbb, out=r3bbb, alpha=-1.0 / 12.0, beta=1.0)
    einsum_(mycc, "abde,ijkdec->ijkabc", mycc.W_VVVV, t3bbb, out=r3bbb, alpha=1.0 / 24.0, beta=1.0)
    einsum_(mycc, "lmij,lmkabc->ijkabc", mycc.W_OOOO, t3bbb, out=r3bbb, alpha=1.0 / 24.0, beta=1.0)
    einsum_(mycc, "alid,ljkdbc->ijkabc", mycc.W_VOOV, t3bbb, out=r3bbb, alpha=0.25, beta=1.0)
    einsum_(mycc, "alid,jkbcld->ijkabc", mycc.W_VoOv, t3bba, out=r3bbb, alpha=0.25, beta=1.0)
    time1 = log.timer_debug1('t3: r3bbb', *time1)

    # aab
    r3aab = np.empty_like(t3aab)
    einsum_(mycc, "bcdk,ijad->ijabkc", mycc.W_vVvO, t2aa, out=r3aab, alpha=0.5, beta=0.0)
    einsum_(mycc, "bcjd,iakd->ijabkc", mycc.W_vVoV, t2ab, out=r3aab, alpha=1.0, beta=1.0)
    einsum_(mycc, "abdi,jdkc->ijabkc", mycc.W_vvvo, t2ab, out=r3aab, alpha=-0.5, beta=1.0)
    einsum_(mycc, "lcjk,ilab->ijabkc", mycc.W_oVoO, t2aa, out=r3aab, alpha=-0.5, beta=1.0)
    einsum_(mycc, "aljk,iblc->ijabkc", mycc.W_vOoO, t2ab, out=r3aab, alpha=1.0, beta=1.0)
    einsum_(mycc, "laij,lbkc->ijabkc", mycc.W_ovoo, t2ab, out=r3aab, alpha=0.5, beta=1.0)
    einsum_(mycc, "cd,ijabkd->ijabkc", mycc.tf_VV, t3aab, out=r3aab, alpha=0.25, beta=1.0)
    einsum_(mycc, "ad,ijbdkc->ijabkc", mycc.tf_vv, t3aab, out=r3aab, alpha=-0.5, beta=1.0)
    einsum_(mycc, "lk,ijablc->ijabkc", mycc.tf_OO, t3aab, out=r3aab, alpha=-0.25, beta=1.0)
    einsum_(mycc, "li,jlabkc->ijabkc", mycc.tf_oo, t3aab, out=r3aab, alpha=0.5, beta=1.0)
    einsum_(mycc, "abde,ijdekc->ijabkc", mycc.W_vvvv, t3aab, out=r3aab, alpha=0.125, beta=1.0)
    einsum_(mycc, "bced,ijaekd->ijabkc", mycc.W_vVvV, t3aab, out=r3aab, alpha=0.5, beta=1.0)
    einsum_(mycc, "lmij,lmabkc->ijabkc", mycc.W_oooo, t3aab, out=r3aab, alpha=0.125, beta=1.0)
    einsum_(mycc, "lmik,ljabmc->ijabkc", mycc.W_oOoO, t3aab, out=r3aab, alpha=0.5, beta=1.0)
    einsum_(mycc, "alid,ljdbkc->ijabkc", mycc.W_voov, t3aab, out=r3aab, alpha=1.0, beta=1.0)
    einsum_(mycc, "alid,lkdcjb->ijabkc", mycc.W_vOoV, t3bba, out=r3aab, alpha=1.0, beta=1.0)
    einsum_(mycc, "lcid,ljabkd->ijabkc", mycc.W_oVoV, t3aab, out=r3aab, alpha=-0.5, beta=1.0)
    einsum_(mycc, "aldk,ijdblc->ijabkc", mycc.W_vOvO_tc, t3aab, out=r3aab, alpha=-0.5, beta=1.0)
    einsum_(mycc, "clkd,ijlabd->ijabkc", mycc.W_VoOv, t3aaa, out=r3aab, alpha=0.25, beta=1.0)
    einsum_(mycc, "clkd,ijabld->ijabkc", mycc.W_VOOV, t3aab, out=r3aab, alpha=0.25, beta=1.0)
    time1 = log.timer_debug1('t3: r3aab', *time1)

    # bba
    r3bba = np.empty_like(t3bba)
    einsum_(mycc, "cbkd,ijad->ijabkc", mycc.W_vVoV, t2bb, out=r3bba, alpha=0.5, beta=0.0)
    einsum_(mycc, "cbdj,kdia->ijabkc", mycc.W_vVvO, t2ab, out=r3bba, alpha=1.0, beta=1.0)
    einsum_(mycc, "abdi,kcjd->ijabkc", mycc.W_VVVO, t2ab, out=r3bba, alpha=-0.5, beta=1.0)
    einsum_(mycc, "clkj,ilab->ijabkc", mycc.W_vOoO, t2bb, out=r3bba, alpha=-0.5, beta=1.0)
    einsum_(mycc, "lakj,lcib->ijabkc", mycc.W_oVoO, t2ab, out=r3bba, alpha=1.0, beta=1.0)
    einsum_(mycc, "laij,kclb->ijabkc", mycc.W_OVOO, t2ab, out=r3bba, alpha=0.5, beta=1.0)
    einsum_(mycc, "cd,ijabkd->ijabkc", mycc.tf_vv, t3bba, out=r3bba, alpha=0.25, beta=1.0)
    einsum_(mycc, "ad,ijbdkc->ijabkc", mycc.tf_VV, t3bba, out=r3bba, alpha=-0.5, beta=1.0)
    einsum_(mycc, "lk,ijablc->ijabkc", mycc.tf_oo, t3bba, out=r3bba, alpha=-0.25, beta=1.0)
    einsum_(mycc, "li,jlabkc->ijabkc", mycc.tf_OO, t3bba, out=r3bba, alpha=0.5, beta=1.0)
    einsum_(mycc, "abde,ijdekc->ijabkc", mycc.W_VVVV, t3bba, out=r3bba, alpha=0.125, beta=1.0)
    einsum_(mycc, "cbde,ijaekd->ijabkc", mycc.W_vVvV, t3bba, out=r3bba, alpha=0.5, beta=1.0)
    einsum_(mycc, "lmij,lmabkc->ijabkc", mycc.W_OOOO, t3bba, out=r3bba, alpha=0.125, beta=1.0)
    einsum_(mycc, "mlki,ljabmc->ijabkc", mycc.W_oOoO, t3bba, out=r3bba, alpha=0.5, beta=1.0)
    einsum_(mycc, "alid,ljdbkc->ijabkc", mycc.W_VOOV, t3bba, out=r3bba, alpha=1.0, beta=1.0)
    einsum_(mycc, "alid,lkdcjb->ijabkc", mycc.W_VoOv, t3aab, out=r3bba, alpha=1.0, beta=1.0)
    einsum_(mycc, "cldi,ljabkd->ijabkc", mycc.W_vOvO_tc, t3bba, out=r3bba, alpha=-0.5, beta=1.0)
    einsum_(mycc, "lakd,ijdblc->ijabkc", mycc.W_oVoV, t3bba, out=r3bba, alpha=-0.5, beta=1.0)
    einsum_(mycc, "clkd,ijlabd->ijabkc", mycc.W_vOoV, t3bbb, out=r3bba, alpha=0.25, beta=1.0)
    einsum_(mycc, "clkd,ijabld->ijabkc", mycc.W_voov, t3bba, out=r3bba, alpha=0.25, beta=1.0)
    time1 = log.timer_debug1('t3: r3bba', *time1)
    return [r3aaa, r3aab, r3bba, r3bbb]

def antisymmetrize_r3_uhf_(r3):
    r3[0] = (r3[0] - r3[0].transpose(1, 0, 2, 3, 4, 5) - r3[0].transpose(0, 2, 1, 3, 4, 5)
    - r3[0].transpose(2, 1, 0, 3, 4, 5) + r3[0].transpose(1, 2, 0, 3, 4, 5) + r3[0].transpose(2, 0, 1, 3, 4, 5))
    r3[0] = (r3[0] - r3[0].transpose(0, 1, 2, 4, 3, 5) - r3[0].transpose(0, 1, 2, 3, 5, 4)
    - r3[0].transpose(0, 1, 2, 5, 4, 3) + r3[0].transpose(0, 1, 2, 4, 5, 3) + r3[0].transpose(0, 1, 2, 5, 3, 4))
    r3[1] -= r3[1].transpose(1, 0, 2, 3, 4, 5)
    r3[1] -= r3[1].transpose(0, 1, 3, 2, 4, 5)
    r3[2] -= r3[2].transpose(1, 0, 2, 3, 4, 5)
    r3[2] -= r3[2].transpose(0, 1, 3, 2, 4, 5)
    r3[3] = (r3[3] - r3[3].transpose(1, 0, 2, 3, 4, 5) - r3[3].transpose(0, 2, 1, 3, 4, 5)
    - r3[3].transpose(2, 1, 0, 3, 4, 5) + r3[3].transpose(1, 2, 0, 3, 4, 5) + r3[3].transpose(2, 0, 1, 3, 4, 5))
    r3[3] = (r3[3] - r3[3].transpose(0, 1, 2, 4, 3, 5) - r3[3].transpose(0, 1, 2, 3, 5, 4)
    - r3[3].transpose(0, 1, 2, 5, 4, 3) + r3[3].transpose(0, 1, 2, 4, 5, 3) + r3[3].transpose(0, 1, 2, 5, 3, 4))

def r3_divide_e_uhf_(mycc, r3, eris):
    nocca, noccb = r3[0].shape[0], r3[-1].shape[0]
    eia_a = eris.mo_energy[0][:nocca, None] - eris.mo_energy[0][None, nocca:] - mycc.level_shift
    eia_b = eris.mo_energy[1][:noccb, None] - eris.mo_energy[1][None, noccb:] - mycc.level_shift

    eijkabc_aaa = (eia_a[:, None, None, :, None, None] + eia_a[None, :, None, None, :, None]
                    + eia_a[None, None, :, None, None, :])
    r3[0] /= eijkabc_aaa
    eijkabc_aaa = None
    eijkabc_aab = (eia_a[:, None, :, None, None, None] + eia_a[None, :, None, :, None,  None]
                    + eia_b[None, None, None, None, :, :])
    r3[1] /= eijkabc_aab
    eijkabc_aab = None
    eijkabc_bba = (eia_b[:, None, :, None, None, None] + eia_b[None, :, None, :, None, None]
                    + eia_a[None, None, None, None, :, :])
    r3[2] /= eijkabc_bba
    eijkabc_bba = None
    eijkabc_bbb = (eia_b[:, None, None, :, None, None] + eia_b[None, :, None, None, :, None]
                    + eia_b[None, None, :, None, None, :])
    r3[3] /= eijkabc_bbb
    eijkabc_bbb = None

def update_amps_uccsdt_(mycc, tamps, eris):
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    t1, t2, t3 = tamps
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    t3aaa, t3aab, t3bba, t3bbb = t3

    # t1 t2
    update_t1_fock_eris_uhf_(mycc, t1, eris)
    time1 = log.timer_debug1('t1t2: update fock and eris', *time0)
    intermediates_t1t2_uhf_(mycc, t2)
    time1 = log.timer_debug1('t1t2: update intermediates', *time1)
    r1, r2 = compute_r1r2_uhf(mycc, t2)
    r1r2_add_t3_uhf_(mycc, t3, r1, r2)
    time1 = log.timer_debug1('t1t2: compute r1 & r2', *time1)
    # antisymmetrize R2
    antisymmetrize_r2_uhf_(r2)
    time1 = log.timer_debug1('t1t2: antisymmetrize r2', *time1)
    # divide by eijkabc
    r1r2_divide_e_uhf_(mycc, r1, r2, eris)
    (r1a, r1b), (r2aa, r2ab, r2bb) = r1, r2
    time1 = log.timer_debug1('t1t2: divide r1 & r2 by eia & eijab', *time1)

    res_norm = [np.linalg.norm(r1a), np.linalg.norm(r1b),
                np.linalg.norm(r2aa), np.linalg.norm(r2ab), np.linalg.norm(r2bb)]

    t1a += r1a
    t1b += r1b
    t2aa += r2aa
    t2ab += r2ab
    t2bb += r2bb
    time1 = log.timer_debug1('t1t2: update t1 & t2', *time1)
    time0 = log.timer_debug1('t1t2 total', *time0)

    # t3
    intermediates_t3_uhf_(mycc, t2)
    intermediates_t3_add_t3_uhf_(mycc, t3)
    mycc.t1_erisaa, mycc.t1_erisab, mycc.t1_erisbb = None, None, None
    time1 = log.timer_debug1('t3: update intermediates', *time0)
    r3 = compute_r3_uhf(mycc, t2, t3)
    time1 = log.timer_debug1('t3: compute r3', *time1)
    # antisymmetrize r3
    antisymmetrize_r3_uhf_(r3)
    time1 = log.timer_debug1('t3: antisymmetrize r3', *time1)
    # divide by eijkabc
    r3_divide_e_uhf_(mycc, r3, eris)
    r3aaa, r3aab, r3bba, r3bbb = r3
    time1 = log.timer_debug1('t3: divide r3 by eijkabc', *time1)

    res_norm += [np.linalg.norm(r3aaa), + np.linalg.norm(r3aab), np.linalg.norm(r3bba), + np.linalg.norm(r3bbb)]

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

    tamps = (t1, t2, t3)
    return res_norm


class UCCSDT(uccsdt.UCCSDT):

    do_tril_maxT = getattr(__config__, 'cc_uccsdt_UCCSDT_do_tril_maxT', False)

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        super().__init__(mf, frozen, mo_coeff, mo_occ)

    update_amps_ = update_amps_uccsdt_


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
        mycc.set_einsum_backend('pytblis')
        mycc.conv_tol = 1e-12
        mycc.conv_tol_normt = 1e-10
        mycc.max_cycle = 100
        mycc.verbose = 5
        mycc.do_diis_maxT = True
        ecorr, tamps = mycc.kernel()
        print("My E_corr: % .16f    Ref E_corr: % .16f    Diff: % .16e"%(ecorr, ref_ecorr, ecorr - ref_ecorr))
        print()
