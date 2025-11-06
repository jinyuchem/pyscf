/* Copyright 2014-2025 The PySCF Developers. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * Author: Yu Jin <yjin@flatironinstitute.org>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

// Signs for 6-fold permutations (matches your Python code)
const double sign6[6] = {1.0, -1.0, -1.0, 1.0, 1.0, -1.0};

// Signs for 2-fold permutations
const double sign2[2] = {1.0, -1.0};

// Unpack 6-fold anti-symmetric tensor
void unpack_6fold_antisymm_c(const double *restrict t3_tril,
                             double *restrict t3_blk,
                             const int64_t *restrict map_o,
                             const bool *restrict mask_o,
                             const int64_t *restrict map_v,
                             const bool *restrict mask_v,
                             int64_t i0, int64_t i1,
                             int64_t j0, int64_t j1,
                             int64_t k0, int64_t k1,
                             int64_t a0, int64_t a1,
                             int64_t b0, int64_t b1,
                             int64_t c0, int64_t c1,
                             int64_t nocc, int64_t nvir,
                             int64_t blk_i, int64_t blk_j, int64_t blk_k,
                             int64_t blk_a, int64_t blk_b, int64_t blk_c)
{
#define MAP_O(sym, x, y, z) map_o[(((sym) * nocc + (x)) * nocc + (y)) * nocc + (z)]
#define MASK_O(sym, x, y, z) mask_o[(((sym) * nocc + (x)) * nocc + (y)) * nocc + (z)]
#define MAP_V(sym, x, y, z) map_v[(((sym) * nvir + (x)) * nvir + (y)) * nvir + (z)]
#define MASK_V(sym, x, y, z) mask_v[(((sym) * nvir + (x)) * nvir + (y)) * nvir + (z)]

    // Compute no3
    int64_t no3 = nvir * (nvir - 1) * (nvir - 2) / 6;

    // Zero out the block
    int64_t blk_size = blk_i * blk_j * blk_k * blk_a * blk_b * blk_c;
    memset(t3_blk, 0, blk_size * sizeof(double));

#pragma omp parallel for collapse(4) schedule(dynamic)
    for (int64_t sym_o = 0; sym_o < 6; ++sym_o)
    {
        for (int64_t i = i0; i < i1; ++i)
        {
            for (int64_t j = j0; j < j1; ++j)
            {
                for (int64_t k = k0; k < k1; ++k)
                {
                    if (!MASK_O(sym_o, i, j, k))
                        continue;

                    double sign_o = sign6[sym_o];
                    int64_t loc_i = i - i0;
                    int64_t loc_j = j - j0;
                    int64_t loc_k = k - k0;
                    int64_t o_idx = MAP_O(sym_o, i, j, k);

                    for (int64_t sym_v = 0; sym_v < 6; ++sym_v)
                    {
                        double sign_v = sign6[sym_v];
                        double sign_xy = sign_o * sign_v;

                        for (int64_t a = a0; a < a1; ++a)
                        {
                            for (int64_t b = b0; b < b1; ++b)
                            {
                                for (int64_t c = c0; c < c1; ++c)
                                {
                                    if (!MASK_V(sym_v, a, b, c))
                                        continue;

                                    int64_t v_idx = MAP_V(sym_v, a, b, c);
                                    int64_t loc_a = a - a0;
                                    int64_t loc_b = b - b0;
                                    int64_t loc_c = c - c0;

                                    // No permutation needed - mask already handles it
                                    int64_t src_idx = o_idx * no3 + v_idx;
                                    int64_t dest_idx = ((((loc_i * blk_j + loc_j) * blk_k + loc_k) * blk_a + loc_a) * blk_b + loc_b) * blk_c + loc_c;

                                    t3_blk[dest_idx] = t3_tril[src_idx] * sign_xy;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
#undef MAP_O
#undef MASK_O
#undef MAP_V
#undef MASK_V
}

// Update packed 6-fold anti-symmetric tensor
void update_packed_6fold_antisymm_c(double *restrict t3_tril,
                                    const double *restrict t3_blk,
                                    const int64_t *restrict map_o,
                                    const int64_t *restrict map_v,
                                    int64_t i0, int64_t i1,
                                    int64_t j0, int64_t j1,
                                    int64_t k0, int64_t k1,
                                    int64_t a0, int64_t a1,
                                    int64_t b0, int64_t b1,
                                    int64_t c0, int64_t c1,
                                    int64_t nocc, int64_t nvir,
                                    int64_t blk_i, int64_t blk_j, int64_t blk_k,
                                    int64_t blk_a, int64_t blk_b, int64_t blk_c,
                                    double alpha, double beta)
{
#define MAP_O(sym, x, y, z) map_o[(((sym) * nocc + (x)) * nocc + (y)) * nocc + (z)]
#define MAP_V(sym, x, y, z) map_v[(((sym) * nvir + (x)) * nvir + (y)) * nvir + (z)]

    if (j1 < i0 || k1 < j0)
        return;
    if (b1 < a0 || c1 < b0)
        return;

    // Compute no3
    int64_t no3 = nvir * (nvir - 1) * (nvir - 2) / 6;

#pragma omp parallel for collapse(3) schedule(dynamic)
    for (int64_t k = k0; k < k1; ++k)
    {
        for (int64_t j = j0; j < j1; ++j)
        {
            for (int64_t i = i0; i < i1; ++i)
            {
                if (i >= j || j >= k)
                    continue;

                int64_t o_idx = MAP_O(0, i, j, k);
                int64_t loc_i = i - i0;
                int64_t loc_j = j - j0;
                int64_t loc_k = k - k0;

                if (beta == 0)
                {
                    for (int64_t c = c0; c < c1; ++c)
                    {
                        for (int64_t b = b0; b < b1; ++b)
                        {
                            for (int64_t a = a0; a < a1; ++a)
                            {
                                if (a >= b || b >= c)
                                    continue;

                                int64_t v_idx = MAP_V(0, a, b, c);
                                int64_t loc_a = a - a0;
                                int64_t loc_b = b - b0;
                                int64_t loc_c = c - c0;

                                int64_t tril_idx = o_idx * no3 + v_idx;
                                int64_t blk_idx = ((((loc_i * blk_j + loc_j) * blk_k + loc_k) * blk_a + loc_a) * blk_b + loc_b) * blk_c + loc_c;

                                t3_tril[tril_idx] = alpha * t3_blk[blk_idx];
                            }
                        }
                    }
                }
                else
                {
                    for (int64_t c = c0; c < c1; ++c)
                    {
                        for (int64_t b = b0; b < b1; ++b)
                        {
                            for (int64_t a = a0; a < a1; ++a)
                            {
                                if (a >= b || b >= c)
                                    continue;

                                int64_t v_idx = MAP_V(0, a, b, c);
                                int64_t loc_a = a - a0;
                                int64_t loc_b = b - b0;
                                int64_t loc_c = c - c0;

                                int64_t tril_idx = o_idx * no3 + v_idx;
                                int64_t blk_idx = ((((loc_i * blk_j + loc_j) * blk_k + loc_k) * blk_a + loc_a) * blk_b + loc_b) * blk_c + loc_c;

                                t3_tril[tril_idx] = beta * t3_tril[tril_idx] + alpha * t3_blk[blk_idx];
                            }
                        }
                    }
                }
            }
        }
    }
#undef MAP_O
#undef MAP_V
}

// Unpack 2-fold anti-symmetric tensor
void unpack_2fold_antisymm_c(const double *restrict t3_tril,
                             double *restrict t3_blk,
                             const int64_t *restrict map_o,
                             const bool *restrict mask_o,
                             const int64_t *restrict map_v,
                             const bool *restrict mask_v,
                             int64_t i0, int64_t i1,
                             int64_t j0, int64_t j1,
                             int64_t a0, int64_t a1,
                             int64_t b0, int64_t b1,
                             int64_t k0, int64_t k1,
                             int64_t c0, int64_t c1,
                             int64_t nocc, int64_t nvir,
                             int64_t dim4, int64_t dim5,
                             int64_t blk_i, int64_t blk_j,
                             int64_t blk_a, int64_t blk_b)
{
#define MAP_O(sym, x, y) map_o[((sym) * nocc + (x)) * nocc + (y)]
#define MASK_O(sym, x, y) mask_o[((sym) * nocc + (x)) * nocc + (y)]
#define MAP_V(sym, x, y) map_v[((sym) * nvir + (x)) * nvir + (y)]
#define MASK_V(sym, x, y) mask_v[((sym) * nvir + (x)) * nvir + (y)]

    // Compute no2
    int64_t no2 = nvir * (nvir - 1) / 2;

    // Zero out the block
    int64_t blk_size = blk_i * blk_j * blk_a * blk_b * dim4 * dim5;
    memset(t3_blk, 0, blk_size * sizeof(double));

#pragma omp parallel for collapse(3) schedule(dynamic)
    for (int64_t sym_o = 0; sym_o < 2; ++sym_o)
    {
        for (int64_t i = i0; i < i1; ++i)
        {
            for (int64_t j = j0; j < j1; ++j)
            {
                if (!MASK_O(sym_o, i, j))
                    continue;

                double sign_o = sign2[sym_o];
                int64_t loc_i = i - i0;
                int64_t loc_j = j - j0;
                int64_t o_idx = MAP_O(sym_o, i, j);

                for (int64_t sym_v = 0; sym_v < 2; ++sym_v)
                {
                    double sign_v = sign2[sym_v];
                    double sign_xy = sign_o * sign_v;

                    for (int64_t a = a0; a < a1; ++a)
                    {
                        for (int64_t b = b0; b < b1; ++b)
                        {
                            if (!MASK_V(sym_v, a, b))
                                continue;

                            int64_t v_idx = MAP_V(sym_v, a, b);
                            int64_t loc_a = a - a0;
                            int64_t loc_b = b - b0;

                            // No permutation needed - mask already handles it
                            for (int64_t d4 = k0; d4 < k1; ++d4)
                            {
                                for (int64_t d5 = c0; d5 < c1; ++d5)
                                {
                                    int64_t src_idx = ((o_idx * no2 + v_idx) * dim4 + d4) * dim5 + d5;
                                    int64_t dest_idx = ((((loc_i * blk_j + loc_j) * blk_a + loc_a) * blk_b + loc_b) * dim4 + d4) * dim5 + d5;

                                    t3_blk[dest_idx] = t3_tril[src_idx] * sign_xy;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
#undef MAP_O
#undef MASK_O
#undef MAP_V
#undef MASK_V
}

// Update packed 2-fold anti-symmetric tensor
void update_packed_2fold_antisymm_c(double *restrict t3_tril,
                                    const double *restrict t3_blk,
                                    const int64_t *restrict map_o,
                                    const int64_t *restrict map_v,
                                    int64_t i0, int64_t i1,
                                    int64_t j0, int64_t j1,
                                    int64_t a0, int64_t a1,
                                    int64_t b0, int64_t b1,
                                    int64_t k0, int64_t k1,
                                    int64_t c0, int64_t c1,
                                    int64_t nocc, int64_t nvir,
                                    int64_t dim4, int64_t dim5,
                                    int64_t blk_i, int64_t blk_j,
                                    int64_t blk_a, int64_t blk_b,
                                    double alpha, double beta)
{
#define MAP_O(sym, x, y) map_o[((sym) * nocc + (x)) * nocc + (y)]
#define MAP_V(sym, x, y) map_v[((sym) * nvir + (x)) * nvir + (y)]

    if (j1 < i0)
        return;
    if (b1 < a0)
        return;

    // Compute no2
    int64_t no2 = nvir * (nvir - 1) / 2;

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int64_t j = j0; j < j1; ++j)
    {
        for (int64_t i = i0; i < i1; ++i)
        {
            if (i >= j)
                continue;

            int64_t o_idx = MAP_O(0, i, j);
            int64_t loc_i = i - i0;
            int64_t loc_j = j - j0;

            for (int64_t b = b0; b < b1; ++b)
            {
                for (int64_t a = a0; a < a1; ++a)
                {
                    if (a >= b)
                        continue;

                    int64_t v_idx = MAP_V(0, a, b);
                    int64_t loc_a = a - a0;
                    int64_t loc_b = b - b0;

                    if (beta == 0)
                    {
                        for (int64_t d4 = k0; d4 < k1; ++d4)
                        {
                            for (int64_t d5 = c0; d5 < c1; ++d5)
                            {
                                int64_t tril_idx = ((o_idx * no2 + v_idx) * dim4 + d4) * dim5 + d5;
                                int64_t blk_idx = ((((loc_i * blk_j + loc_j) * blk_a + loc_a) * blk_b + loc_b) * dim4 + d4) * dim5 + d5;

                                t3_tril[tril_idx] = alpha * t3_blk[blk_idx];
                            }
                        }
                    }
                    else
                    {
                        for (int64_t d4 = k0; d4 < k1; ++d4)
                        {
                            for (int64_t d5 = c0; d5 < c1; ++d5)
                            {
                                int64_t tril_idx = ((o_idx * no2 + v_idx) * dim4 + d4) * dim5 + d5;
                                int64_t blk_idx = ((((loc_i * blk_j + loc_j) * blk_a + loc_a) * blk_b + loc_b) * dim4 + d4) * dim5 + d5;

                                t3_tril[tril_idx] = beta * t3_tril[tril_idx] + alpha * t3_blk[blk_idx];
                            }
                        }
                    }
                }
            }
        }
    }
#undef MAP_O
#undef MAP_V
}
