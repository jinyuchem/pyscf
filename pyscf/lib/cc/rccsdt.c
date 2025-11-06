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

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// A unified function for the symmetry operations
void t3_symm_ip_c(double *A, int64_t nocc3, int64_t nvir, char *pattern, double alpha, double beta)
{
    int64_t ijk;
    // test bl
    const int64_t bl = 16;
    int64_t nvv = nvir * nvir;
    double p0 = 0.0, p1 = 0.0, p2 = 0.0, p3 = 0.0, p4 = 0.0;

    if (strcmp(pattern, "111111") == 0)
    {
        // abc + acb + bac + bca + cab + cba
        // (1 + P_b^c) (1 + P_a^b + P_a^c) abc
        p0 = 1.0;
        p1 = 1.0;
        p2 = 1.0;
        p3 = 1.0;
        p4 = 1.0;
    }
    else if (strcmp(pattern, "4-2-211-2") == 0)
    {
        // 4 abc - 2 acb - 2 bac + bca + cab - 2 cba
        // (2 - P_b^c) (2 - P_a^b - P_a^c) abc
        p0 = 2.0;
        p1 = -1.0;
        p2 = -1.0;
        p3 = 2.0;
        p4 = -1.0;
    }
    else if (strcmp(pattern, "20-100-1") == 0)
    {
        // 2 * abc + 0 * acb - bac + 0 * bca + 0 * cab - cba
        // (1 - 0 P_b^c) (2 - P_a^b - P_a^c) abc
        p0 = 2.0;
        p1 = -1.0;
        p2 = -1.0;
        p3 = 1.0;
        p4 = 0.0;
    }
    else
    {
        fprintf(stderr, "Error: unrecognized pattern \"%s\"\n", pattern);
        return;
    }

#pragma omp parallel for schedule(static)
    for (ijk = 0; ijk < nocc3; ijk++)
    {
        int64_t h = ijk * nvir * nvir * nvir;
        for (int64_t a0 = 0; a0 < nvir; a0 += bl)
        {
            for (int64_t b0 = 0; b0 <= a0; b0 += bl)
            {
                for (int64_t c0 = 0; c0 <= b0; c0 += bl)
                {
                    for (int64_t a = a0; a < a0 + bl && a < nvir; a++)
                    {
                        for (int64_t b = b0; b < b0 + bl && b <= a; b++)
                        {
                            for (int64_t c = c0; c < c0 + bl && c <= b; c++)
                            {

                                if (a > b && b > c)
                                {
                                    double T_local[6];

                                    int64_t idx1 = a * nvv + b * nvir + c; // abc
                                    int64_t idx2 = c * nvv + b * nvir + a; // cba
                                    int64_t idx3 = a * nvv + c * nvir + b; // acb
                                    int64_t idx4 = b * nvv + a * nvir + c; // bac
                                    int64_t idx5 = b * nvv + c * nvir + a; // bca
                                    int64_t idx6 = c * nvv + a * nvir + b; // cab

                                    T_local[0] = p0 * A[h + idx1] + p1 * A[h + idx2] + p2 * A[h + idx4]; // abc -> cba -> bac
                                    T_local[1] = p0 * A[h + idx2] + p1 * A[h + idx1] + p2 * A[h + idx5]; // cba -> abc -> bca
                                    T_local[2] = p0 * A[h + idx3] + p1 * A[h + idx5] + p2 * A[h + idx6]; // acb -> bca -> cab
                                    T_local[3] = p0 * A[h + idx4] + p1 * A[h + idx6] + p2 * A[h + idx1]; // bac -> cab -> abc
                                    T_local[4] = p0 * A[h + idx5] + p1 * A[h + idx3] + p2 * A[h + idx2]; // bca -> acb -> cba
                                    T_local[5] = p0 * A[h + idx6] + p1 * A[h + idx4] + p2 * A[h + idx3]; // cab -> bac -> acb

                                    A[h + idx1] = beta * A[h + idx1] + alpha * (p3 * T_local[0] + p4 * T_local[2]); // abc -> acb
                                    A[h + idx2] = beta * A[h + idx2] + alpha * (p3 * T_local[1] + p4 * T_local[5]); // cba -> cab
                                    A[h + idx3] = beta * A[h + idx3] + alpha * (p3 * T_local[2] + p4 * T_local[0]); // acb -> abc
                                    A[h + idx4] = beta * A[h + idx4] + alpha * (p3 * T_local[3] + p4 * T_local[4]); // bac -> bca
                                    A[h + idx5] = beta * A[h + idx5] + alpha * (p3 * T_local[4] + p4 * T_local[3]); // bca -> bac
                                    A[h + idx6] = beta * A[h + idx6] + alpha * (p3 * T_local[5] + p4 * T_local[1]); // cab -> cba
                                }

                                else if (a > b && b == c)
                                {
                                    double T_local[3];

                                    int64_t idx1 = a * nvv + b * nvir + b; // abb
                                    int64_t idx2 = b * nvv + b * nvir + a; // bba
                                    int64_t idx4 = b * nvv + a * nvir + b; // bab

                                    T_local[0] = p0 * A[h + idx1] + p1 * A[h + idx2] + p2 * A[h + idx4]; // abc -> cba -> bac
                                    T_local[1] = p0 * A[h + idx2] + p1 * A[h + idx1] + p2 * A[h + idx2]; // cba -> abc -> bca
                                    T_local[2] = p0 * A[h + idx4] + p1 * A[h + idx4] + p2 * A[h + idx1]; // bac -> cab -> abc

                                    A[h + idx1] = beta * A[h + idx1] + alpha * (p3 * T_local[0] + p4 * T_local[0]); // abc -> acb
                                    A[h + idx2] = beta * A[h + idx2] + alpha * (p3 * T_local[1] + p4 * T_local[2]); // cba -> cab
                                    A[h + idx4] = beta * A[h + idx4] + alpha * (p3 * T_local[2] + p4 * T_local[1]); // bac -> bca
                                }

                                else if (a == b && b > c)
                                {

                                    double T_local[3];

                                    int64_t idx1 = a * nvv + a * nvir + c; // aac
                                    int64_t idx2 = c * nvv + a * nvir + a; // caa
                                    int64_t idx3 = a * nvv + c * nvir + a; // aca

                                    T_local[0] = p0 * A[h + idx1] + p1 * A[h + idx2] + p2 * A[h + idx1]; // abc -> cba -> bac
                                    T_local[1] = p0 * A[h + idx2] + p1 * A[h + idx1] + p2 * A[h + idx3]; // cba -> abc -> bca
                                    T_local[2] = p0 * A[h + idx3] + p1 * A[h + idx3] + p2 * A[h + idx2]; // acb -> bca -> cab

                                    A[h + idx1] = beta * A[h + idx1] + alpha * (p3 * T_local[0] + p4 * T_local[2]); // abc -> acb
                                    A[h + idx2] = beta * A[h + idx2] + alpha * (p3 * T_local[1] + p4 * T_local[1]); // cba -> cab
                                    A[h + idx3] = beta * A[h + idx3] + alpha * (p3 * T_local[2] + p4 * T_local[0]); // acb -> abc
                                }
                                else if (a == b && b == c)
                                {
                                    double T_local[1];

                                    int64_t idx1 = a * nvv + a * nvir + a; // aaa

                                    T_local[0] = p0 * A[h + idx1] + p1 * A[h + idx1] + p2 * A[h + idx1]; // abc -> cba -> acb

                                    A[h + idx1] = beta * A[h + idx1] + alpha * (p3 * T_local[0] + p4 * T_local[0]); // abc -> bac
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// A unified function for the symmetry operations
void t3_symm_c(const double *A, double *B, int64_t nocc3, int64_t nvir, char *pattern, double alpha, double beta)
{
    int64_t ijk;
    // test bl
    const int64_t bl = 16;
    int64_t nvv = nvir * nvir;
    double p0 = 0.0, p1 = 0.0, p2 = 0.0, p3 = 0.0, p4 = 0.0;

    if (strcmp(pattern, "111111") == 0)
    {
        // abc + acb + bac + bca + cab + cba
        // (1 + P_b^c) (1 + P_a^b + P_a^c) abc
        p0 = 1.0;
        p1 = 1.0;
        p2 = 1.0;
        p3 = 1.0;
        p4 = 1.0;
    }
    else if (strcmp(pattern, "4-2-211-2") == 0)
    {
        // 4 abc - 2 acb - 2 bac + bca + cab - 2 cba
        // (2 - P_b^c) (2 - P_a^b - P_a^c) abc
        p0 = 2.0;
        p1 = -1.0;
        p2 = -1.0;
        p3 = 2.0;
        p4 = -1.0;
    }
    else if (strcmp(pattern, "20-100-1") == 0)
    {
        // 2 * abc + 0 * acb - bac + 0 * bca + 0 * cab - cba
        // (1 - 0 P_b^c) (2 - P_a^b - P_a^c) abc
        p0 = 2.0;
        p1 = -1.0;
        p2 = -1.0;
        p3 = 1.0;
        p4 = 0.0;
    }
    else
    {
        fprintf(stderr, "Error: unrecognized pattern \"%s\"\n", pattern);
        return;
    }

#pragma omp parallel for schedule(static)
    for (ijk = 0; ijk < nocc3; ijk++)
    {
        int64_t h = ijk * nvir * nvir * nvir;
        for (int64_t a0 = 0; a0 < nvir; a0 += bl)
        {
            for (int64_t b0 = 0; b0 <= a0; b0 += bl)
            {
                for (int64_t c0 = 0; c0 <= b0; c0 += bl)
                {
                    for (int64_t a = a0; a < a0 + bl && a < nvir; a++)
                    {
                        for (int64_t b = b0; b < b0 + bl && b <= a; b++)
                        {
                            for (int64_t c = c0; c < c0 + bl && c <= b; c++)
                            {

                                if (a > b && b > c)
                                {
                                    double T_local[6];

                                    int64_t idx1 = a * nvv + b * nvir + c; // abc
                                    int64_t idx2 = c * nvv + b * nvir + a; // cba
                                    int64_t idx3 = a * nvv + c * nvir + b; // acb
                                    int64_t idx4 = b * nvv + a * nvir + c; // bac
                                    int64_t idx5 = b * nvv + c * nvir + a; // bca
                                    int64_t idx6 = c * nvv + a * nvir + b; // cab

                                    T_local[0] = p0 * A[h + idx1] + p1 * A[h + idx2] + p2 * A[h + idx4]; // abc -> cba -> bac
                                    T_local[1] = p0 * A[h + idx2] + p1 * A[h + idx1] + p2 * A[h + idx5]; // cba -> abc -> bca
                                    T_local[2] = p0 * A[h + idx3] + p1 * A[h + idx5] + p2 * A[h + idx6]; // acb -> bca -> cab
                                    T_local[3] = p0 * A[h + idx4] + p1 * A[h + idx6] + p2 * A[h + idx1]; // bac -> cab -> abc
                                    T_local[4] = p0 * A[h + idx5] + p1 * A[h + idx3] + p2 * A[h + idx2]; // bca -> acb -> cba
                                    T_local[5] = p0 * A[h + idx6] + p1 * A[h + idx4] + p2 * A[h + idx3]; // cab -> bac -> acb

                                    B[h + idx1] = beta * A[h + idx1] + alpha * (p3 * T_local[0] + p4 * T_local[2]); // abc -> acb
                                    B[h + idx2] = beta * A[h + idx2] + alpha * (p3 * T_local[1] + p4 * T_local[5]); // cba -> cab
                                    B[h + idx3] = beta * A[h + idx3] + alpha * (p3 * T_local[2] + p4 * T_local[0]); // acb -> abc
                                    B[h + idx4] = beta * A[h + idx4] + alpha * (p3 * T_local[3] + p4 * T_local[4]); // bac -> bca
                                    B[h + idx5] = beta * A[h + idx5] + alpha * (p3 * T_local[4] + p4 * T_local[3]); // bca -> bac
                                    B[h + idx6] = beta * A[h + idx6] + alpha * (p3 * T_local[5] + p4 * T_local[1]); // cab -> cba
                                }

                                else if (a > b && b == c)
                                {
                                    double T_local[3];

                                    int64_t idx1 = a * nvv + b * nvir + b; // abb
                                    int64_t idx2 = b * nvv + b * nvir + a; // bba
                                    int64_t idx4 = b * nvv + a * nvir + b; // bab

                                    T_local[0] = p0 * A[h + idx1] + p1 * A[h + idx2] + p2 * A[h + idx4]; // abc -> cba -> bac
                                    T_local[1] = p0 * A[h + idx2] + p1 * A[h + idx1] + p2 * A[h + idx2]; // cba -> abc -> bca
                                    T_local[2] = p0 * A[h + idx4] + p1 * A[h + idx4] + p2 * A[h + idx1]; // bac -> cab -> abc

                                    B[h + idx1] = beta * A[h + idx1] + alpha * (p3 * T_local[0] + p4 * T_local[0]); // abc -> acb
                                    B[h + idx2] = beta * A[h + idx2] + alpha * (p3 * T_local[1] + p4 * T_local[2]); // cba -> cab
                                    B[h + idx4] = beta * A[h + idx4] + alpha * (p3 * T_local[2] + p4 * T_local[1]); // bac -> bca
                                }

                                else if (a == b && b > c)
                                {

                                    double T_local[3];

                                    int64_t idx1 = a * nvv + a * nvir + c; // aac
                                    int64_t idx2 = c * nvv + a * nvir + a; // caa
                                    int64_t idx3 = a * nvv + c * nvir + a; // aca

                                    T_local[0] = p0 * A[h + idx1] + p1 * A[h + idx2] + p2 * A[h + idx1]; // abc -> cba -> bac
                                    T_local[1] = p0 * A[h + idx2] + p1 * A[h + idx1] + p2 * A[h + idx3]; // cba -> abc -> bca
                                    T_local[2] = p0 * A[h + idx3] + p1 * A[h + idx3] + p2 * A[h + idx2]; // acb -> bca -> cab

                                    B[h + idx1] = beta * A[h + idx1] + alpha * (p3 * T_local[0] + p4 * T_local[2]); // abc -> acb
                                    B[h + idx2] = beta * A[h + idx2] + alpha * (p3 * T_local[1] + p4 * T_local[1]); // cba -> cab
                                    B[h + idx3] = beta * A[h + idx3] + alpha * (p3 * T_local[2] + p4 * T_local[0]); // acb -> abc
                                }
                                else if (a == b && b == c)
                                {
                                    double T_local[1];

                                    int64_t idx1 = a * nvv + a * nvir + a; // aaa

                                    T_local[0] = p0 * A[h + idx1] + p1 * A[h + idx1] + p2 * A[h + idx1]; // abc -> cba -> acb

                                    B[h + idx1] = beta * A[h + idx1] + alpha * (p3 * T_local[0] + p4 * T_local[0]); // abc -> bac
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void t3_p_sum_ip_c(double *A, int64_t nocc, int64_t nvir, double alpha, double beta)
{
    // test bl
    const int64_t bl = 16;
    int64_t nvv = nvir * nvir;
    int64_t nvvv = nvir * nvv;
    int64_t novvv = nocc * nvvv;
    int64_t noovvv = nocc * novvv;

    int64_t ntriplets = 0;
    for (int i = 0; i < nocc; i++)
        for (int j = 0; j <= i; j++)
            for (int k = 0; k <= j; k++)
                ntriplets++;

#pragma omp parallel for schedule(static)
    for (int64_t idx = 0; idx < ntriplets; idx++)
    {
        int i, j, k = 0;
        int64_t tmp = idx;

        for (i = 0; i < nocc; i++)
        {
            int64_t count_ij = (i + 1) * (i + 2) / 2;
            if (tmp < count_ij)
                break;
            tmp -= count_ij;
        }

        for (j = 0; j <= i; j++)
        {
            if (tmp < (j + 1))
            {
                k = tmp;
                break;
            }
            tmp -= (j + 1);
        }

        double T_local[36];
        for (int64_t a0 = 0; a0 < nvir; a0 += bl)
        {
            for (int64_t b0 = 0; b0 <= a0; b0 += bl)
            {
                for (int64_t c0 = 0; c0 <= b0; c0 += bl)
                {
                    for (int64_t a = a0; a < a0 + bl && a < nvir; a++)
                    {
                        for (int64_t b = b0; b < b0 + bl && b <= a; b++)
                        {
                            for (int64_t c = c0; c < c0 + bl && c <= b; c++)
                            {

                                // ijk
                                int64_t idx11 = i * noovvv + j * novvv + k * nvvv + a * nvv + b * nvir + c; // ijkabc
                                int64_t idx12 = i * noovvv + j * novvv + k * nvvv + c * nvv + b * nvir + a; // ijkcba
                                int64_t idx13 = i * noovvv + j * novvv + k * nvvv + a * nvv + c * nvir + b; // ijkacb
                                int64_t idx14 = i * noovvv + j * novvv + k * nvvv + b * nvv + a * nvir + c; // ijkbac
                                int64_t idx15 = i * noovvv + j * novvv + k * nvvv + b * nvv + c * nvir + a; // ijkbca
                                int64_t idx16 = i * noovvv + j * novvv + k * nvvv + c * nvv + a * nvir + b; // ijkcab
                                // kji
                                int64_t idx21 = k * noovvv + j * novvv + i * nvvv + a * nvv + b * nvir + c; // kjiabc
                                int64_t idx22 = k * noovvv + j * novvv + i * nvvv + c * nvv + b * nvir + a; // kjicba
                                int64_t idx23 = k * noovvv + j * novvv + i * nvvv + a * nvv + c * nvir + b; // kjiacb
                                int64_t idx24 = k * noovvv + j * novvv + i * nvvv + b * nvv + a * nvir + c; // kjibac
                                int64_t idx25 = k * noovvv + j * novvv + i * nvvv + b * nvv + c * nvir + a; // kjibca
                                int64_t idx26 = k * noovvv + j * novvv + i * nvvv + c * nvv + a * nvir + b; // kjicab
                                // ijk
                                int64_t idx31 = i * noovvv + k * novvv + j * nvvv + a * nvv + b * nvir + c; // ikjabc
                                int64_t idx32 = i * noovvv + k * novvv + j * nvvv + c * nvv + b * nvir + a; // ikjcba
                                int64_t idx33 = i * noovvv + k * novvv + j * nvvv + a * nvv + c * nvir + b; // ikjacb
                                int64_t idx34 = i * noovvv + k * novvv + j * nvvv + b * nvv + a * nvir + c; // ikjbac
                                int64_t idx35 = i * noovvv + k * novvv + j * nvvv + b * nvv + c * nvir + a; // ikjbca
                                int64_t idx36 = i * noovvv + k * novvv + j * nvvv + c * nvv + a * nvir + b; // ikjcab
                                // jik
                                int64_t idx41 = j * noovvv + i * novvv + k * nvvv + a * nvv + b * nvir + c; // jikabc
                                int64_t idx42 = j * noovvv + i * novvv + k * nvvv + c * nvv + b * nvir + a; // jikcba
                                int64_t idx43 = j * noovvv + i * novvv + k * nvvv + a * nvv + c * nvir + b; // jikacb
                                int64_t idx44 = j * noovvv + i * novvv + k * nvvv + b * nvv + a * nvir + c; // jikbac
                                int64_t idx45 = j * noovvv + i * novvv + k * nvvv + b * nvv + c * nvir + a; // jikbca
                                int64_t idx46 = j * noovvv + i * novvv + k * nvvv + c * nvv + a * nvir + b; // jikcab
                                // jki
                                int64_t idx51 = j * noovvv + k * novvv + i * nvvv + a * nvv + b * nvir + c; // jkiabc
                                int64_t idx52 = j * noovvv + k * novvv + i * nvvv + c * nvv + b * nvir + a; // jkicba
                                int64_t idx53 = j * noovvv + k * novvv + i * nvvv + a * nvv + c * nvir + b; // jkiacb
                                int64_t idx54 = j * noovvv + k * novvv + i * nvvv + b * nvv + a * nvir + c; // jkibac
                                int64_t idx55 = j * noovvv + k * novvv + i * nvvv + b * nvv + c * nvir + a; // jkibca
                                int64_t idx56 = j * noovvv + k * novvv + i * nvvv + c * nvv + a * nvir + b; // jkicab
                                // kij
                                int64_t idx61 = k * noovvv + i * novvv + j * nvvv + a * nvv + b * nvir + c; // kijabc
                                int64_t idx62 = k * noovvv + i * novvv + j * nvvv + c * nvv + b * nvir + a; // kijcba
                                int64_t idx63 = k * noovvv + i * novvv + j * nvvv + a * nvv + c * nvir + b; // kijacb
                                int64_t idx64 = k * noovvv + i * novvv + j * nvvv + b * nvv + a * nvir + c; // kijbac
                                int64_t idx65 = k * noovvv + i * novvv + j * nvvv + b * nvv + c * nvir + a; // kijbca
                                int64_t idx66 = k * noovvv + i * novvv + j * nvvv + c * nvv + a * nvir + b; // kijcab

                                // ijk___ = ijk___ + kij___ + ikj___
                                T_local[0 * 6 + 0] = A[idx11] + A[idx22] + A[idx33]; // ijkabc -> kjicba -> ikjacb
                                T_local[0 * 6 + 1] = A[idx12] + A[idx21] + A[idx36]; // ijkcba -> kjiabc -> ikjcab
                                T_local[0 * 6 + 2] = A[idx13] + A[idx25] + A[idx31]; // ijkacb -> kjibca -> ikjabc
                                T_local[0 * 6 + 3] = A[idx14] + A[idx26] + A[idx35]; // ijkbac -> kjicab -> ikjbca
                                T_local[0 * 6 + 4] = A[idx15] + A[idx23] + A[idx34]; // ijkbca -> kjiacb -> ikjbac
                                T_local[0 * 6 + 5] = A[idx16] + A[idx24] + A[idx32]; // ijkcab -> kjibac -> ikjcba
                                // kji___ = kji___ + ijk___ + kij___
                                T_local[1 * 6 + 0] = A[idx21] + A[idx12] + A[idx63]; // kjiabc -> ijkcba -> kijacb
                                T_local[1 * 6 + 1] = A[idx22] + A[idx11] + A[idx66]; // kjicba -> ijkabc -> kijcab
                                T_local[1 * 6 + 2] = A[idx23] + A[idx15] + A[idx61]; // kjiacb -> ijkbca -> kijabc
                                T_local[1 * 6 + 3] = A[idx24] + A[idx16] + A[idx65]; // kjibac -> ijkcab -> kijbca
                                T_local[1 * 6 + 4] = A[idx25] + A[idx13] + A[idx64]; // kjibca -> ijkacb -> kijbac
                                T_local[1 * 6 + 5] = A[idx26] + A[idx14] + A[idx62]; // kjicab -> ijkbac -> kijcba
                                // ikj___ = ikj__ + jki___ + ijk___
                                T_local[2 * 6 + 0] = A[idx31] + A[idx52] + A[idx13]; // ikjabc -> jkicba -> ijkacb
                                T_local[2 * 6 + 1] = A[idx32] + A[idx51] + A[idx16]; // ikjcba -> jkiabc -> ijkcab
                                T_local[2 * 6 + 2] = A[idx33] + A[idx55] + A[idx11]; // ikjacb -> jkibca -> ijkabc
                                T_local[2 * 6 + 3] = A[idx34] + A[idx56] + A[idx15]; // ikjbac -> jkicab -> ijkbca
                                T_local[2 * 6 + 4] = A[idx35] + A[idx53] + A[idx14]; // ikjbca -> jkiacb -> ijkbac
                                T_local[2 * 6 + 5] = A[idx36] + A[idx54] + A[idx12]; // ikjcab -> jkibac -> ijkcba
                                // jik___ = jik___ + kij___ + jki___
                                T_local[3 * 6 + 0] = A[idx41] + A[idx62] + A[idx53];
                                T_local[3 * 6 + 1] = A[idx42] + A[idx61] + A[idx56];
                                T_local[3 * 6 + 2] = A[idx43] + A[idx65] + A[idx51];
                                T_local[3 * 6 + 3] = A[idx44] + A[idx66] + A[idx55];
                                T_local[3 * 6 + 4] = A[idx45] + A[idx63] + A[idx54];
                                T_local[3 * 6 + 5] = A[idx46] + A[idx64] + A[idx52];
                                // jki___ = jki___ + ikj___ + jik___
                                T_local[4 * 6 + 0] = A[idx51] + A[idx32] + A[idx43];
                                T_local[4 * 6 + 1] = A[idx52] + A[idx31] + A[idx46];
                                T_local[4 * 6 + 2] = A[idx53] + A[idx35] + A[idx41];
                                T_local[4 * 6 + 3] = A[idx54] + A[idx36] + A[idx45];
                                T_local[4 * 6 + 4] = A[idx55] + A[idx33] + A[idx44];
                                T_local[4 * 6 + 5] = A[idx56] + A[idx34] + A[idx42];
                                // kij___ = kij___ + jik___ + kji___
                                T_local[5 * 6 + 0] = A[idx61] + A[idx42] + A[idx23];
                                T_local[5 * 6 + 1] = A[idx62] + A[idx41] + A[idx26];
                                T_local[5 * 6 + 2] = A[idx63] + A[idx45] + A[idx21];
                                T_local[5 * 6 + 3] = A[idx64] + A[idx46] + A[idx25];
                                T_local[5 * 6 + 4] = A[idx65] + A[idx43] + A[idx24];
                                T_local[5 * 6 + 5] = A[idx66] + A[idx44] + A[idx22];

                                // ijk___ = ijk___ + jik___
                                A[idx11] = beta * A[idx11] + alpha * (T_local[0 * 6 + 0] + T_local[3 * 6 + 3]); // abc -> bac
                                A[idx12] = beta * A[idx12] + alpha * (T_local[0 * 6 + 1] + T_local[3 * 6 + 4]); // cba -> bca
                                A[idx13] = beta * A[idx13] + alpha * (T_local[0 * 6 + 2] + T_local[3 * 6 + 5]); // acb -> cab
                                A[idx14] = beta * A[idx14] + alpha * (T_local[0 * 6 + 3] + T_local[3 * 6 + 0]); // bac -> abc
                                A[idx15] = beta * A[idx15] + alpha * (T_local[0 * 6 + 4] + T_local[3 * 6 + 1]); // bca -> cba
                                A[idx16] = beta * A[idx16] + alpha * (T_local[0 * 6 + 5] + T_local[3 * 6 + 2]); // cab -> acb
                                // kji___ = kji___ + jki___
                                A[idx21] = beta * A[idx21] + alpha * (T_local[1 * 6 + 0] + T_local[4 * 6 + 3]); // abc -> bac
                                A[idx22] = beta * A[idx22] + alpha * (T_local[1 * 6 + 1] + T_local[4 * 6 + 4]); // cba -> bca
                                A[idx23] = beta * A[idx23] + alpha * (T_local[1 * 6 + 2] + T_local[4 * 6 + 5]); // acb -> cab
                                A[idx24] = beta * A[idx24] + alpha * (T_local[1 * 6 + 3] + T_local[4 * 6 + 0]); // bac -> abc
                                A[idx25] = beta * A[idx25] + alpha * (T_local[1 * 6 + 4] + T_local[4 * 6 + 1]); // bca -> cba
                                A[idx26] = beta * A[idx26] + alpha * (T_local[1 * 6 + 5] + T_local[4 * 6 + 2]); // cab -> acb
                                // ikj___ = ikj___ + kij___
                                A[idx31] = beta * A[idx31] + alpha * (T_local[2 * 6 + 0] + T_local[5 * 6 + 3]); // abc -> bac
                                A[idx32] = beta * A[idx32] + alpha * (T_local[2 * 6 + 1] + T_local[5 * 6 + 4]); // cba -> bca
                                A[idx33] = beta * A[idx33] + alpha * (T_local[2 * 6 + 2] + T_local[5 * 6 + 5]); // acb -> cab
                                A[idx34] = beta * A[idx34] + alpha * (T_local[2 * 6 + 3] + T_local[5 * 6 + 0]); // bac -> abc
                                A[idx35] = beta * A[idx35] + alpha * (T_local[2 * 6 + 4] + T_local[5 * 6 + 1]); // bca -> cba
                                A[idx36] = beta * A[idx36] + alpha * (T_local[2 * 6 + 5] + T_local[5 * 6 + 2]); // cab -> acb
                                // jik___ = jik___ + ijk___
                                A[idx41] = beta * A[idx41] + alpha * (T_local[3 * 6 + 0] + T_local[0 * 6 + 3]); // abc -> bac
                                A[idx42] = beta * A[idx42] + alpha * (T_local[3 * 6 + 1] + T_local[0 * 6 + 4]); // cba -> bca
                                A[idx43] = beta * A[idx43] + alpha * (T_local[3 * 6 + 2] + T_local[0 * 6 + 5]); // acb -> cab
                                A[idx44] = beta * A[idx44] + alpha * (T_local[3 * 6 + 3] + T_local[0 * 6 + 0]); // bac -> abc
                                A[idx45] = beta * A[idx45] + alpha * (T_local[3 * 6 + 4] + T_local[0 * 6 + 1]); // bca -> cba
                                A[idx46] = beta * A[idx46] + alpha * (T_local[3 * 6 + 5] + T_local[0 * 6 + 2]); // cab -> acb
                                // jki___ = jki___ + kji___
                                A[idx51] = beta * A[idx51] + alpha * (T_local[4 * 6 + 0] + T_local[1 * 6 + 3]); // abc -> bac
                                A[idx52] = beta * A[idx52] + alpha * (T_local[4 * 6 + 1] + T_local[1 * 6 + 4]); // cba -> bca
                                A[idx53] = beta * A[idx53] + alpha * (T_local[4 * 6 + 2] + T_local[1 * 6 + 5]); // acb -> cab
                                A[idx54] = beta * A[idx54] + alpha * (T_local[4 * 6 + 3] + T_local[1 * 6 + 0]); // bac -> abc
                                A[idx55] = beta * A[idx55] + alpha * (T_local[4 * 6 + 4] + T_local[1 * 6 + 1]); // bca -> cba
                                A[idx56] = beta * A[idx56] + alpha * (T_local[4 * 6 + 5] + T_local[1 * 6 + 2]); // cab -> acb
                                // kij___ = kij___ + ikj___
                                A[idx61] = beta * A[idx61] + alpha * (T_local[5 * 6 + 0] + T_local[2 * 6 + 3]); // abc -> bac
                                A[idx62] = beta * A[idx62] + alpha * (T_local[5 * 6 + 1] + T_local[2 * 6 + 4]); // cba -> bca
                                A[idx63] = beta * A[idx63] + alpha * (T_local[5 * 6 + 2] + T_local[2 * 6 + 5]); // acb -> cab
                                A[idx64] = beta * A[idx64] + alpha * (T_local[5 * 6 + 3] + T_local[2 * 6 + 0]); // bac -> abc
                                A[idx65] = beta * A[idx65] + alpha * (T_local[5 * 6 + 4] + T_local[2 * 6 + 1]); // bca -> cba
                                A[idx66] = beta * A[idx66] + alpha * (T_local[5 * 6 + 5] + T_local[2 * 6 + 2]); // cab -> acb
                            }
                        }
                    }
                }
            }
        }
    }
}

const int64_t tp_t3[6][3] = {
    {0, 1, 2}, // no permutation
    {0, 2, 1}, // swap b <-> c
    {1, 0, 2}, // swap a <-> b
    {1, 2, 0}, // a->b, b->c, c->a
    {2, 0, 1}, // a->c, b->a, c->b
    {2, 1, 0}, // reverse
};

void unpack_6fold_c(const double *restrict t3_tril,
                    double *restrict t3_blk,
                    const int64_t *restrict map,
                    const bool *restrict mask,
                    int64_t i0, int64_t i1,
                    int64_t j0, int64_t j1,
                    int64_t k0, int64_t k1,
                    int64_t nocc, int64_t nvir,
                    int64_t blk_i, int64_t blk_j, int64_t blk_k)
{
#define MAP(sym, x, y, z) map[(((sym) * nocc + (x)) * nocc + (y)) * nocc + (z)]
#define MASK(sym, x, y, z) mask[(((sym) * nocc + (x)) * nocc + (y)) * nocc + (z)]

#pragma omp parallel for collapse(4) schedule(static)
    for (int64_t sym = 0; sym < 6; ++sym)
    {
        for (int64_t i = i0; i < i1; ++i)
        {
            for (int64_t j = j0; j < j1; ++j)
            {
                for (int64_t k = k0; k < k1; ++k)
                {
                    if (!MASK(sym, i, j, k))
                        continue;

                    const int64_t *perm = tp_t3[sym];

                    int64_t loc_i = i - i0;
                    int64_t loc_j = j - j0;
                    int64_t loc_k = k - k0;

                    int64_t src_base = MAP(sym, i, j, k) * nvir * nvir * nvir;
                    int64_t dest_base = ((loc_i * blk_j + loc_j) * blk_k + loc_k) * nvir * nvir * nvir;

                    for (int64_t a = 0; a < nvir; ++a)
                    {
                        for (int64_t b = 0; b < nvir; ++b)
                        {
                            for (int64_t c = 0; c < nvir; ++c)
                            {
                                int64_t abc[3] = {a, b, c};
                                int64_t aa = abc[perm[0]];
                                int64_t bb = abc[perm[1]];
                                int64_t cc = abc[perm[2]];

                                int64_t src_idx = src_base + (a * nvir + b) * nvir + c;
                                int64_t dest_idx = dest_base + (aa * nvir + bb) * nvir + cc;

                                t3_blk[dest_idx] = t3_tril[src_idx];
                            }
                        }
                    }
                }
            }
        }
    }
#undef MAP
#undef MASK
}

void unpack_6fold_pair_c(const double *restrict t3_tril,
                         double *restrict t3_blk,
                         const int64_t *restrict map,
                         const bool *restrict mask,
                         int64_t i0, int64_t i1,
                         int64_t j0, int64_t j1,
                         int64_t k0, int64_t k1,
                         int64_t nocc, int64_t nvir,
                         int64_t blk_i, int64_t blk_j, int64_t blk_k)
{

#define MAP(sym, x, y, z) map[(((sym) * nocc + (x)) * nocc + (y)) * nocc + (z)]
#define MASK(sym, x, y, z) mask[(((sym) * nocc + (x)) * nocc + (y)) * nocc + (z)]

    const int64_t tmp_indices[6] = {2, 4, 0, 5, 1, 3};

#pragma omp parallel for collapse(4) schedule(static)
    for (int64_t sym = 0; sym < 6; ++sym)
    {
        for (int64_t i = i0; i < i1; ++i)
        {
            for (int64_t j = j0; j < j1; ++j)
            {
                for (int64_t k = k0; k < k1; ++k)
                {
                    if (!MASK(sym, i, j, k))
                        continue;

                    const int64_t *perm = tp_t3[sym];

                    int64_t loc_i = i - i0;
                    int64_t loc_j = j - j0;
                    int64_t loc_k = k - k0;

                    int64_t src_base = MAP(sym, i, j, k) * nvir * nvir * nvir;
                    int64_t dest_base = ((loc_i * blk_j + loc_j) * blk_k + loc_k) * nvir * nvir * nvir;

                    for (int64_t a = 0; a < nvir; ++a)
                    {
                        for (int64_t b = 0; b < nvir; ++b)
                        {
                            for (int64_t c = 0; c < nvir; ++c)
                            {
                                int64_t abc[3] = {a, b, c};
                                int64_t aa = abc[perm[0]];
                                int64_t bb = abc[perm[1]];
                                int64_t cc = abc[perm[2]];

                                int64_t src_idx = src_base + (a * nvir + b) * nvir + c;
                                int64_t dest_idx = dest_base + (aa * nvir + bb) * nvir + cc;

                                t3_blk[dest_idx] = t3_tril[src_idx];
                            }
                        }
                    }
                }
            }
        }
    }

#pragma omp parallel for collapse(4) schedule(static)
    for (int64_t sym = 0; sym < 6; ++sym)
    {
        for (int64_t i = i0; i < i1; ++i)
        {
            for (int64_t j = j0; j < j1; ++j)
            {
                for (int64_t k = k0; k < k1; ++k)
                {
                    if (!MASK(tmp_indices[sym], i, j, k))
                        continue;

                    const int64_t *perm2 = tp_t3[tmp_indices[sym]];

                    int64_t loc_i = i - i0;
                    int64_t loc_j = j - j0;
                    int64_t loc_k = k - k0;

                    int64_t src_base = MAP(tmp_indices[sym], i, j, k) * nvir * nvir * nvir;
                    int64_t dest_base = ((loc_i * blk_j + loc_j) * blk_k + loc_k) * nvir * nvir * nvir;

                    for (int64_t a = 0; a < nvir; ++a)
                    {
                        for (int64_t b = 0; b < nvir; ++b)
                        {
                            for (int64_t c = 0; c < nvir; ++c)
                            {
                                int64_t abc[3] = {a, b, c};
                                int64_t aa = abc[perm2[0]];
                                int64_t bb = abc[perm2[1]];
                                int64_t cc = abc[perm2[2]];

                                int64_t src_idx = src_base + (a * nvir + b) * nvir + c;
                                int64_t dest_idx = dest_base + (aa * nvir + bb) * nvir + cc;

                                t3_blk[dest_idx] += t3_tril[src_idx];
                            }
                        }
                    }
                }
            }
        }
    }
#undef MAP
#undef MASK
}

void unpack_6fold_pair_s_c(const double *restrict t3_tril,
                           double *restrict t3_blk,
                           const int64_t *restrict map,
                           const bool *restrict mask,
                           int64_t i0, int64_t j0, int64_t k0,
                           int64_t nocc, int64_t nvir)
{

#define MAP(sym, x, y, z) map[(((sym) * nocc + (x)) * nocc + (y)) * nocc + (z)]
#define MASK(sym, x, y, z) mask[(((sym) * nocc + (x)) * nocc + (y)) * nocc + (z)]

    int64_t sym;
    for (sym = 0; sym < 6; ++sym)
    {
        if (MASK(sym, i0, j0, k0))
            break;
    }

    const int64_t *perm = tp_t3[sym];
    int64_t idx = MAP(sym, i0, j0, k0);

#pragma omp parallel for collapse(3) schedule(static)
    for (int64_t a = 0; a < nvir; ++a)
    {
        for (int64_t b = 0; b < nvir; ++b)
        {
            for (int64_t c = 0; c < nvir; ++c)
            {
                int64_t abc[3] = {a, b, c};
                int64_t aa = abc[perm[0]];
                int64_t bb = abc[perm[1]];
                int64_t cc = abc[perm[2]];

                int64_t src_idx = ((idx * nvir + a) * nvir + b) * nvir + c;
                int64_t dest_idx = (aa * nvir + bb) * nvir + cc;

                t3_blk[dest_idx] = t3_tril[src_idx];
            }
        }
    }

    const int64_t tmp_indices[6] = {2, 4, 0, 5, 1, 3};

    for (sym = 0; sym < 6; ++sym)
    {
        if (MASK(tmp_indices[sym], i0, j0, k0))
            break;
    }

    const int64_t *perm2 = tp_t3[tmp_indices[sym]];
    idx = MAP(tmp_indices[sym], i0, j0, k0);

#pragma omp parallel for collapse(3) schedule(static)
    for (int64_t a = 0; a < nvir; ++a)
    {
        for (int64_t b = 0; b < nvir; ++b)
        {
            for (int64_t c = 0; c < nvir; ++c)
            {
                int64_t abc[3] = {a, b, c};
                int64_t aa = abc[perm2[0]];
                int64_t bb = abc[perm2[1]];
                int64_t cc = abc[perm2[2]];

                int64_t src_idx = ((idx * nvir + a) * nvir + b) * nvir + c;
                int64_t dest_idx = (aa * nvir + bb) * nvir + cc;

                t3_blk[dest_idx] += t3_tril[src_idx];
            }
        }
    }
#undef MAP
#undef MASK
}

void unpack_6fold_pair_2_c(const double *restrict t3_tril,
                           double *restrict t3_blk,
                           const int64_t *restrict map,
                           const bool *restrict mask,
                           int64_t i0, int64_t i1,
                           int64_t j0, int64_t j1,
                           int64_t k0, int64_t k1,
                           int64_t nocc, int64_t nvir,
                           int64_t blk_i, int64_t blk_j, int64_t blk_k)
{

#define MAP(sym, x, y, z) map[(((sym) * nocc + (x)) * nocc + (y)) * nocc + (z)]
#define MASK(sym, x, y, z) mask[(((sym) * nocc + (x)) * nocc + (y)) * nocc + (z)]

    const int64_t tmp_indices[6] = {5, 3, 4, 1, 2, 0};
    const int64_t trans_indices[6] = {1, 0, 3, 2, 5, 4};

#pragma omp parallel for collapse(4) schedule(static)
    for (int64_t sym = 0; sym < 6; ++sym)
    {
        for (int64_t i = i0; i < i1; ++i)
        {
            for (int64_t j = j0; j < j1; ++j)
            {
                for (int64_t k = k0; k < k1; ++k)
                {
                    if (!MASK(sym, i, j, k))
                        continue;

                    const int64_t *perm = tp_t3[sym];

                    int64_t loc_i = i - i0;
                    int64_t loc_j = j - j0;
                    int64_t loc_k = k - k0;

                    int64_t src_base = MAP(sym, i, j, k) * nvir * nvir * nvir;
                    int64_t dest_base = ((loc_i * blk_j + loc_j) * blk_k + loc_k) * nvir * nvir * nvir;

                    for (int64_t a = 0; a < nvir; ++a)
                    {
                        for (int64_t b = 0; b < nvir; ++b)
                        {
                            for (int64_t c = 0; c < nvir; ++c)
                            {
                                int64_t abc[3] = {a, b, c};
                                int64_t aa = abc[perm[0]];
                                int64_t bb = abc[perm[1]];
                                int64_t cc = abc[perm[2]];

                                int64_t src_idx = src_base + (a * nvir + b) * nvir + c;
                                int64_t dest_idx = dest_base + (aa * nvir + bb) * nvir + cc;

                                t3_blk[dest_idx] = t3_tril[src_idx];
                            }
                        }
                    }
                }
            }
        }
    }

#pragma omp parallel for collapse(4) schedule(static)
    for (int64_t sym = 0; sym < 6; ++sym)
    {
        for (int64_t i = i0; i < i1; ++i)
        {
            for (int64_t j = j0; j < j1; ++j)
            {
                for (int64_t k = k0; k < k1; ++k)
                {
                    if (!MASK(tmp_indices[sym], i, j, k))
                        continue;

                    const int64_t *perm2 = tp_t3[trans_indices[sym]];

                    int64_t loc_i = i - i0;
                    int64_t loc_j = j - j0;
                    int64_t loc_k = k - k0;

                    int64_t src_base = MAP(tmp_indices[sym], i, j, k) * nvir * nvir * nvir;
                    int64_t dest_base = ((loc_i * blk_j + loc_j) * blk_k + loc_k) * nvir * nvir * nvir;

                    for (int64_t a = 0; a < nvir; ++a)
                    {
                        for (int64_t b = 0; b < nvir; ++b)
                        {
                            for (int64_t c = 0; c < nvir; ++c)
                            {
                                int64_t abc[3] = {a, b, c};
                                int64_t aa = abc[perm2[0]];
                                int64_t bb = abc[perm2[1]];
                                int64_t cc = abc[perm2[2]];

                                int64_t src_idx = src_base + (a * nvir + b) * nvir + c;
                                int64_t dest_idx = dest_base + (aa * nvir + bb) * nvir + cc;

                                t3_blk[dest_idx] += t3_tril[src_idx];
                            }
                        }
                    }
                }
            }
        }
    }
#undef MAP
#undef MASK
}

void update_packed_6fold_c(double *restrict t3_tril,
                           const double *restrict t3_blk,
                           const int64_t *restrict map,
                           int64_t i0, int64_t i1,
                           int64_t j0, int64_t j1,
                           int64_t k0, int64_t k1,
                           int64_t nocc, int64_t nvir,
                           int64_t blk_i, int64_t blk_j, int64_t blk_k,
                           double alpha, double beta)
{
#define MAP(sym, x, y, z) map[(((sym) * nocc + (x)) * nocc + (y)) * nocc + (z)]

    if (j1 < i0 || k1 < j0)
        return;

#pragma omp parallel for collapse(3)
    for (int64_t k = k0; k < k1; ++k)
    {
        for (int64_t j = j0; j < j1; ++j)
        {
            for (int64_t i = i0; i < i1; ++i)
            {
                if (j > k || i > j)
                    continue;

                int64_t p = MAP(0, i, j, k);
                int64_t tril_base = p * nvir * nvir * nvir;

                int64_t loc_i = i - i0;
                int64_t loc_j = j - j0;
                int64_t loc_k = k - k0;
                int64_t blk_base = ((loc_i * blk_j + loc_j) * blk_k + loc_k) * nvir * nvir * nvir;

                for (int64_t a = 0; a < nvir; ++a)
                {
                    for (int64_t b = 0; b < nvir; ++b)
                    {
                        for (int64_t c = 0; c < nvir; ++c)
                        {
                            int64_t idx = ((a * nvir + b) * nvir + c);
                            t3_tril[tril_base + idx] = beta * t3_tril[tril_base + idx] + alpha * t3_blk[blk_base + idx];
                        }
                    }
                }
            }
        }
    }
#undef MAP
}

void update_packed_6fold_s_c(double *restrict t3_tril,
                             const double *restrict t3_blk,
                             const int64_t *restrict map,
                             int64_t i0, int64_t j0, int64_t k0,
                             int64_t nocc, int64_t nvir,
                             double alpha, double beta)
{
#define MAP(sym, x, y, z) map[(((sym) * nocc + (x)) * nocc + (y)) * nocc + (z)]

    int64_t p = MAP(0, i0, j0, k0);
    int64_t tril_base = p * nvir * nvir * nvir;

#pragma omp parallel for collapse(3)
    for (int64_t a = 0; a < nvir; ++a)
    {
        for (int64_t b = 0; b < nvir; ++b)
        {
            for (int64_t c = 0; c < nvir; ++c)
            {
                int64_t idx = ((a * nvir + b) * nvir + c);
                t3_tril[tril_base + idx] = beta * t3_tril[tril_base + idx] + alpha * t3_blk[idx];
            }
        }
    }
#undef MAP
}
