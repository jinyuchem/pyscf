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
#include <math.h>

// A unified function for the four-fold symmetry operations
void t4_symm_ip_c(double *A, int64_t nocc4, int64_t nvir, char *pattern, double alpha, double beta)
{
    int64_t ijkl;
    const int64_t bl = 16;
    int64_t nvv = nvir * nvir;
    int64_t nvvv = nvir * nvv;
    int64_t nvvvv = nvir * nvvv;

    // Coefficients for the permutation pattern
    double p[9];

    if (strcmp(pattern, "11111111") == 0)
    {
        // (1 + P_c^d) (1 + P_b^c + P_b^d) (1 + P_a^b + P_a^c + P_a^d) abcd
        for (int i = 0; i < 9; i++)
            p[i] = 1.0;
    }
    else if (strcmp(pattern, "cnnn") == 0)
    {
        // (1 + 0 * P_c^d) (1 + 0 * P_b^c + 0 * P_b^d) (2 - P_a^b - P_a^c - P_a^d) abcd
        p[0] = 2.0;
        p[1] = -1.0;
        p[2] = -1.0;
        p[3] = -1.0;
        p[4] = 1.0;
        p[5] = 0.0;
        p[6] = 0.0;
        p[7] = 1.0;
        p[8] = 0.0;
    }
    else if (strcmp(pattern, "ccnn") == 0)
    {
        // (1 + 0 * P_c^d) (2 - P_b^c - P_b^d) (2 - P_a^b - P_a^c - P_a^d) abcd
        p[0] = 2.0;
        p[1] = -1.0;
        p[2] = -1.0;
        p[3] = -1.0;
        p[4] = 2.0;
        p[5] = -1.0;
        p[6] = -1.0;
        p[7] = 1.0;
        p[8] = 0.0;
    }
    else
    {
        fprintf(stderr, "Error: unrecognized pattern \"%s\"\n", pattern);
        return;
    }

#pragma omp parallel for schedule(static)
    for (ijkl = 0; ijkl < nocc4; ijkl++)
    {
        int64_t h = ijkl * nvvvv;
        for (int64_t a0 = 0; a0 < nvir; a0 += bl)
        {
            for (int64_t b0 = 0; b0 <= a0; b0 += bl)
            {
                for (int64_t c0 = 0; c0 <= b0; c0 += bl)
                {
                    for (int64_t d0 = 0; d0 <= c0; d0 += bl)
                    {
                        for (int64_t a = a0; a < a0 + bl && a < nvir; a++)
                        {
                            for (int64_t b = b0; b < b0 + bl && b <= a; b++)
                            {
                                for (int64_t c = c0; c < c0 + bl && c <= b; c++)
                                {
                                    for (int64_t d = d0; d < d0 + bl && d <= c; d++)
                                    {
                                        if (a > b && b > c && c > d)
                                        {
                                            double T1_local[24];
                                            double T2_local[24];

                                            int64_t indices[24];
                                            indices[0] = a * nvvv + b * nvv + c * nvir + d; // abcd
                                            indices[1] = a * nvvv + b * nvv + d * nvir + c; // abdc
                                            indices[2] = a * nvvv + c * nvv + b * nvir + d; // acbd
                                            indices[3] = a * nvvv + c * nvv + d * nvir + b; // acdb
                                            indices[4] = a * nvvv + d * nvv + b * nvir + c; // adbc
                                            indices[5] = a * nvvv + d * nvv + c * nvir + b; // adcb

                                            indices[6] = b * nvvv + a * nvv + c * nvir + d;  // bacd
                                            indices[7] = b * nvvv + a * nvv + d * nvir + c;  // badc
                                            indices[8] = b * nvvv + c * nvv + a * nvir + d;  // bcad
                                            indices[9] = b * nvvv + c * nvv + d * nvir + a;  // bcda
                                            indices[10] = b * nvvv + d * nvv + a * nvir + c; // bdac
                                            indices[11] = b * nvvv + d * nvv + c * nvir + a; // bdca

                                            indices[12] = c * nvvv + a * nvv + b * nvir + d; // cabd
                                            indices[13] = c * nvvv + a * nvv + d * nvir + b; // cadb
                                            indices[14] = c * nvvv + b * nvv + a * nvir + d; // cbad
                                            indices[15] = c * nvvv + b * nvv + d * nvir + a; // cbda
                                            indices[16] = c * nvvv + d * nvv + a * nvir + b; // cdab
                                            indices[17] = c * nvvv + d * nvv + b * nvir + a; // cdba

                                            indices[18] = d * nvvv + a * nvv + b * nvir + c; // dabc
                                            indices[19] = d * nvvv + a * nvv + c * nvir + b; // dacb
                                            indices[20] = d * nvvv + b * nvv + a * nvir + c; // dbac
                                            indices[21] = d * nvvv + b * nvv + c * nvir + a; // dbca
                                            indices[22] = d * nvvv + c * nvv + a * nvir + b; // dcab
                                            indices[23] = d * nvvv + c * nvv + b * nvir + a; // dcba

                                            // (1 + P_a^b + P_a^c + P_a^d)
                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[6]] + p[2] * A[h + indices[14]] + p[3] * A[h + indices[21]];   // abcd + bacd + cbad + dbca
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[7]] + p[2] * A[h + indices[20]] + p[3] * A[h + indices[15]];   // abdc + badc + dbac + cbda
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[12]] + p[2] * A[h + indices[8]] + p[3] * A[h + indices[23]];   // acbd + cabd + bcad + dcba
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[13]] + p[2] * A[h + indices[22]] + p[3] * A[h + indices[9]];   // acdb + cadb + dcab + bcda
                                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[18]] + p[2] * A[h + indices[10]] + p[3] * A[h + indices[17]];  // adbc + dabc + bdac + cdba
                                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[19]] + p[2] * A[h + indices[16]] + p[3] * A[h + indices[11]];  // adcb + dacb + cdab + bdca
                                            T1_local[6] = p[0] * A[h + indices[6]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[12]] + p[3] * A[h + indices[19]];   // bacd + abcd + cabd + dacb
                                            T1_local[7] = p[0] * A[h + indices[7]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[18]] + p[3] * A[h + indices[13]];   // badc + abdc + dabc + cadb
                                            T1_local[8] = p[0] * A[h + indices[8]] + p[1] * A[h + indices[14]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[22]];   // bcad + cbad + acbd + dcab
                                            T1_local[9] = p[0] * A[h + indices[9]] + p[1] * A[h + indices[15]] + p[2] * A[h + indices[23]] + p[3] * A[h + indices[3]];   // bcda + cbda + dcba + acdb
                                            T1_local[10] = p[0] * A[h + indices[10]] + p[1] * A[h + indices[20]] + p[2] * A[h + indices[4]] + p[3] * A[h + indices[16]]; // bdac + dbac + adbc + cdab
                                            T1_local[11] = p[0] * A[h + indices[11]] + p[1] * A[h + indices[21]] + p[2] * A[h + indices[17]] + p[3] * A[h + indices[5]]; // bdca + dbca + cdba + adcb
                                            T1_local[12] = p[0] * A[h + indices[12]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[6]] + p[3] * A[h + indices[18]];  // cabd + acbd + bacd + dabc
                                            T1_local[13] = p[0] * A[h + indices[13]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[19]] + p[3] * A[h + indices[7]];  // cadb + acdb + dacb + badc
                                            T1_local[14] = p[0] * A[h + indices[14]] + p[1] * A[h + indices[8]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[20]];  // cbad + bcad + abcd + dbac
                                            T1_local[15] = p[0] * A[h + indices[15]] + p[1] * A[h + indices[9]] + p[2] * A[h + indices[21]] + p[3] * A[h + indices[1]];  // cbda + bcda + dbca + abdc
                                            T1_local[16] = p[0] * A[h + indices[16]] + p[1] * A[h + indices[22]] + p[2] * A[h + indices[5]] + p[3] * A[h + indices[10]]; // cdab + dcab + adcb + bdac
                                            T1_local[17] = p[0] * A[h + indices[17]] + p[1] * A[h + indices[23]] + p[2] * A[h + indices[11]] + p[3] * A[h + indices[4]]; // cdba + dcba + bdca + adbc
                                            T1_local[18] = p[0] * A[h + indices[18]] + p[1] * A[h + indices[4]] + p[2] * A[h + indices[7]] + p[3] * A[h + indices[12]];  // dabc + adbc + badc + cabd
                                            T1_local[19] = p[0] * A[h + indices[19]] + p[1] * A[h + indices[5]] + p[2] * A[h + indices[13]] + p[3] * A[h + indices[6]];  // dacb + adcb + cadb + bacd
                                            T1_local[20] = p[0] * A[h + indices[20]] + p[1] * A[h + indices[10]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[14]]; // dbac + bdac + abdc + cbad
                                            T1_local[21] = p[0] * A[h + indices[21]] + p[1] * A[h + indices[11]] + p[2] * A[h + indices[15]] + p[3] * A[h + indices[0]]; // dbca + bdca + cbda + abcd
                                            T1_local[22] = p[0] * A[h + indices[22]] + p[1] * A[h + indices[16]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[8]];  // dcab + cdab + acdb + bcad
                                            T1_local[23] = p[0] * A[h + indices[23]] + p[1] * A[h + indices[17]] + p[2] * A[h + indices[9]] + p[3] * A[h + indices[2]];  // dcba + cdba + bcda + acbd

                                            // (1 + P_b^c + P_b^d)
                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[2] + p[6] * T1_local[5];     // abcd + acbd + adcb
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[4] + p[6] * T1_local[3];     // abdc + adbc + acdb
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[0] + p[6] * T1_local[4];     // acbd + abcd + adbc
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[5] + p[6] * T1_local[1];     // acdb + adcb + abdc
                                            T2_local[4] = p[4] * T1_local[4] + p[5] * T1_local[1] + p[6] * T1_local[2];     // adbc + abdc + acbd
                                            T2_local[5] = p[4] * T1_local[5] + p[5] * T1_local[3] + p[6] * T1_local[0];     // adcb + acdb + abcd
                                            T2_local[6] = p[4] * T1_local[6] + p[5] * T1_local[8] + p[6] * T1_local[11];    // bacd + bcad + bdca
                                            T2_local[7] = p[4] * T1_local[7] + p[5] * T1_local[10] + p[6] * T1_local[9];    // badc + bdac + bcda
                                            T2_local[8] = p[4] * T1_local[8] + p[5] * T1_local[6] + p[6] * T1_local[10];    // bcad + bacd + bdac
                                            T2_local[9] = p[4] * T1_local[9] + p[5] * T1_local[11] + p[6] * T1_local[7];    // bcda + bdca + badc
                                            T2_local[10] = p[4] * T1_local[10] + p[5] * T1_local[7] + p[6] * T1_local[8];   // bdac + badc + bcad
                                            T2_local[11] = p[4] * T1_local[11] + p[5] * T1_local[9] + p[6] * T1_local[6];   // bdca + bcda + bacd
                                            T2_local[12] = p[4] * T1_local[12] + p[5] * T1_local[14] + p[6] * T1_local[17]; // cabd + cbad + cdba
                                            T2_local[13] = p[4] * T1_local[13] + p[5] * T1_local[16] + p[6] * T1_local[15]; // cadb + cdab + cbda
                                            T2_local[14] = p[4] * T1_local[14] + p[5] * T1_local[12] + p[6] * T1_local[16]; // cbad + cabd + cdab
                                            T2_local[15] = p[4] * T1_local[15] + p[5] * T1_local[17] + p[6] * T1_local[13]; // cbda + cdba + cadb
                                            T2_local[16] = p[4] * T1_local[16] + p[5] * T1_local[13] + p[6] * T1_local[14]; // cdab + cadb + cbad
                                            T2_local[17] = p[4] * T1_local[17] + p[5] * T1_local[15] + p[6] * T1_local[12]; // cdba + cbda + cabd
                                            T2_local[18] = p[4] * T1_local[18] + p[5] * T1_local[20] + p[6] * T1_local[23]; // dabc + dbac + dcba
                                            T2_local[19] = p[4] * T1_local[19] + p[5] * T1_local[22] + p[6] * T1_local[21]; // dacb + dcab + dbca
                                            T2_local[20] = p[4] * T1_local[20] + p[5] * T1_local[18] + p[6] * T1_local[22]; // dbac + dabc + dcab
                                            T2_local[21] = p[4] * T1_local[21] + p[5] * T1_local[23] + p[6] * T1_local[19]; // dbca + dcba + dacb
                                            T2_local[22] = p[4] * T1_local[22] + p[5] * T1_local[19] + p[6] * T1_local[20]; // dcab + dacb + dbac
                                            T2_local[23] = p[4] * T1_local[23] + p[5] * T1_local[21] + p[6] * T1_local[18]; // dcba + dbca + dabc

                                            // (1 + P_c^d)
                                            A[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[1]) + beta * A[h + indices[0]];     // abcd + abdc
                                            A[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[0]) + beta * A[h + indices[1]];     // abdc + abcd
                                            A[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[3]) + beta * A[h + indices[2]];     // acbd + acdb
                                            A[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[2]) + beta * A[h + indices[3]];     // acdb + acbd
                                            A[h + indices[4]] = alpha * (p[7] * T2_local[4] + p[8] * T2_local[5]) + beta * A[h + indices[4]];     // adbc + adcb
                                            A[h + indices[5]] = alpha * (p[7] * T2_local[5] + p[8] * T2_local[4]) + beta * A[h + indices[5]];     // adcb + adbc
                                            A[h + indices[6]] = alpha * (p[7] * T2_local[6] + p[8] * T2_local[7]) + beta * A[h + indices[6]];     // bacd + badc
                                            A[h + indices[7]] = alpha * (p[7] * T2_local[7] + p[8] * T2_local[6]) + beta * A[h + indices[7]];     // badc + bacd
                                            A[h + indices[8]] = alpha * (p[7] * T2_local[8] + p[8] * T2_local[9]) + beta * A[h + indices[8]];     // bcad + bcda
                                            A[h + indices[9]] = alpha * (p[7] * T2_local[9] + p[8] * T2_local[8]) + beta * A[h + indices[9]];     // bcda + bcad
                                            A[h + indices[10]] = alpha * (p[7] * T2_local[10] + p[8] * T2_local[11]) + beta * A[h + indices[10]]; // bdac + bdca
                                            A[h + indices[11]] = alpha * (p[7] * T2_local[11] + p[8] * T2_local[10]) + beta * A[h + indices[11]]; // bdca + bdac
                                            A[h + indices[12]] = alpha * (p[7] * T2_local[12] + p[8] * T2_local[13]) + beta * A[h + indices[12]]; // cabd + cadb
                                            A[h + indices[13]] = alpha * (p[7] * T2_local[13] + p[8] * T2_local[12]) + beta * A[h + indices[13]]; // cadb + cabd
                                            A[h + indices[14]] = alpha * (p[7] * T2_local[14] + p[8] * T2_local[15]) + beta * A[h + indices[14]]; // cbad + cbda
                                            A[h + indices[15]] = alpha * (p[7] * T2_local[15] + p[8] * T2_local[14]) + beta * A[h + indices[15]]; // cbda + cbad
                                            A[h + indices[16]] = alpha * (p[7] * T2_local[16] + p[8] * T2_local[17]) + beta * A[h + indices[16]]; // cdab + cdba
                                            A[h + indices[17]] = alpha * (p[7] * T2_local[17] + p[8] * T2_local[16]) + beta * A[h + indices[17]]; // cdba + cdab
                                            A[h + indices[18]] = alpha * (p[7] * T2_local[18] + p[8] * T2_local[19]) + beta * A[h + indices[18]]; // dabc + dacb
                                            A[h + indices[19]] = alpha * (p[7] * T2_local[19] + p[8] * T2_local[18]) + beta * A[h + indices[19]]; // dacb + dabc
                                            A[h + indices[20]] = alpha * (p[7] * T2_local[20] + p[8] * T2_local[21]) + beta * A[h + indices[20]]; // dbac + dbca
                                            A[h + indices[21]] = alpha * (p[7] * T2_local[21] + p[8] * T2_local[20]) + beta * A[h + indices[21]]; // dbca + dbac
                                            A[h + indices[22]] = alpha * (p[7] * T2_local[22] + p[8] * T2_local[23]) + beta * A[h + indices[22]]; // dcab + dcba
                                            A[h + indices[23]] = alpha * (p[7] * T2_local[23] + p[8] * T2_local[22]) + beta * A[h + indices[23]]; // dcba + dcab
                                        }
                                        else if (a > b && b > c && c == d)
                                        {
                                            double T1_local[12];
                                            double T2_local[12];

                                            int64_t indices[12];
                                            indices[0] = a * nvvv + b * nvv + c * nvir + c;  // abcc
                                            indices[1] = a * nvvv + c * nvv + b * nvir + c;  // acbc
                                            indices[2] = a * nvvv + c * nvv + c * nvir + b;  // accb
                                            indices[3] = b * nvvv + a * nvv + c * nvir + c;  // bacc
                                            indices[4] = b * nvvv + c * nvv + a * nvir + c;  // bcac
                                            indices[5] = b * nvvv + c * nvv + c * nvir + a;  // bcca
                                            indices[6] = c * nvvv + a * nvv + b * nvir + c;  // cabc
                                            indices[7] = c * nvvv + a * nvv + c * nvir + b;  // cacb
                                            indices[8] = c * nvvv + b * nvv + a * nvir + c;  // cbac
                                            indices[9] = c * nvvv + b * nvv + c * nvir + a;  // cbca
                                            indices[10] = c * nvvv + c * nvv + a * nvir + b; // ccab
                                            indices[11] = c * nvvv + c * nvv + b * nvir + a; // ccba

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[8]] + p[3] * A[h + indices[9]];    // abcc + bacc + cbac + cbca
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[6]] + p[2] * A[h + indices[4]] + p[3] * A[h + indices[11]];   // acbc + cabc + bcac + ccba
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[7]] + p[2] * A[h + indices[10]] + p[3] * A[h + indices[5]];   // accb + cacb + ccab + bcca
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[6]] + p[3] * A[h + indices[7]];    // bacc + abcc + cabc + cacb
                                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[8]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[10]];   // bcac + cbac + acbc + ccab
                                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[9]] + p[2] * A[h + indices[11]] + p[3] * A[h + indices[2]];   // bcca + cbca + ccba + accb
                                            T1_local[6] = p[0] * A[h + indices[6]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[6]];    // cabc + acbc + bacc + cabc
                                            T1_local[7] = p[0] * A[h + indices[7]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[7]] + p[3] * A[h + indices[3]];    // cacb + accb + cacb + bacc
                                            T1_local[8] = p[0] * A[h + indices[8]] + p[1] * A[h + indices[4]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[8]];    // cbac + bcac + abcc + cbac
                                            T1_local[9] = p[0] * A[h + indices[9]] + p[1] * A[h + indices[5]] + p[2] * A[h + indices[9]] + p[3] * A[h + indices[0]];    // cbca + bcca + cbca + abcc
                                            T1_local[10] = p[0] * A[h + indices[10]] + p[1] * A[h + indices[10]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[4]]; // ccab + ccab + accb + bcac
                                            T1_local[11] = p[0] * A[h + indices[11]] + p[1] * A[h + indices[11]] + p[2] * A[h + indices[5]] + p[3] * A[h + indices[1]]; // ccba + ccba + bcca + acbc

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[1] + p[6] * T1_local[2];   // abcc + acbc + accb
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[0] + p[6] * T1_local[1];   // acbc + abcc + acbc
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[2] + p[6] * T1_local[0];   // accb + accb + abcc
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[4] + p[6] * T1_local[5];   // bacc + bcac + bcca
                                            T2_local[4] = p[4] * T1_local[4] + p[5] * T1_local[3] + p[6] * T1_local[4];   // bcac + bacc + bcac
                                            T2_local[5] = p[4] * T1_local[5] + p[5] * T1_local[5] + p[6] * T1_local[3];   // bcca + bcca + bacc
                                            T2_local[6] = p[4] * T1_local[6] + p[5] * T1_local[8] + p[6] * T1_local[11];  // cabc + cbac + ccba
                                            T2_local[7] = p[4] * T1_local[7] + p[5] * T1_local[10] + p[6] * T1_local[9];  // cacb + ccab + cbca
                                            T2_local[8] = p[4] * T1_local[8] + p[5] * T1_local[6] + p[6] * T1_local[10];  // cbac + cabc + ccab
                                            T2_local[9] = p[4] * T1_local[9] + p[5] * T1_local[11] + p[6] * T1_local[7];  // cbca + ccba + cacb
                                            T2_local[10] = p[4] * T1_local[10] + p[5] * T1_local[7] + p[6] * T1_local[8]; // ccab + cacb + cbac
                                            T2_local[11] = p[4] * T1_local[11] + p[5] * T1_local[9] + p[6] * T1_local[6]; // ccba + cbca + cabc

                                            A[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[0]) + beta * A[h + indices[0]];     // abcc + abcc
                                            A[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[2]) + beta * A[h + indices[1]];     // acbc + accb
                                            A[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[1]) + beta * A[h + indices[2]];     // accb + acbc
                                            A[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[3]) + beta * A[h + indices[3]];     // bacc + bacc
                                            A[h + indices[4]] = alpha * (p[7] * T2_local[4] + p[8] * T2_local[5]) + beta * A[h + indices[4]];     // bcac + bcca
                                            A[h + indices[5]] = alpha * (p[7] * T2_local[5] + p[8] * T2_local[4]) + beta * A[h + indices[5]];     // bcca + bcac
                                            A[h + indices[6]] = alpha * (p[7] * T2_local[6] + p[8] * T2_local[7]) + beta * A[h + indices[6]];     // cabc + cacb
                                            A[h + indices[7]] = alpha * (p[7] * T2_local[7] + p[8] * T2_local[6]) + beta * A[h + indices[7]];     // cacb + cabc
                                            A[h + indices[8]] = alpha * (p[7] * T2_local[8] + p[8] * T2_local[9]) + beta * A[h + indices[8]];     // cbac + cbca
                                            A[h + indices[9]] = alpha * (p[7] * T2_local[9] + p[8] * T2_local[8]) + beta * A[h + indices[9]];     // cbca + cbac
                                            A[h + indices[10]] = alpha * (p[7] * T2_local[10] + p[8] * T2_local[11]) + beta * A[h + indices[10]]; // ccab + ccba
                                            A[h + indices[11]] = alpha * (p[7] * T2_local[11] + p[8] * T2_local[10]) + beta * A[h + indices[11]]; // ccba + ccab
                                        }
                                        else if (a > b && b == c && c > d)
                                        {
                                            double T1_local[12];
                                            double T2_local[12];

                                            int64_t indices[12];
                                            indices[0] = a * nvvv + b * nvv + b * nvir + d;  // abbd
                                            indices[1] = a * nvvv + b * nvv + d * nvir + b;  // abdb
                                            indices[2] = a * nvvv + d * nvv + b * nvir + b;  // adbb
                                            indices[3] = b * nvvv + a * nvv + b * nvir + d;  // babd
                                            indices[4] = b * nvvv + a * nvv + d * nvir + b;  // badb
                                            indices[5] = b * nvvv + b * nvv + a * nvir + d;  // bbad
                                            indices[6] = b * nvvv + b * nvv + d * nvir + a;  // bbda
                                            indices[7] = b * nvvv + d * nvv + a * nvir + b;  // bdab
                                            indices[8] = b * nvvv + d * nvv + b * nvir + a;  // bdba
                                            indices[9] = d * nvvv + a * nvv + b * nvir + b;  // dabb
                                            indices[10] = d * nvvv + b * nvv + a * nvir + b; // dbab
                                            indices[11] = d * nvvv + b * nvv + b * nvir + a; // dbba

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[5]] + p[3] * A[h + indices[11]];  // abbd + babd + bbad + dbba
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[4]] + p[2] * A[h + indices[10]] + p[3] * A[h + indices[6]];  // abdb + badb + dbab + bbda
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[9]] + p[2] * A[h + indices[7]] + p[3] * A[h + indices[8]];   // adbb + dabb + bdab + bdba
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[9]];   // babd + abbd + babd + dabb
                                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[9]] + p[3] * A[h + indices[4]];   // badb + abdb + dabb + badb
                                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[5]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[10]];  // bbad + bbad + abbd + dbab
                                            T1_local[6] = p[0] * A[h + indices[6]] + p[1] * A[h + indices[6]] + p[2] * A[h + indices[11]] + p[3] * A[h + indices[1]];  // bbda + bbda + dbba + abdb
                                            T1_local[7] = p[0] * A[h + indices[7]] + p[1] * A[h + indices[10]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[7]];  // bdab + dbab + adbb + bdab
                                            T1_local[8] = p[0] * A[h + indices[8]] + p[1] * A[h + indices[11]] + p[2] * A[h + indices[8]] + p[3] * A[h + indices[2]];  // bdba + dbba + bdba + adbb
                                            T1_local[9] = p[0] * A[h + indices[9]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[4]] + p[3] * A[h + indices[3]];   // dabb + adbb + badb + babd
                                            T1_local[10] = p[0] * A[h + indices[10]] + p[1] * A[h + indices[7]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[5]]; // dbab + bdab + abdb + bbad
                                            T1_local[11] = p[0] * A[h + indices[11]] + p[1] * A[h + indices[8]] + p[2] * A[h + indices[6]] + p[3] * A[h + indices[0]]; // dbba + bdba + bbda + abbd

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[0] + p[6] * T1_local[2];    // abbd + abbd + adbb
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[2] + p[6] * T1_local[1];    // abdb + adbb + abdb
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[1] + p[6] * T1_local[0];    // adbb + abdb + abbd
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[5] + p[6] * T1_local[8];    // babd + bbad + bdba
                                            T2_local[4] = p[4] * T1_local[4] + p[5] * T1_local[7] + p[6] * T1_local[6];    // badb + bdab + bbda
                                            T2_local[5] = p[4] * T1_local[5] + p[5] * T1_local[3] + p[6] * T1_local[7];    // bbad + babd + bdab
                                            T2_local[6] = p[4] * T1_local[6] + p[5] * T1_local[8] + p[6] * T1_local[4];    // bbda + bdba + badb
                                            T2_local[7] = p[4] * T1_local[7] + p[5] * T1_local[4] + p[6] * T1_local[5];    // bdab + badb + bbad
                                            T2_local[8] = p[4] * T1_local[8] + p[5] * T1_local[6] + p[6] * T1_local[3];    // bdba + bbda + babd
                                            T2_local[9] = p[4] * T1_local[9] + p[5] * T1_local[10] + p[6] * T1_local[11];  // dabb + dbab + dbba
                                            T2_local[10] = p[4] * T1_local[10] + p[5] * T1_local[9] + p[6] * T1_local[10]; // dbab + dabb + dbab
                                            T2_local[11] = p[4] * T1_local[11] + p[5] * T1_local[11] + p[6] * T1_local[9]; // dbba + dbba + dabb

                                            A[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[1]) + beta * A[h + indices[0]];     // abbd + abdb
                                            A[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[0]) + beta * A[h + indices[1]];     // abdb + abbd
                                            A[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[2]) + beta * A[h + indices[2]];     // adbb + adbb
                                            A[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[4]) + beta * A[h + indices[3]];     // babd + badb
                                            A[h + indices[4]] = alpha * (p[7] * T2_local[4] + p[8] * T2_local[3]) + beta * A[h + indices[4]];     // badb + babd
                                            A[h + indices[5]] = alpha * (p[7] * T2_local[5] + p[8] * T2_local[6]) + beta * A[h + indices[5]];     // bbad + bbda
                                            A[h + indices[6]] = alpha * (p[7] * T2_local[6] + p[8] * T2_local[5]) + beta * A[h + indices[6]];     // bbda + bbad
                                            A[h + indices[7]] = alpha * (p[7] * T2_local[7] + p[8] * T2_local[8]) + beta * A[h + indices[7]];     // bdab + bdba
                                            A[h + indices[8]] = alpha * (p[7] * T2_local[8] + p[8] * T2_local[7]) + beta * A[h + indices[8]];     // bdba + bdab
                                            A[h + indices[9]] = alpha * (p[7] * T2_local[9] + p[8] * T2_local[9]) + beta * A[h + indices[9]];     // dabb + dabb
                                            A[h + indices[10]] = alpha * (p[7] * T2_local[10] + p[8] * T2_local[11]) + beta * A[h + indices[10]]; // dbab + dbba
                                            A[h + indices[11]] = alpha * (p[7] * T2_local[11] + p[8] * T2_local[10]) + beta * A[h + indices[11]]; // dbba + dbab
                                        }
                                        else if (a == b && b > c && c > d)
                                        {
                                            double T1_local[12];
                                            double T2_local[12];

                                            int64_t indices[12];
                                            indices[0] = a * nvvv + a * nvv + c * nvir + d;  // aacd
                                            indices[1] = a * nvvv + a * nvv + d * nvir + c;  // aadc
                                            indices[2] = a * nvvv + c * nvv + a * nvir + d;  // acad
                                            indices[3] = a * nvvv + c * nvv + d * nvir + a;  // acda
                                            indices[4] = a * nvvv + d * nvv + a * nvir + c;  // adac
                                            indices[5] = a * nvvv + d * nvv + c * nvir + a;  // adca
                                            indices[6] = c * nvvv + a * nvv + a * nvir + d;  // caad
                                            indices[7] = c * nvvv + a * nvv + d * nvir + a;  // cada
                                            indices[8] = c * nvvv + d * nvv + a * nvir + a;  // cdaa
                                            indices[9] = d * nvvv + a * nvv + a * nvir + c;  // daac
                                            indices[10] = d * nvvv + a * nvv + c * nvir + a; // daca
                                            indices[11] = d * nvvv + c * nvv + a * nvir + a; // dcaa

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[6]] + p[3] * A[h + indices[10]];  // aacd + aacd + caad + daca
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[9]] + p[3] * A[h + indices[7]];   // aadc + aadc + daac + cada
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[6]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[11]];  // acad + caad + acad + dcaa
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[7]] + p[2] * A[h + indices[11]] + p[3] * A[h + indices[3]];  // acda + cada + dcaa + acda
                                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[9]] + p[2] * A[h + indices[4]] + p[3] * A[h + indices[8]];   // adac + daac + adac + cdaa
                                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[10]] + p[2] * A[h + indices[8]] + p[3] * A[h + indices[5]];  // adca + daca + cdaa + adca
                                            T1_local[6] = p[0] * A[h + indices[6]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[9]];   // caad + acad + aacd + daac
                                            T1_local[7] = p[0] * A[h + indices[7]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[10]] + p[3] * A[h + indices[1]];  // cada + acda + daca + aadc
                                            T1_local[8] = p[0] * A[h + indices[8]] + p[1] * A[h + indices[11]] + p[2] * A[h + indices[5]] + p[3] * A[h + indices[4]];  // cdaa + dcaa + adca + adac
                                            T1_local[9] = p[0] * A[h + indices[9]] + p[1] * A[h + indices[4]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[6]];   // daac + adac + aadc + caad
                                            T1_local[10] = p[0] * A[h + indices[10]] + p[1] * A[h + indices[5]] + p[2] * A[h + indices[7]] + p[3] * A[h + indices[0]]; // daca + adca + cada + aacd
                                            T1_local[11] = p[0] * A[h + indices[11]] + p[1] * A[h + indices[8]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[2]]; // dcaa + cdaa + acda + acad

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[2] + p[6] * T1_local[5];     // aacd + acad + adca
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[4] + p[6] * T1_local[3];     // aadc + adac + acda
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[0] + p[6] * T1_local[4];     // acad + aacd + adac
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[5] + p[6] * T1_local[1];     // acda + adca + aadc
                                            T2_local[4] = p[4] * T1_local[4] + p[5] * T1_local[1] + p[6] * T1_local[2];     // adac + aadc + acad
                                            T2_local[5] = p[4] * T1_local[5] + p[5] * T1_local[3] + p[6] * T1_local[0];     // adca + acda + aacd
                                            T2_local[6] = p[4] * T1_local[6] + p[5] * T1_local[6] + p[6] * T1_local[8];     // caad + caad + cdaa
                                            T2_local[7] = p[4] * T1_local[7] + p[5] * T1_local[8] + p[6] * T1_local[7];     // cada + cdaa + cada
                                            T2_local[8] = p[4] * T1_local[8] + p[5] * T1_local[7] + p[6] * T1_local[6];     // cdaa + cada + caad
                                            T2_local[9] = p[4] * T1_local[9] + p[5] * T1_local[9] + p[6] * T1_local[11];    // daac + daac + dcaa
                                            T2_local[10] = p[4] * T1_local[10] + p[5] * T1_local[11] + p[6] * T1_local[10]; // daca + dcaa + daca
                                            T2_local[11] = p[4] * T1_local[11] + p[5] * T1_local[10] + p[6] * T1_local[9];  // dcaa + daca + daac

                                            A[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[1]) + beta * A[h + indices[0]];     // aacd + aadc
                                            A[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[0]) + beta * A[h + indices[1]];     // aadc + aacd
                                            A[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[3]) + beta * A[h + indices[2]];     // acad + acda
                                            A[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[2]) + beta * A[h + indices[3]];     // acda + acad
                                            A[h + indices[4]] = alpha * (p[7] * T2_local[4] + p[8] * T2_local[5]) + beta * A[h + indices[4]];     // adac + adca
                                            A[h + indices[5]] = alpha * (p[7] * T2_local[5] + p[8] * T2_local[4]) + beta * A[h + indices[5]];     // adca + adac
                                            A[h + indices[6]] = alpha * (p[7] * T2_local[6] + p[8] * T2_local[7]) + beta * A[h + indices[6]];     // caad + cada
                                            A[h + indices[7]] = alpha * (p[7] * T2_local[7] + p[8] * T2_local[6]) + beta * A[h + indices[7]];     // cada + caad
                                            A[h + indices[8]] = alpha * (p[7] * T2_local[8] + p[8] * T2_local[8]) + beta * A[h + indices[8]];     // cdaa + cdaa
                                            A[h + indices[9]] = alpha * (p[7] * T2_local[9] + p[8] * T2_local[10]) + beta * A[h + indices[9]];    // daac + daca
                                            A[h + indices[10]] = alpha * (p[7] * T2_local[10] + p[8] * T2_local[9]) + beta * A[h + indices[10]];  // daca + daac
                                            A[h + indices[11]] = alpha * (p[7] * T2_local[11] + p[8] * T2_local[11]) + beta * A[h + indices[11]]; // dcaa + dcaa
                                        }
                                        else if (a > b && b == c && c == d)
                                        {
                                            double T1_local[4];
                                            double T2_local[4];

                                            int64_t indices[4];
                                            indices[0] = a * nvvv + b * nvv + b * nvir + b; // abbb
                                            indices[1] = b * nvvv + a * nvv + b * nvir + b; // babb
                                            indices[2] = b * nvvv + b * nvv + a * nvir + b; // bbab
                                            indices[3] = b * nvvv + b * nvv + b * nvir + a; // bbba

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[3]]; // abbb + babb + bbab + bbba
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[1]]; // babb + abbb + babb + babb
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[2]]; // bbab + bbab + abbb + bbab
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[0]]; // bbba + bbba + bbba + abbb

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[0] + p[6] * T1_local[0]; // abbb + abbb + abbb
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[2] + p[6] * T1_local[3]; // babb + bbab + bbba
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[1] + p[6] * T1_local[2]; // bbab + babb + bbab
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[3] + p[6] * T1_local[1]; // bbba + bbba + babb

                                            A[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[0]) + beta * A[h + indices[0]]; // abbb + abbb
                                            A[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[1]) + beta * A[h + indices[1]]; // babb + babb
                                            A[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[3]) + beta * A[h + indices[2]]; // bbab + bbba
                                            A[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[2]) + beta * A[h + indices[3]]; // bbba + bbab
                                        }
                                        else if (a == b && b == c && c > d)
                                        {
                                            double T1_local[4];
                                            double T2_local[4];

                                            int64_t indices[4];
                                            indices[0] = a * nvvv + a * nvv + a * nvir + d; // aaad
                                            indices[1] = a * nvvv + a * nvv + d * nvir + a; // aada
                                            indices[2] = a * nvvv + d * nvv + a * nvir + a; // adaa
                                            indices[3] = d * nvvv + a * nvv + a * nvir + a; // daaa

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[3]]; // aaad + aaad + aaad + daaa
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[1]]; // aada + aada + daaa + aada
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[2]]; // adaa + daaa + adaa + adaa
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[0]]; // daaa + adaa + aada + aaad

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[0] + p[6] * T1_local[2]; // aaad + aaad + adaa
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[2] + p[6] * T1_local[1]; // aada + adaa + aada
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[1] + p[6] * T1_local[0]; // adaa + aada + aaad
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[3] + p[6] * T1_local[3]; // daaa + daaa + daaa

                                            A[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[1]) + beta * A[h + indices[0]]; // aaad + aada
                                            A[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[0]) + beta * A[h + indices[1]]; // aada + aaad
                                            A[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[2]) + beta * A[h + indices[2]]; // adaa + adaa
                                            A[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[3]) + beta * A[h + indices[3]]; // daaa + daaa
                                        }
                                        else if (a == b && b > c && c == d)
                                        {
                                            double T1_local[6];
                                            double T2_local[6];

                                            int64_t indices[6];
                                            indices[0] = b * nvvv + b * nvv + c * nvir + c; // bbcc
                                            indices[1] = b * nvvv + c * nvv + b * nvir + c; // bcbc
                                            indices[2] = b * nvvv + c * nvv + c * nvir + b; // bccb
                                            indices[3] = c * nvvv + b * nvv + b * nvir + c; // cbbc
                                            indices[4] = c * nvvv + b * nvv + c * nvir + b; // cbcb
                                            indices[5] = c * nvvv + c * nvv + b * nvir + b; // ccbb

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[4]]; // bbcc + bbcc + cbbc + cbcb
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[5]]; // bcbc + cbbc + bcbc + ccbb
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[4]] + p[2] * A[h + indices[5]] + p[3] * A[h + indices[2]]; // bccb + cbcb + ccbb + bccb
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[3]]; // cbbc + bcbc + bbcc + cbbc
                                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[4]] + p[3] * A[h + indices[0]]; // cbcb + bccb + cbcb + bbcc
                                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[5]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[1]]; // ccbb + ccbb + bccb + bcbc

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[1] + p[6] * T1_local[2]; // bbcc + bcbc + bccb
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[0] + p[6] * T1_local[1]; // bcbc + bbcc + bcbc
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[2] + p[6] * T1_local[0]; // bccb + bccb + bbcc
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[3] + p[6] * T1_local[5]; // cbbc + cbbc + ccbb
                                            T2_local[4] = p[4] * T1_local[4] + p[5] * T1_local[5] + p[6] * T1_local[4]; // cbcb + ccbb + cbcb
                                            T2_local[5] = p[4] * T1_local[5] + p[5] * T1_local[4] + p[6] * T1_local[3]; // ccbb + cbcb + cbbc

                                            A[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[0]) + beta * A[h + indices[0]]; // bbcc + bbcc
                                            A[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[2]) + beta * A[h + indices[1]]; // bcbc + bccb
                                            A[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[1]) + beta * A[h + indices[2]]; // bccb + bcbc
                                            A[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[4]) + beta * A[h + indices[3]]; // cbbc + cbcb
                                            A[h + indices[4]] = alpha * (p[7] * T2_local[4] + p[8] * T2_local[3]) + beta * A[h + indices[4]]; // cbcb + cbbc
                                            A[h + indices[5]] = alpha * (p[7] * T2_local[5] + p[8] * T2_local[5]) + beta * A[h + indices[5]]; // ccbb + ccbb
                                        }
                                        else if (a == b && b == c && c == d)
                                        {
                                            double T1_local[1];
                                            double T2_local[1];

                                            int64_t indices[1];
                                            indices[0] = a * nvvv + a * nvv + a * nvir + a; // aaaa

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[0]]; // aaaa + aaaa + aaaa + aaaa

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[0] + p[6] * T1_local[0]; // aaaa + aaaa + aaaa

                                            A[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[0]) + beta * A[h + indices[0]]; // aaaa + aaaa
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// A unified function for the four-fold symmetry operations
void t4_symm_c(const double *A, double *B, int64_t nocc4, int64_t nvir, char *pattern, double alpha, double beta)
{
    int64_t ijkl;
    const int64_t bl = 8;
    int64_t nvv = nvir * nvir;
    int64_t nvvv = nvir * nvv;
    int64_t nvvvv = nvir * nvvv;

    // Coefficients for the permutation pattern
    double p[9];

    if (strcmp(pattern, "11111111") == 0)
    {
        // (1 + P_c^d) (1 + P_b^c + P_b^d) (1 + P_a^b + P_a^c + P_a^d) abcd
        for (int i = 0; i < 9; i++)
            p[i] = 1.0;
    }
    else if (strcmp(pattern, "cnnn") == 0)
    {
        // (1 + 0 * P_c^d) (1 + 0 * P_b^c + 0 * P_b^d) (2 - P_a^b - P_a^c - P_a^d) abcd
        p[0] = 2.0;
        p[1] = -1.0;
        p[2] = -1.0;
        p[3] = -1.0;
        p[4] = 1.0;
        p[5] = 0.0;
        p[6] = 0.0;
        p[7] = 1.0;
        p[8] = 0.0;
    }
    else if (strcmp(pattern, "ccnn") == 0)
    {
        // (1 + 0 * P_c^d) (2 - P_b^c - P_b^d) (2 - P_a^b - P_a^c - P_a^d) abcd
        p[0] = 2.0;
        p[1] = -1.0;
        p[2] = -1.0;
        p[3] = -1.0;
        p[4] = 2.0;
        p[5] = -1.0;
        p[6] = -1.0;
        p[7] = 1.0;
        p[8] = 0.0;
    }
    else
    {
        fprintf(stderr, "Error: unrecognized pattern \"%s\"\n", pattern);
        return;
    }

#pragma omp parallel for schedule(static)
    for (ijkl = 0; ijkl < nocc4; ijkl++)
    {
        int64_t h = ijkl * nvvvv;
        for (int64_t a0 = 0; a0 < nvir; a0 += bl)
        {
            for (int64_t b0 = 0; b0 <= a0; b0 += bl)
            {
                for (int64_t c0 = 0; c0 <= b0; c0 += bl)
                {
                    for (int64_t d0 = 0; d0 <= c0; d0 += bl)
                    {
                        for (int64_t a = a0; a < a0 + bl && a < nvir; a++)
                        {
                            for (int64_t b = b0; b < b0 + bl && b <= a; b++)
                            {
                                for (int64_t c = c0; c < c0 + bl && c <= b; c++)
                                {
                                    for (int64_t d = d0; d < d0 + bl && d <= c; d++)
                                    {
                                        if (a > b && b > c && c > d)
                                        {
                                            double T1_local[24];
                                            double T2_local[24];

                                            int64_t indices[24];
                                            indices[0] = a * nvvv + b * nvv + c * nvir + d; // abcd
                                            indices[1] = a * nvvv + b * nvv + d * nvir + c; // abdc
                                            indices[2] = a * nvvv + c * nvv + b * nvir + d; // acbd
                                            indices[3] = a * nvvv + c * nvv + d * nvir + b; // acdb
                                            indices[4] = a * nvvv + d * nvv + b * nvir + c; // adbc
                                            indices[5] = a * nvvv + d * nvv + c * nvir + b; // adcb

                                            indices[6] = b * nvvv + a * nvv + c * nvir + d;  // bacd
                                            indices[7] = b * nvvv + a * nvv + d * nvir + c;  // badc
                                            indices[8] = b * nvvv + c * nvv + a * nvir + d;  // bcad
                                            indices[9] = b * nvvv + c * nvv + d * nvir + a;  // bcda
                                            indices[10] = b * nvvv + d * nvv + a * nvir + c; // bdac
                                            indices[11] = b * nvvv + d * nvv + c * nvir + a; // bdca

                                            indices[12] = c * nvvv + a * nvv + b * nvir + d; // cabd
                                            indices[13] = c * nvvv + a * nvv + d * nvir + b; // cadb
                                            indices[14] = c * nvvv + b * nvv + a * nvir + d; // cbad
                                            indices[15] = c * nvvv + b * nvv + d * nvir + a; // cbda
                                            indices[16] = c * nvvv + d * nvv + a * nvir + b; // cdab
                                            indices[17] = c * nvvv + d * nvv + b * nvir + a; // cdba

                                            indices[18] = d * nvvv + a * nvv + b * nvir + c; // dabc
                                            indices[19] = d * nvvv + a * nvv + c * nvir + b; // dacb
                                            indices[20] = d * nvvv + b * nvv + a * nvir + c; // dbac
                                            indices[21] = d * nvvv + b * nvv + c * nvir + a; // dbca
                                            indices[22] = d * nvvv + c * nvv + a * nvir + b; // dcab
                                            indices[23] = d * nvvv + c * nvv + b * nvir + a; // dcba

                                            // (1 + P_a^b + P_a^c + P_a^d)
                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[6]] + p[2] * A[h + indices[14]] + p[3] * A[h + indices[21]];   // abcd + bacd + cbad + dbca
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[7]] + p[2] * A[h + indices[20]] + p[3] * A[h + indices[15]];   // abdc + badc + dbac + cbda
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[12]] + p[2] * A[h + indices[8]] + p[3] * A[h + indices[23]];   // acbd + cabd + bcad + dcba
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[13]] + p[2] * A[h + indices[22]] + p[3] * A[h + indices[9]];   // acdb + cadb + dcab + bcda
                                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[18]] + p[2] * A[h + indices[10]] + p[3] * A[h + indices[17]];  // adbc + dabc + bdac + cdba
                                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[19]] + p[2] * A[h + indices[16]] + p[3] * A[h + indices[11]];  // adcb + dacb + cdab + bdca
                                            T1_local[6] = p[0] * A[h + indices[6]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[12]] + p[3] * A[h + indices[19]];   // bacd + abcd + cabd + dacb
                                            T1_local[7] = p[0] * A[h + indices[7]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[18]] + p[3] * A[h + indices[13]];   // badc + abdc + dabc + cadb
                                            T1_local[8] = p[0] * A[h + indices[8]] + p[1] * A[h + indices[14]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[22]];   // bcad + cbad + acbd + dcab
                                            T1_local[9] = p[0] * A[h + indices[9]] + p[1] * A[h + indices[15]] + p[2] * A[h + indices[23]] + p[3] * A[h + indices[3]];   // bcda + cbda + dcba + acdb
                                            T1_local[10] = p[0] * A[h + indices[10]] + p[1] * A[h + indices[20]] + p[2] * A[h + indices[4]] + p[3] * A[h + indices[16]]; // bdac + dbac + adbc + cdab
                                            T1_local[11] = p[0] * A[h + indices[11]] + p[1] * A[h + indices[21]] + p[2] * A[h + indices[17]] + p[3] * A[h + indices[5]]; // bdca + dbca + cdba + adcb
                                            T1_local[12] = p[0] * A[h + indices[12]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[6]] + p[3] * A[h + indices[18]];  // cabd + acbd + bacd + dabc
                                            T1_local[13] = p[0] * A[h + indices[13]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[19]] + p[3] * A[h + indices[7]];  // cadb + acdb + dacb + badc
                                            T1_local[14] = p[0] * A[h + indices[14]] + p[1] * A[h + indices[8]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[20]];  // cbad + bcad + abcd + dbac
                                            T1_local[15] = p[0] * A[h + indices[15]] + p[1] * A[h + indices[9]] + p[2] * A[h + indices[21]] + p[3] * A[h + indices[1]];  // cbda + bcda + dbca + abdc
                                            T1_local[16] = p[0] * A[h + indices[16]] + p[1] * A[h + indices[22]] + p[2] * A[h + indices[5]] + p[3] * A[h + indices[10]]; // cdab + dcab + adcb + bdac
                                            T1_local[17] = p[0] * A[h + indices[17]] + p[1] * A[h + indices[23]] + p[2] * A[h + indices[11]] + p[3] * A[h + indices[4]]; // cdba + dcba + bdca + adbc
                                            T1_local[18] = p[0] * A[h + indices[18]] + p[1] * A[h + indices[4]] + p[2] * A[h + indices[7]] + p[3] * A[h + indices[12]];  // dabc + adbc + badc + cabd
                                            T1_local[19] = p[0] * A[h + indices[19]] + p[1] * A[h + indices[5]] + p[2] * A[h + indices[13]] + p[3] * A[h + indices[6]];  // dacb + adcb + cadb + bacd
                                            T1_local[20] = p[0] * A[h + indices[20]] + p[1] * A[h + indices[10]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[14]]; // dbac + bdac + abdc + cbad
                                            T1_local[21] = p[0] * A[h + indices[21]] + p[1] * A[h + indices[11]] + p[2] * A[h + indices[15]] + p[3] * A[h + indices[0]]; // dbca + bdca + cbda + abcd
                                            T1_local[22] = p[0] * A[h + indices[22]] + p[1] * A[h + indices[16]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[8]];  // dcab + cdab + acdb + bcad
                                            T1_local[23] = p[0] * A[h + indices[23]] + p[1] * A[h + indices[17]] + p[2] * A[h + indices[9]] + p[3] * A[h + indices[2]];  // dcba + cdba + bcda + acbd

                                            // (1 + P_b^c + P_b^d)
                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[2] + p[6] * T1_local[5];     // abcd + acbd + adcb
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[4] + p[6] * T1_local[3];     // abdc + adbc + acdb
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[0] + p[6] * T1_local[4];     // acbd + abcd + adbc
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[5] + p[6] * T1_local[1];     // acdb + adcb + abdc
                                            T2_local[4] = p[4] * T1_local[4] + p[5] * T1_local[1] + p[6] * T1_local[2];     // adbc + abdc + acbd
                                            T2_local[5] = p[4] * T1_local[5] + p[5] * T1_local[3] + p[6] * T1_local[0];     // adcb + acdb + abcd
                                            T2_local[6] = p[4] * T1_local[6] + p[5] * T1_local[8] + p[6] * T1_local[11];    // bacd + bcad + bdca
                                            T2_local[7] = p[4] * T1_local[7] + p[5] * T1_local[10] + p[6] * T1_local[9];    // badc + bdac + bcda
                                            T2_local[8] = p[4] * T1_local[8] + p[5] * T1_local[6] + p[6] * T1_local[10];    // bcad + bacd + bdac
                                            T2_local[9] = p[4] * T1_local[9] + p[5] * T1_local[11] + p[6] * T1_local[7];    // bcda + bdca + badc
                                            T2_local[10] = p[4] * T1_local[10] + p[5] * T1_local[7] + p[6] * T1_local[8];   // bdac + badc + bcad
                                            T2_local[11] = p[4] * T1_local[11] + p[5] * T1_local[9] + p[6] * T1_local[6];   // bdca + bcda + bacd
                                            T2_local[12] = p[4] * T1_local[12] + p[5] * T1_local[14] + p[6] * T1_local[17]; // cabd + cbad + cdba
                                            T2_local[13] = p[4] * T1_local[13] + p[5] * T1_local[16] + p[6] * T1_local[15]; // cadb + cdab + cbda
                                            T2_local[14] = p[4] * T1_local[14] + p[5] * T1_local[12] + p[6] * T1_local[16]; // cbad + cabd + cdab
                                            T2_local[15] = p[4] * T1_local[15] + p[5] * T1_local[17] + p[6] * T1_local[13]; // cbda + cdba + cadb
                                            T2_local[16] = p[4] * T1_local[16] + p[5] * T1_local[13] + p[6] * T1_local[14]; // cdab + cadb + cbad
                                            T2_local[17] = p[4] * T1_local[17] + p[5] * T1_local[15] + p[6] * T1_local[12]; // cdba + cbda + cabd
                                            T2_local[18] = p[4] * T1_local[18] + p[5] * T1_local[20] + p[6] * T1_local[23]; // dabc + dbac + dcba
                                            T2_local[19] = p[4] * T1_local[19] + p[5] * T1_local[22] + p[6] * T1_local[21]; // dacb + dcab + dbca
                                            T2_local[20] = p[4] * T1_local[20] + p[5] * T1_local[18] + p[6] * T1_local[22]; // dbac + dabc + dcab
                                            T2_local[21] = p[4] * T1_local[21] + p[5] * T1_local[23] + p[6] * T1_local[19]; // dbca + dcba + dacb
                                            T2_local[22] = p[4] * T1_local[22] + p[5] * T1_local[19] + p[6] * T1_local[20]; // dcab + dacb + dbac
                                            T2_local[23] = p[4] * T1_local[23] + p[5] * T1_local[21] + p[6] * T1_local[18]; // dcba + dbca + dabc

                                            // (1 + P_c^d)
                                            B[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[1]) + beta * A[h + indices[0]];     // abcd + abdc
                                            B[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[0]) + beta * A[h + indices[1]];     // abdc + abcd
                                            B[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[3]) + beta * A[h + indices[2]];     // acbd + acdb
                                            B[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[2]) + beta * A[h + indices[3]];     // acdb + acbd
                                            B[h + indices[4]] = alpha * (p[7] * T2_local[4] + p[8] * T2_local[5]) + beta * A[h + indices[4]];     // adbc + adcb
                                            B[h + indices[5]] = alpha * (p[7] * T2_local[5] + p[8] * T2_local[4]) + beta * A[h + indices[5]];     // adcb + adbc
                                            B[h + indices[6]] = alpha * (p[7] * T2_local[6] + p[8] * T2_local[7]) + beta * A[h + indices[6]];     // bacd + badc
                                            B[h + indices[7]] = alpha * (p[7] * T2_local[7] + p[8] * T2_local[6]) + beta * A[h + indices[7]];     // badc + bacd
                                            B[h + indices[8]] = alpha * (p[7] * T2_local[8] + p[8] * T2_local[9]) + beta * A[h + indices[8]];     // bcad + bcda
                                            B[h + indices[9]] = alpha * (p[7] * T2_local[9] + p[8] * T2_local[8]) + beta * A[h + indices[9]];     // bcda + bcad
                                            B[h + indices[10]] = alpha * (p[7] * T2_local[10] + p[8] * T2_local[11]) + beta * A[h + indices[10]]; // bdac + bdca
                                            B[h + indices[11]] = alpha * (p[7] * T2_local[11] + p[8] * T2_local[10]) + beta * A[h + indices[11]]; // bdca + bdac
                                            B[h + indices[12]] = alpha * (p[7] * T2_local[12] + p[8] * T2_local[13]) + beta * A[h + indices[12]]; // cabd + cadb
                                            B[h + indices[13]] = alpha * (p[7] * T2_local[13] + p[8] * T2_local[12]) + beta * A[h + indices[13]]; // cadb + cabd
                                            B[h + indices[14]] = alpha * (p[7] * T2_local[14] + p[8] * T2_local[15]) + beta * A[h + indices[14]]; // cbad + cbda
                                            B[h + indices[15]] = alpha * (p[7] * T2_local[15] + p[8] * T2_local[14]) + beta * A[h + indices[15]]; // cbda + cbad
                                            B[h + indices[16]] = alpha * (p[7] * T2_local[16] + p[8] * T2_local[17]) + beta * A[h + indices[16]]; // cdab + cdba
                                            B[h + indices[17]] = alpha * (p[7] * T2_local[17] + p[8] * T2_local[16]) + beta * A[h + indices[17]]; // cdba + cdab
                                            B[h + indices[18]] = alpha * (p[7] * T2_local[18] + p[8] * T2_local[19]) + beta * A[h + indices[18]]; // dabc + dacb
                                            B[h + indices[19]] = alpha * (p[7] * T2_local[19] + p[8] * T2_local[18]) + beta * A[h + indices[19]]; // dacb + dabc
                                            B[h + indices[20]] = alpha * (p[7] * T2_local[20] + p[8] * T2_local[21]) + beta * A[h + indices[20]]; // dbac + dbca
                                            B[h + indices[21]] = alpha * (p[7] * T2_local[21] + p[8] * T2_local[20]) + beta * A[h + indices[21]]; // dbca + dbac
                                            B[h + indices[22]] = alpha * (p[7] * T2_local[22] + p[8] * T2_local[23]) + beta * A[h + indices[22]]; // dcab + dcba
                                            B[h + indices[23]] = alpha * (p[7] * T2_local[23] + p[8] * T2_local[22]) + beta * A[h + indices[23]]; // dcba + dcab
                                        }
                                        else if (a > b && b > c && c == d)
                                        {
                                            double T1_local[12];
                                            double T2_local[12];

                                            int64_t indices[12];
                                            indices[0] = a * nvvv + b * nvv + c * nvir + c;  // abcc
                                            indices[1] = a * nvvv + c * nvv + b * nvir + c;  // acbc
                                            indices[2] = a * nvvv + c * nvv + c * nvir + b;  // accb
                                            indices[3] = b * nvvv + a * nvv + c * nvir + c;  // bacc
                                            indices[4] = b * nvvv + c * nvv + a * nvir + c;  // bcac
                                            indices[5] = b * nvvv + c * nvv + c * nvir + a;  // bcca
                                            indices[6] = c * nvvv + a * nvv + b * nvir + c;  // cabc
                                            indices[7] = c * nvvv + a * nvv + c * nvir + b;  // cacb
                                            indices[8] = c * nvvv + b * nvv + a * nvir + c;  // cbac
                                            indices[9] = c * nvvv + b * nvv + c * nvir + a;  // cbca
                                            indices[10] = c * nvvv + c * nvv + a * nvir + b; // ccab
                                            indices[11] = c * nvvv + c * nvv + b * nvir + a; // ccba

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[8]] + p[3] * A[h + indices[9]];    // abcc + bacc + cbac + cbca
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[6]] + p[2] * A[h + indices[4]] + p[3] * A[h + indices[11]];   // acbc + cabc + bcac + ccba
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[7]] + p[2] * A[h + indices[10]] + p[3] * A[h + indices[5]];   // accb + cacb + ccab + bcca
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[6]] + p[3] * A[h + indices[7]];    // bacc + abcc + cabc + cacb
                                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[8]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[10]];   // bcac + cbac + acbc + ccab
                                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[9]] + p[2] * A[h + indices[11]] + p[3] * A[h + indices[2]];   // bcca + cbca + ccba + accb
                                            T1_local[6] = p[0] * A[h + indices[6]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[6]];    // cabc + acbc + bacc + cabc
                                            T1_local[7] = p[0] * A[h + indices[7]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[7]] + p[3] * A[h + indices[3]];    // cacb + accb + cacb + bacc
                                            T1_local[8] = p[0] * A[h + indices[8]] + p[1] * A[h + indices[4]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[8]];    // cbac + bcac + abcc + cbac
                                            T1_local[9] = p[0] * A[h + indices[9]] + p[1] * A[h + indices[5]] + p[2] * A[h + indices[9]] + p[3] * A[h + indices[0]];    // cbca + bcca + cbca + abcc
                                            T1_local[10] = p[0] * A[h + indices[10]] + p[1] * A[h + indices[10]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[4]]; // ccab + ccab + accb + bcac
                                            T1_local[11] = p[0] * A[h + indices[11]] + p[1] * A[h + indices[11]] + p[2] * A[h + indices[5]] + p[3] * A[h + indices[1]]; // ccba + ccba + bcca + acbc

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[1] + p[6] * T1_local[2];   // abcc + acbc + accb
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[0] + p[6] * T1_local[1];   // acbc + abcc + acbc
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[2] + p[6] * T1_local[0];   // accb + accb + abcc
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[4] + p[6] * T1_local[5];   // bacc + bcac + bcca
                                            T2_local[4] = p[4] * T1_local[4] + p[5] * T1_local[3] + p[6] * T1_local[4];   // bcac + bacc + bcac
                                            T2_local[5] = p[4] * T1_local[5] + p[5] * T1_local[5] + p[6] * T1_local[3];   // bcca + bcca + bacc
                                            T2_local[6] = p[4] * T1_local[6] + p[5] * T1_local[8] + p[6] * T1_local[11];  // cabc + cbac + ccba
                                            T2_local[7] = p[4] * T1_local[7] + p[5] * T1_local[10] + p[6] * T1_local[9];  // cacb + ccab + cbca
                                            T2_local[8] = p[4] * T1_local[8] + p[5] * T1_local[6] + p[6] * T1_local[10];  // cbac + cabc + ccab
                                            T2_local[9] = p[4] * T1_local[9] + p[5] * T1_local[11] + p[6] * T1_local[7];  // cbca + ccba + cacb
                                            T2_local[10] = p[4] * T1_local[10] + p[5] * T1_local[7] + p[6] * T1_local[8]; // ccab + cacb + cbac
                                            T2_local[11] = p[4] * T1_local[11] + p[5] * T1_local[9] + p[6] * T1_local[6]; // ccba + cbca + cabc

                                            B[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[0]) + beta * A[h + indices[0]];     // abcc + abcc
                                            B[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[2]) + beta * A[h + indices[1]];     // acbc + accb
                                            B[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[1]) + beta * A[h + indices[2]];     // accb + acbc
                                            B[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[3]) + beta * A[h + indices[3]];     // bacc + bacc
                                            B[h + indices[4]] = alpha * (p[7] * T2_local[4] + p[8] * T2_local[5]) + beta * A[h + indices[4]];     // bcac + bcca
                                            B[h + indices[5]] = alpha * (p[7] * T2_local[5] + p[8] * T2_local[4]) + beta * A[h + indices[5]];     // bcca + bcac
                                            B[h + indices[6]] = alpha * (p[7] * T2_local[6] + p[8] * T2_local[7]) + beta * A[h + indices[6]];     // cabc + cacb
                                            B[h + indices[7]] = alpha * (p[7] * T2_local[7] + p[8] * T2_local[6]) + beta * A[h + indices[7]];     // cacb + cabc
                                            B[h + indices[8]] = alpha * (p[7] * T2_local[8] + p[8] * T2_local[9]) + beta * A[h + indices[8]];     // cbac + cbca
                                            B[h + indices[9]] = alpha * (p[7] * T2_local[9] + p[8] * T2_local[8]) + beta * A[h + indices[9]];     // cbca + cbac
                                            B[h + indices[10]] = alpha * (p[7] * T2_local[10] + p[8] * T2_local[11]) + beta * A[h + indices[10]]; // ccab + ccba
                                            B[h + indices[11]] = alpha * (p[7] * T2_local[11] + p[8] * T2_local[10]) + beta * A[h + indices[11]]; // ccba + ccab
                                        }
                                        else if (a > b && b == c && c > d)
                                        {
                                            double T1_local[12];
                                            double T2_local[12];

                                            int64_t indices[12];
                                            indices[0] = a * nvvv + b * nvv + b * nvir + d;  // abbd
                                            indices[1] = a * nvvv + b * nvv + d * nvir + b;  // abdb
                                            indices[2] = a * nvvv + d * nvv + b * nvir + b;  // adbb
                                            indices[3] = b * nvvv + a * nvv + b * nvir + d;  // babd
                                            indices[4] = b * nvvv + a * nvv + d * nvir + b;  // badb
                                            indices[5] = b * nvvv + b * nvv + a * nvir + d;  // bbad
                                            indices[6] = b * nvvv + b * nvv + d * nvir + a;  // bbda
                                            indices[7] = b * nvvv + d * nvv + a * nvir + b;  // bdab
                                            indices[8] = b * nvvv + d * nvv + b * nvir + a;  // bdba
                                            indices[9] = d * nvvv + a * nvv + b * nvir + b;  // dabb
                                            indices[10] = d * nvvv + b * nvv + a * nvir + b; // dbab
                                            indices[11] = d * nvvv + b * nvv + b * nvir + a; // dbba

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[5]] + p[3] * A[h + indices[11]];  // abbd + babd + bbad + dbba
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[4]] + p[2] * A[h + indices[10]] + p[3] * A[h + indices[6]];  // abdb + badb + dbab + bbda
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[9]] + p[2] * A[h + indices[7]] + p[3] * A[h + indices[8]];   // adbb + dabb + bdab + bdba
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[9]];   // babd + abbd + babd + dabb
                                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[9]] + p[3] * A[h + indices[4]];   // badb + abdb + dabb + badb
                                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[5]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[10]];  // bbad + bbad + abbd + dbab
                                            T1_local[6] = p[0] * A[h + indices[6]] + p[1] * A[h + indices[6]] + p[2] * A[h + indices[11]] + p[3] * A[h + indices[1]];  // bbda + bbda + dbba + abdb
                                            T1_local[7] = p[0] * A[h + indices[7]] + p[1] * A[h + indices[10]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[7]];  // bdab + dbab + adbb + bdab
                                            T1_local[8] = p[0] * A[h + indices[8]] + p[1] * A[h + indices[11]] + p[2] * A[h + indices[8]] + p[3] * A[h + indices[2]];  // bdba + dbba + bdba + adbb
                                            T1_local[9] = p[0] * A[h + indices[9]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[4]] + p[3] * A[h + indices[3]];   // dabb + adbb + badb + babd
                                            T1_local[10] = p[0] * A[h + indices[10]] + p[1] * A[h + indices[7]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[5]]; // dbab + bdab + abdb + bbad
                                            T1_local[11] = p[0] * A[h + indices[11]] + p[1] * A[h + indices[8]] + p[2] * A[h + indices[6]] + p[3] * A[h + indices[0]]; // dbba + bdba + bbda + abbd

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[0] + p[6] * T1_local[2];    // abbd + abbd + adbb
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[2] + p[6] * T1_local[1];    // abdb + adbb + abdb
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[1] + p[6] * T1_local[0];    // adbb + abdb + abbd
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[5] + p[6] * T1_local[8];    // babd + bbad + bdba
                                            T2_local[4] = p[4] * T1_local[4] + p[5] * T1_local[7] + p[6] * T1_local[6];    // badb + bdab + bbda
                                            T2_local[5] = p[4] * T1_local[5] + p[5] * T1_local[3] + p[6] * T1_local[7];    // bbad + babd + bdab
                                            T2_local[6] = p[4] * T1_local[6] + p[5] * T1_local[8] + p[6] * T1_local[4];    // bbda + bdba + badb
                                            T2_local[7] = p[4] * T1_local[7] + p[5] * T1_local[4] + p[6] * T1_local[5];    // bdab + badb + bbad
                                            T2_local[8] = p[4] * T1_local[8] + p[5] * T1_local[6] + p[6] * T1_local[3];    // bdba + bbda + babd
                                            T2_local[9] = p[4] * T1_local[9] + p[5] * T1_local[10] + p[6] * T1_local[11];  // dabb + dbab + dbba
                                            T2_local[10] = p[4] * T1_local[10] + p[5] * T1_local[9] + p[6] * T1_local[10]; // dbab + dabb + dbab
                                            T2_local[11] = p[4] * T1_local[11] + p[5] * T1_local[11] + p[6] * T1_local[9]; // dbba + dbba + dabb

                                            B[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[1]) + beta * A[h + indices[0]];     // abbd + abdb
                                            B[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[0]) + beta * A[h + indices[1]];     // abdb + abbd
                                            B[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[2]) + beta * A[h + indices[2]];     // adbb + adbb
                                            B[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[4]) + beta * A[h + indices[3]];     // babd + badb
                                            B[h + indices[4]] = alpha * (p[7] * T2_local[4] + p[8] * T2_local[3]) + beta * A[h + indices[4]];     // badb + babd
                                            B[h + indices[5]] = alpha * (p[7] * T2_local[5] + p[8] * T2_local[6]) + beta * A[h + indices[5]];     // bbad + bbda
                                            B[h + indices[6]] = alpha * (p[7] * T2_local[6] + p[8] * T2_local[5]) + beta * A[h + indices[6]];     // bbda + bbad
                                            B[h + indices[7]] = alpha * (p[7] * T2_local[7] + p[8] * T2_local[8]) + beta * A[h + indices[7]];     // bdab + bdba
                                            B[h + indices[8]] = alpha * (p[7] * T2_local[8] + p[8] * T2_local[7]) + beta * A[h + indices[8]];     // bdba + bdab
                                            B[h + indices[9]] = alpha * (p[7] * T2_local[9] + p[8] * T2_local[9]) + beta * A[h + indices[9]];     // dabb + dabb
                                            B[h + indices[10]] = alpha * (p[7] * T2_local[10] + p[8] * T2_local[11]) + beta * A[h + indices[10]]; // dbab + dbba
                                            B[h + indices[11]] = alpha * (p[7] * T2_local[11] + p[8] * T2_local[10]) + beta * A[h + indices[11]]; // dbba + dbab
                                        }
                                        else if (a == b && b > c && c > d)
                                        {
                                            double T1_local[12];
                                            double T2_local[12];

                                            int64_t indices[12];
                                            indices[0] = a * nvvv + a * nvv + c * nvir + d;  // aacd
                                            indices[1] = a * nvvv + a * nvv + d * nvir + c;  // aadc
                                            indices[2] = a * nvvv + c * nvv + a * nvir + d;  // acad
                                            indices[3] = a * nvvv + c * nvv + d * nvir + a;  // acda
                                            indices[4] = a * nvvv + d * nvv + a * nvir + c;  // adac
                                            indices[5] = a * nvvv + d * nvv + c * nvir + a;  // adca
                                            indices[6] = c * nvvv + a * nvv + a * nvir + d;  // caad
                                            indices[7] = c * nvvv + a * nvv + d * nvir + a;  // cada
                                            indices[8] = c * nvvv + d * nvv + a * nvir + a;  // cdaa
                                            indices[9] = d * nvvv + a * nvv + a * nvir + c;  // daac
                                            indices[10] = d * nvvv + a * nvv + c * nvir + a; // daca
                                            indices[11] = d * nvvv + c * nvv + a * nvir + a; // dcaa

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[6]] + p[3] * A[h + indices[10]];  // aacd + aacd + caad + daca
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[9]] + p[3] * A[h + indices[7]];   // aadc + aadc + daac + cada
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[6]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[11]];  // acad + caad + acad + dcaa
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[7]] + p[2] * A[h + indices[11]] + p[3] * A[h + indices[3]];  // acda + cada + dcaa + acda
                                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[9]] + p[2] * A[h + indices[4]] + p[3] * A[h + indices[8]];   // adac + daac + adac + cdaa
                                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[10]] + p[2] * A[h + indices[8]] + p[3] * A[h + indices[5]];  // adca + daca + cdaa + adca
                                            T1_local[6] = p[0] * A[h + indices[6]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[9]];   // caad + acad + aacd + daac
                                            T1_local[7] = p[0] * A[h + indices[7]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[10]] + p[3] * A[h + indices[1]];  // cada + acda + daca + aadc
                                            T1_local[8] = p[0] * A[h + indices[8]] + p[1] * A[h + indices[11]] + p[2] * A[h + indices[5]] + p[3] * A[h + indices[4]];  // cdaa + dcaa + adca + adac
                                            T1_local[9] = p[0] * A[h + indices[9]] + p[1] * A[h + indices[4]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[6]];   // daac + adac + aadc + caad
                                            T1_local[10] = p[0] * A[h + indices[10]] + p[1] * A[h + indices[5]] + p[2] * A[h + indices[7]] + p[3] * A[h + indices[0]]; // daca + adca + cada + aacd
                                            T1_local[11] = p[0] * A[h + indices[11]] + p[1] * A[h + indices[8]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[2]]; // dcaa + cdaa + acda + acad

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[2] + p[6] * T1_local[5];     // aacd + acad + adca
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[4] + p[6] * T1_local[3];     // aadc + adac + acda
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[0] + p[6] * T1_local[4];     // acad + aacd + adac
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[5] + p[6] * T1_local[1];     // acda + adca + aadc
                                            T2_local[4] = p[4] * T1_local[4] + p[5] * T1_local[1] + p[6] * T1_local[2];     // adac + aadc + acad
                                            T2_local[5] = p[4] * T1_local[5] + p[5] * T1_local[3] + p[6] * T1_local[0];     // adca + acda + aacd
                                            T2_local[6] = p[4] * T1_local[6] + p[5] * T1_local[6] + p[6] * T1_local[8];     // caad + caad + cdaa
                                            T2_local[7] = p[4] * T1_local[7] + p[5] * T1_local[8] + p[6] * T1_local[7];     // cada + cdaa + cada
                                            T2_local[8] = p[4] * T1_local[8] + p[5] * T1_local[7] + p[6] * T1_local[6];     // cdaa + cada + caad
                                            T2_local[9] = p[4] * T1_local[9] + p[5] * T1_local[9] + p[6] * T1_local[11];    // daac + daac + dcaa
                                            T2_local[10] = p[4] * T1_local[10] + p[5] * T1_local[11] + p[6] * T1_local[10]; // daca + dcaa + daca
                                            T2_local[11] = p[4] * T1_local[11] + p[5] * T1_local[10] + p[6] * T1_local[9];  // dcaa + daca + daac

                                            B[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[1]) + beta * A[h + indices[0]];     // aacd + aadc
                                            B[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[0]) + beta * A[h + indices[1]];     // aadc + aacd
                                            B[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[3]) + beta * A[h + indices[2]];     // acad + acda
                                            B[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[2]) + beta * A[h + indices[3]];     // acda + acad
                                            B[h + indices[4]] = alpha * (p[7] * T2_local[4] + p[8] * T2_local[5]) + beta * A[h + indices[4]];     // adac + adca
                                            B[h + indices[5]] = alpha * (p[7] * T2_local[5] + p[8] * T2_local[4]) + beta * A[h + indices[5]];     // adca + adac
                                            B[h + indices[6]] = alpha * (p[7] * T2_local[6] + p[8] * T2_local[7]) + beta * A[h + indices[6]];     // caad + cada
                                            B[h + indices[7]] = alpha * (p[7] * T2_local[7] + p[8] * T2_local[6]) + beta * A[h + indices[7]];     // cada + caad
                                            B[h + indices[8]] = alpha * (p[7] * T2_local[8] + p[8] * T2_local[8]) + beta * A[h + indices[8]];     // cdaa + cdaa
                                            B[h + indices[9]] = alpha * (p[7] * T2_local[9] + p[8] * T2_local[10]) + beta * A[h + indices[9]];    // daac + daca
                                            B[h + indices[10]] = alpha * (p[7] * T2_local[10] + p[8] * T2_local[9]) + beta * A[h + indices[10]];  // daca + daac
                                            B[h + indices[11]] = alpha * (p[7] * T2_local[11] + p[8] * T2_local[11]) + beta * A[h + indices[11]]; // dcaa + dcaa
                                        }
                                        else if (a > b && b == c && c == d)
                                        {
                                            double T1_local[4];
                                            double T2_local[4];

                                            int64_t indices[4];
                                            indices[0] = a * nvvv + b * nvv + b * nvir + b; // abbb
                                            indices[1] = b * nvvv + a * nvv + b * nvir + b; // babb
                                            indices[2] = b * nvvv + b * nvv + a * nvir + b; // bbab
                                            indices[3] = b * nvvv + b * nvv + b * nvir + a; // bbba

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[3]]; // abbb + babb + bbab + bbba
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[1]]; // babb + abbb + babb + babb
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[2]]; // bbab + bbab + abbb + bbab
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[0]]; // bbba + bbba + bbba + abbb

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[0] + p[6] * T1_local[0]; // abbb + abbb + abbb
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[2] + p[6] * T1_local[3]; // babb + bbab + bbba
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[1] + p[6] * T1_local[2]; // bbab + babb + bbab
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[3] + p[6] * T1_local[1]; // bbba + bbba + babb

                                            B[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[0]) + beta * A[h + indices[0]]; // abbb + abbb
                                            B[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[1]) + beta * A[h + indices[1]]; // babb + babb
                                            B[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[3]) + beta * A[h + indices[2]]; // bbab + bbba
                                            B[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[2]) + beta * A[h + indices[3]]; // bbba + bbab
                                        }
                                        else if (a == b && b == c && c > d)
                                        {
                                            double T1_local[4];
                                            double T2_local[4];

                                            int64_t indices[4];
                                            indices[0] = a * nvvv + a * nvv + a * nvir + d; // aaad
                                            indices[1] = a * nvvv + a * nvv + d * nvir + a; // aada
                                            indices[2] = a * nvvv + d * nvv + a * nvir + a; // adaa
                                            indices[3] = d * nvvv + a * nvv + a * nvir + a; // daaa

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[3]]; // aaad + aaad + aaad + daaa
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[1]]; // aada + aada + daaa + aada
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[2]]; // adaa + daaa + adaa + adaa
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[0]]; // daaa + adaa + aada + aaad

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[0] + p[6] * T1_local[2]; // aaad + aaad + adaa
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[2] + p[6] * T1_local[1]; // aada + adaa + aada
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[1] + p[6] * T1_local[0]; // adaa + aada + aaad
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[3] + p[6] * T1_local[3]; // daaa + daaa + daaa

                                            B[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[1]) + beta * A[h + indices[0]]; // aaad + aada
                                            B[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[0]) + beta * A[h + indices[1]]; // aada + aaad
                                            B[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[2]) + beta * A[h + indices[2]]; // adaa + adaa
                                            B[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[3]) + beta * A[h + indices[3]]; // daaa + daaa
                                        }
                                        else if (a == b && b > c && c == d)
                                        {
                                            double T1_local[6];
                                            double T2_local[6];

                                            int64_t indices[6];
                                            indices[0] = b * nvvv + b * nvv + c * nvir + c; // bbcc
                                            indices[1] = b * nvvv + c * nvv + b * nvir + c; // bcbc
                                            indices[2] = b * nvvv + c * nvv + c * nvir + b; // bccb
                                            indices[3] = c * nvvv + b * nvv + b * nvir + c; // cbbc
                                            indices[4] = c * nvvv + b * nvv + c * nvir + b; // cbcb
                                            indices[5] = c * nvvv + c * nvv + b * nvir + b; // ccbb

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[4]]; // bbcc + bbcc + cbbc + cbcb
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[5]]; // bcbc + cbbc + bcbc + ccbb
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[4]] + p[2] * A[h + indices[5]] + p[3] * A[h + indices[2]]; // bccb + cbcb + ccbb + bccb
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[3]]; // cbbc + bcbc + bbcc + cbbc
                                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[4]] + p[3] * A[h + indices[0]]; // cbcb + bccb + cbcb + bbcc
                                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[5]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[1]]; // ccbb + ccbb + bccb + bcbc

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[1] + p[6] * T1_local[2]; // bbcc + bcbc + bccb
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[0] + p[6] * T1_local[1]; // bcbc + bbcc + bcbc
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[2] + p[6] * T1_local[0]; // bccb + bccb + bbcc
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[3] + p[6] * T1_local[5]; // cbbc + cbbc + ccbb
                                            T2_local[4] = p[4] * T1_local[4] + p[5] * T1_local[5] + p[6] * T1_local[4]; // cbcb + ccbb + cbcb
                                            T2_local[5] = p[4] * T1_local[5] + p[5] * T1_local[4] + p[6] * T1_local[3]; // ccbb + cbcb + cbbc

                                            B[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[0]) + beta * A[h + indices[0]]; // bbcc + bbcc
                                            B[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[2]) + beta * A[h + indices[1]]; // bcbc + bccb
                                            B[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[1]) + beta * A[h + indices[2]]; // bccb + bcbc
                                            B[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[4]) + beta * A[h + indices[3]]; // cbbc + cbcb
                                            B[h + indices[4]] = alpha * (p[7] * T2_local[4] + p[8] * T2_local[3]) + beta * A[h + indices[4]]; // cbcb + cbbc
                                            B[h + indices[5]] = alpha * (p[7] * T2_local[5] + p[8] * T2_local[5]) + beta * A[h + indices[5]]; // ccbb + ccbb
                                        }
                                        else if (a == b && b == c && c == d)
                                        {
                                            double T1_local[1];
                                            double T2_local[1];

                                            int64_t indices[1];
                                            indices[0] = a * nvvv + a * nvv + a * nvir + a; // aaaa

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[0]]; // aaaa + aaaa + aaaa + aaaa

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[0] + p[6] * T1_local[0]; // aaaa + aaaa + aaaa

                                            B[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[0]) + beta * A[h + indices[0]]; // aaaa + aaaa
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void t4_p_sum_ip_c(double *A, int64_t nocc, int64_t nvir, double alpha, double beta)
{
    // test bl
    const int64_t bl = 8;
    int64_t nvv = nvir * nvir;
    int64_t nvvv = nvir * nvv;
    int64_t nvvvv = nvir * nvvv;
    int64_t novvvv = nocc * nvvvv;
    int64_t noovvvv = nocc * novvvv;
    int64_t nooovvvv = nocc * noovvvv;

    int64_t ntriplets = 0;
    for (int i = 0; i < nocc; i++)
        for (int j = 0; j <= i; j++)
            for (int k = 0; k <= j; k++)
                for (int l = 0; l <= k; l++)
                    ntriplets++;

    const int64_t map1[24][4] = {
        {0, 6, 14, 21},
        {1, 7, 20, 15},
        {2, 12, 8, 23},
        {3, 13, 22, 9},
        {4, 18, 10, 17},
        {5, 19, 16, 11},
        {6, 0, 12, 19},
        {7, 1, 18, 13},
        {8, 14, 2, 22},
        {9, 15, 23, 3},
        {10, 20, 4, 16},
        {11, 21, 17, 5},
        {12, 2, 6, 18},
        {13, 3, 19, 7},
        {14, 8, 0, 20},
        {15, 9, 21, 1},
        {16, 22, 5, 10},
        {17, 23, 11, 4},
        {18, 4, 7, 12},
        {19, 5, 13, 6},
        {20, 10, 1, 14},
        {21, 11, 15, 0},
        {22, 16, 3, 8},
        {23, 17, 9, 2},
    };
    const int64_t map2[24][3] = {
        {0, 2, 5},
        {1, 4, 3},
        {2, 0, 4},
        {3, 5, 1},
        {4, 1, 2},
        {5, 3, 0},
        {6, 8, 11},
        {7, 10, 9},
        {8, 6, 10},
        {9, 11, 7},
        {10, 7, 8},
        {11, 9, 6},
        {12, 14, 17},
        {13, 16, 15},
        {14, 12, 16},
        {15, 17, 13},
        {16, 13, 14},
        {17, 15, 12},
        {18, 20, 23},
        {19, 22, 21},
        {20, 18, 22},
        {21, 23, 19},
        {22, 19, 20},
        {23, 21, 18},
    };
    const int64_t map3[24][2] = {
        {0, 1},
        {1, 0},
        {2, 3},
        {3, 2},
        {4, 5},
        {5, 4},
        {6, 7},
        {7, 6},
        {8, 9},
        {9, 8},
        {10, 11},
        {11, 10},
        {12, 13},
        {13, 12},
        {14, 15},
        {15, 14},
        {16, 17},
        {17, 16},
        {18, 19},
        {19, 18},
        {20, 21},
        {21, 20},
        {22, 23},
        {23, 22},
    };

#pragma omp parallel for schedule(static)
    for (int64_t idx = 0; idx < ntriplets; idx++)
    {
        int64_t i, j, k, l = 0;
        int64_t tmp = idx;

        // Find i
        for (i = 0; i < nocc; i++)
        {
            // Number of (j,k,l) combinations for this i
            int64_t count_i = (int64_t)(i + 1) * (i + 2) * (i + 3) / 6;
            if (tmp < count_i)
                break;
            tmp -= count_i;
        }

        // Find j (given i)
        for (j = 0; j <= i; j++)
        {
            // Number of (k,l) combinations for this j
            int64_t count_j = (int64_t)(j + 1) * (j + 2) / 2;
            if (tmp < count_j)
                break;
            tmp -= count_j;
        }

        // Find k (given j)
        for (k = 0; k <= j; k++)
        {
            // Number of l values for this k
            int64_t count_k = k + 1;
            if (tmp < count_k)
            {
                l = (int)tmp;
                break;
            }
            tmp -= count_k;
        }

        int64_t occ_perms[24][4] = {
            {i, j, k, l},
            {i, j, l, k},
            {i, k, j, l},
            {i, k, l, j},
            {i, l, j, k},
            {i, l, k, j},
            {j, i, k, l},
            {j, i, l, k},
            {j, k, i, l},
            {j, k, l, i},
            {j, l, i, k},
            {j, l, k, i},
            {k, i, j, l},
            {k, i, l, j},
            {k, j, i, l},
            {k, j, l, i},
            {k, l, i, j},
            {k, l, j, i},
            {l, i, j, k},
            {l, i, k, j},
            {l, j, i, k},
            {l, j, k, i},
            {l, k, i, j},
            {l, k, j, i},
        };

        double T1_local[24][24];
        double T2_local[24][24];
        for (int64_t a0 = 0; a0 < nvir; a0 += bl)
        {
            for (int64_t b0 = 0; b0 <= a0; b0 += bl)
            {
                for (int64_t c0 = 0; c0 <= b0; c0 += bl)
                {
                    for (int64_t d0 = 0; d0 <= c0; d0 += bl)
                    {
                        for (int64_t a = a0; a < a0 + bl && a < nvir; a++)
                        {
                            for (int64_t b = b0; b < b0 + bl && b <= a; b++)
                            {
                                for (int64_t c = c0; c < c0 + bl && c <= b; c++)
                                {
                                    for (int64_t d = d0; d < d0 + bl && d <= c; d++)
                                    {
                                        int64_t vir_perms[24][4] = {
                                            {a, b, c, d},
                                            {a, b, d, c},
                                            {a, c, b, d},
                                            {a, c, d, b},
                                            {a, d, b, c},
                                            {a, d, c, b},
                                            {b, a, c, d},
                                            {b, a, d, c},
                                            {b, c, a, d},
                                            {b, c, d, a},
                                            {b, d, a, c},
                                            {b, d, c, a},
                                            {c, a, b, d},
                                            {c, a, d, b},
                                            {c, b, a, d},
                                            {c, b, d, a},
                                            {c, d, a, b},
                                            {c, d, b, a},
                                            {d, a, b, c},
                                            {d, a, c, b},
                                            {d, b, a, c},
                                            {d, b, c, a},
                                            {d, c, a, b},
                                            {d, c, b, a},
                                        };

                                        int64_t indices[24][24];
                                        for (int perm_occ = 0; perm_occ < 24; perm_occ++)
                                        {
                                            for (int perm_vir = 0; perm_vir < 24; perm_vir++)
                                            {
                                                indices[perm_occ][perm_vir] =
                                                    occ_perms[perm_occ][0] * nooovvvv +
                                                    occ_perms[perm_occ][1] * noovvvv +
                                                    occ_perms[perm_occ][2] * novvvv +
                                                    occ_perms[perm_occ][3] * nvvvv +
                                                    vir_perms[perm_vir][0] * nvvv +
                                                    vir_perms[perm_vir][1] * nvv +
                                                    vir_perms[perm_vir][2] * nvir +
                                                    vir_perms[perm_vir][3];
                                            }
                                        }

                                        // (1 + P_ia^jb + P_ia^kc + P_ia^ld)
                                        for (int perm_occ = 0; perm_occ < 24; perm_occ++)
                                        {
                                            for (int perm_vir = 0; perm_vir < 24; perm_vir++)
                                            {
                                                T1_local[perm_occ][perm_vir] = A[indices[map1[perm_occ][0]][map1[perm_vir][0]]] + A[indices[map1[perm_occ][1]][map1[perm_vir][1]]] + A[indices[map1[perm_occ][2]][map1[perm_vir][2]]] + A[indices[map1[perm_occ][3]][map1[perm_vir][3]]];
                                            }
                                        }

                                        // (1 + P_jb^kc + P_jb^ld)
                                        // ijk___ = ijk___ + jik___
                                        for (int perm_occ = 0; perm_occ < 24; perm_occ++)
                                        {
                                            for (int perm_vir = 0; perm_vir < 24; perm_vir++)
                                            {
                                                T2_local[perm_occ][perm_vir] = T1_local[map2[perm_occ][0]][map2[perm_vir][0]] + T1_local[map2[perm_occ][1]][map2[perm_vir][1]] + T1_local[map2[perm_occ][2]][map2[perm_vir][2]];
                                            }
                                        }
                                        // (1 + P_kc^jd)
                                        for (int perm_occ = 0; perm_occ < 24; perm_occ++)
                                        {
                                            for (int perm_vir = 0; perm_vir < 24; perm_vir++)
                                            {
                                                A[indices[map3[perm_occ][0]][map3[perm_vir][0]]] = beta * A[indices[map3[perm_occ][0]][map3[perm_vir][0]]] + alpha * (T2_local[map3[perm_occ][0]][map3[perm_vir][0]] + T2_local[map3[perm_occ][1]][map3[perm_vir][1]]);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void eijkl_division_c(double *r4, const double *eia, const int64_t nocc, const int64_t nvir)
{
// Parallelize over the outermost loops for better load balancing
#pragma omp parallel for collapse(4) schedule(static)
    for (int64_t i = 0; i < nocc; i++)
    {
        for (int64_t j = 0; j < nocc; j++)
        {
            for (int64_t k = 0; k < nocc; k++)
            {
                for (int64_t l = 0; l < nocc; l++)
                {
                    size_t base_size = ((((size_t)i * nocc + j) * nocc + k) * nocc + l) * nvir * nvir * nvir * nvir;
                    for (int64_t a = 0; a < nvir; a++)
                    {
                        for (int64_t b = 0; b < nvir; b++)
                        {
                            for (int64_t c = 0; c < nvir; c++)
                            {
                                for (int64_t d = 0; d < nvir; d++)
                                {
                                    size_t r4_idx = (size_t)base_size + ((a * nvir + b) * nvir + c) * nvir + d;

                                    // Calculate eia indices (assuming eia is stored as [row][col])
                                    size_t eia_ia_idx = (size_t)i * nvir + a; // eia[i, a]
                                    size_t eia_jb_idx = (size_t)j * nvir + b; // eia[j, b]
                                    size_t eia_kc_idx = (size_t)k * nvir + c; // eia[k, c]
                                    size_t eia_ld_idx = (size_t)l * nvir + d; // eia[l, d]

                                    // Compute the denominator (broadcasted sum)
                                    double eijklabcd = eia[eia_ia_idx] + eia[eia_jb_idx] +
                                                       eia[eia_kc_idx] + eia[eia_ld_idx];

                                    // Perform division (with zero-division protection)
                                    if (fabs(eijklabcd) > 1e-15)
                                    {
                                        r4[r4_idx] /= eijklabcd;
                                    }
                                    else
                                    {
                                        // Handle division by zero - set to zero or keep original value
                                        r4[r4_idx] = 0.0; // or use a different strategy
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void t4_add_c(double *t4, const double *r4, const int64_t nocc4, const int64_t nvir)
{
    const int64_t total_size = nocc4 * nvir * nvir * nvir * nvir;

#pragma omp parallel for schedule(static, 1024)
    for (int64_t i = 0; i < total_size; i++)
    {
        t4[i] += r4[i];
    }
}

const int64_t tp_t4[24][4] = {
    {0, 1, 2, 3},
    {0, 1, 3, 2},
    {0, 2, 1, 3},
    {0, 2, 3, 1},
    {0, 3, 1, 2},
    {0, 3, 2, 1},
    {1, 0, 2, 3},
    {1, 0, 3, 2},
    {1, 2, 0, 3},
    {1, 2, 3, 0},
    {1, 3, 0, 2},
    {1, 3, 2, 0},
    {2, 0, 1, 3},
    {2, 0, 3, 1},
    {2, 1, 0, 3},
    {2, 1, 3, 0},
    {2, 3, 0, 1},
    {2, 3, 1, 0},
    {3, 0, 1, 2},
    {3, 0, 2, 1},
    {3, 1, 0, 2},
    {3, 1, 2, 0},
    {3, 2, 0, 1},
    {3, 2, 1, 0},
};

void unpack_24fold_c(const double *restrict t4_tril,
                     double *restrict t4_blk,
                     const int64_t *restrict map,
                     const bool *restrict mask,
                     int64_t i0, int64_t i1,
                     int64_t j0, int64_t j1,
                     int64_t k0, int64_t k1,
                     int64_t l0, int64_t l1,
                     int64_t nocc, int64_t nvir,
                     int64_t blk_i, int64_t blk_j, int64_t blk_k, int64_t blk_l)
{
#define MAP(sym, w, x, y, z) map[((((sym) * nocc + (w)) * nocc + (x)) * nocc + (y)) * nocc + (z)]
#define MASK(sym, w, x, y, z) mask[((((sym) * nocc + (w)) * nocc + (x)) * nocc + (y)) * nocc + (z)]

#pragma omp parallel for collapse(5) schedule(static)
    for (int64_t sym = 0; sym < 24; ++sym)
    {
        for (int64_t i = i0; i < i1; ++i)
        {
            for (int64_t j = j0; j < j1; ++j)
            {
                for (int64_t k = k0; k < k1; ++k)
                {
                    for (int64_t l = l0; l < l1; ++l)
                    {
                        if (!MASK(sym, i, j, k, l))
                            continue;

                        const int64_t *perm = tp_t4[sym];

                        int64_t loc_i = i - i0;
                        int64_t loc_j = j - j0;
                        int64_t loc_k = k - k0;
                        int64_t loc_l = l - l0;

                        int64_t src_base = MAP(sym, i, j, k, l) * nvir * nvir * nvir * nvir;
                        int64_t dest_base = (((loc_i * blk_j + loc_j) * blk_k + loc_k) * blk_l + loc_l) * nvir * nvir * nvir * nvir;

                        for (int64_t a = 0; a < nvir; ++a)
                        {
                            for (int64_t b = 0; b < nvir; ++b)
                            {
                                for (int64_t c = 0; c < nvir; ++c)
                                {
                                    for (int64_t d = 0; d < nvir; ++d)
                                    {
                                        int64_t abcd[4] = {a, b, c, d};
                                        int64_t aa = abcd[perm[0]];
                                        int64_t bb = abcd[perm[1]];
                                        int64_t cc = abcd[perm[2]];
                                        int64_t dd = abcd[perm[3]];

                                        int64_t src_idx = src_base + ((a * nvir + b) * nvir + c) * nvir + d;
                                        int64_t dest_idx = dest_base + ((aa * nvir + bb) * nvir + cc) * nvir + dd;

                                        t4_blk[dest_idx] = t4_tril[src_idx];
                                    }
                                }
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

void update_packed_24fold_c(double *restrict t4_tril,
                            const double *restrict t4_blk,
                            const int64_t *restrict map,
                            int64_t i0, int64_t i1,
                            int64_t j0, int64_t j1,
                            int64_t k0, int64_t k1,
                            int64_t l0, int64_t l1,
                            int64_t nocc, int64_t nvir,
                            int64_t blk_i, int64_t blk_j, int64_t blk_k, int64_t blk_l,
                            double alpha, double beta)
{
#define MAP(sym, w, x, y, z) map[((((sym) * nocc + (w)) * nocc + (x)) * nocc + (y)) * nocc + (z)]

    if (j1 < i0 || k1 < j0 || l1 < k0)
        return;

#pragma omp parallel for collapse(4)
    for (int64_t l = l0; l < l1; ++l)
    {
        for (int64_t k = k0; k < k1; ++k)
        {
            for (int64_t j = j0; j < j1; ++j)
            {
                for (int64_t i = i0; i < i1; ++i)
                {
                    if (k > l || j > k || i > j)
                        continue;

                    int64_t p = MAP(0, i, j, k, l);
                    int64_t tril_base = p * nvir * nvir * nvir * nvir;

                    int64_t loc_i = i - i0;
                    int64_t loc_j = j - j0;
                    int64_t loc_k = k - k0;
                    int64_t loc_l = l - l0;
                    int64_t blk_base = (((loc_i * blk_j + loc_j) * blk_k + loc_k) * blk_l + loc_l) * nvir * nvir * nvir * nvir;

                    for (int64_t a = 0; a < nvir; ++a)
                    {
                        for (int64_t b = 0; b < nvir; ++b)
                        {
                            for (int64_t c = 0; c < nvir; ++c)
                            {
                                for (int64_t d = 0; d < nvir; ++d)
                                {
                                    int64_t idx = (((a * nvir + b) * nvir + c) * nvir + d);
                                    t4_tril[tril_base + idx] = beta * t4_tril[tril_base + idx] + alpha * t4_blk[blk_base + idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
#undef MAP
}
