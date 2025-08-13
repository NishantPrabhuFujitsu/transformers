/*******************************************************************************
* Copyright 2025 FUJITSU LIMITED
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <iostream>
#include <thread>
#include <arm_sve.h>  
#include <omp.h> 
#include <fstream>
#include <cmath>


size_t get_sve_vector_length() {
    return svcntw();
}

/* Below function was borrowed from the GitHub repository: 
https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu/src/nodes/kernels/scaled_attn/common.hpp */
svfloat32_t exp_ps_sve(svbool_t& pg, svfloat32_t& src) {
    // Constants
    const auto log2_e = svdup_n_f32(1.4426950409f);
    const auto ln2 = svdup_n_f32(0.6931473921f);
    const auto half_ln2_sq = svdup_n_f32(0.2413862043f);
    const auto not_mask17 = svdup_n_u32(~((1u << 17) - 1));
    const auto one = svdup_n_f32(1.0f);
    const svfloat32_t inactive1 = svdup_n_f32(0.0f);
    const svint32_t inactive2 = svdup_n_s32(0);

    // Algorithm starts here
    svfloat32_t t0 = svmul_f32_m(pg, src, log2_e);  // y = x * log2(e)
    svfloat32_t t1 = svrintm_f32_m(inactive1, pg, t0);         // rount to int (float)
    svint32_t t2 = svcvt_s32_f32_m(inactive2, pg, t1);         // n

    t1 = svsub_f32_m(pg, t0, t1);   // a = y - floor(y)
    t1 = svadd_f32_m(pg, t1, one);  // b = a + 1

    svuint32_t t3 = svlsr_n_u32_m(pg, svreinterpret_u32_f32(t1), 17);  // v = b >> 17 (u32)
    svfloat32_t t4 = svexpa_f32(t3);                                   // c = fexpa(v)
    t4 = svscale_f32_m(pg, t4, t2);                                    // fexpa(v) * 2^(n)

    // and_(t2.d, t1.d, not_mask17.d)
    svfloat32_t t5 = svreinterpret_f32_u32(svand_u32_m(pg, svreinterpret_u32_f32(t1), not_mask17));
    t5 = svsub_f32_m(pg, t1, t5);                // z
    t0 = svmla_f32_m(pg, ln2, t5, half_ln2_sq);  // ln2 + half_ln2_sq * z
    t0 = svmla_f32_m(pg, one, t5, t0);           // 1 + (ln2 * z) + (half_ln2_sq * z * z)
    t0 = svmul_f32_m(pg, t0, t4);                // Final result

    return t0;
}

// void scan_sve_impl(float* A, 
//                    float* B, 
//                    float* M, 
//                    float* hidden_states, 
//                    float* discrete_time_step, 
//                    float* ssm_state, 
//                    float* scan_output, 
//                    int B_size, 
//                    int D_size, 
//                    int L_size) 
// {
//     unsigned int total_cores = std::thread::hardware_concurrency();
//     #pragma omp parallel for schedule(static,int(B_size*D_size/total_cores)) collapse(2)

//     for (int i = 0; i < B_size; i++) {
//         for (int j = 0; j < D_size; j += svcntw()) {
//             svbool_t pg = svwhilelt_b32(j, D_size); //decleration of predicate
//             //---load_variables---
//             svfloat32_t v_A = svld1_f32(pg, &A[j]);//load A: [D_size] 
//             svfloat32_t v_ssm = svld1_f32(pg, &ssm_state[i * D_size + j]); //load ssm_state: [B_size, D_size]
//             //------
//             for (int l = 0; l < L_size; l++) {
//                 int index = l*B_size*D_size + i*D_size + j; // index for tensors of shapre [L_size, B_size, D_size]
//                 int index_B = l* B_size + i; // index for tensors of shapre [L_size, B_size]
//                 //---load_variables---
//                 svfloat32_t v_dt = svld1_f32(pg, &discrete_time_step[index]); //load  discrete_time_step: [L_size, B_size, D_size]
//                 svfloat32_t v_hs = svld1_f32(pg, &hidden_states[index]); //load hidden_state: [L_size, B_size, D_size]
//                 svfloat32_t v_B = svdup_n_f32(B[index_B]); //load and broadcast value, B: [L_size,B_size]  
//                 svfloat32_t v_M = svld1_f32(pg, &M[index]); //load M: [L_size,B_size, D_size] 
//                 //---construct discrete_A---
//                 svfloat32_t v_dA = svmul_f32_m(pg, v_dt, v_A); // v_dt*v_A
//                 v_dA = exp_ps_sve(pg, v_dA); //v_dA = exp(v_dA)

//                 //---construct discrete_B---
//                 svfloat32_t v_dB = svmad_f32_m(pg, v_B, v_dt, v_M); // v_B*v_dt + v_M
//                 svfloat32_t v_ddB = svmul_f32_m(pg, v_dB, v_hs); // v_dB*v_hs
//                 //------
//                 //---main iterate---
//                 v_ssm = svmad_f32_m(pg, v_dA, v_ssm, v_ddB); //iterate
//                 //------
//                 svst1_f32(pg, &scan_output[index], v_ssm); //store v_ssm
//             }
//         } 
//     }
// }

void scan_sve_impl(float* A, 
                   float* B, 
                   float* E, 
                   float* hidden_states, 
                   float* discrete_time_step, 
                   float* ssm_state, 
                   float* scan_output, 
                   int B_size, 
                   int D_size, 
                   int L_size) 
{
    unsigned int total_cores = std::thread::hardware_concurrency();
    #pragma omp parallel for schedule(static,int(B_size*D_size/total_cores)) collapse(2)

    for (int i = 0; i < B_size; i++) {
        for (int j = 0; j < D_size; j += svcntw()) {
            svbool_t pg = svwhilelt_b32(j, D_size);                             // predicate
            //---load_variables---
            svfloat32_t v_A = svld1_f32(pg, &A[j]);                             // load A: [D_size] 
            svfloat32_t v_E = svld1_f32(pg, &E[j]);                             // load E: [D_size]
            svfloat32_t v_ssm = svld1_f32(pg, &ssm_state[i * D_size + j]);      // load ssm_state: [B_size, D_size]
            //------
            for (int l = 0; l < L_size; l++) {
                int index = l*B_size*D_size + i*D_size + j;                     // index for tensors of shapre [L_size, B_size, D_size]
                int index_n = (l+1)*B_size*D_size + i*D_size + j;               // index for tensors of shapre [L_size, B_size, D_size]
                int index_B = l* B_size + i;                                    // index for tensors of shapre [L_size, B_size]
                //---load_variables---
                svfloat32_t v_dt = svld1_f32(pg, &discrete_time_step[index]);   // load  discrete_time_step: [L_size, B_size, D_size]
                svfloat32_t v_hs = svld1_f32(pg, &hidden_states[index_n]);      // load hidden_state: [L_size, B_size, D_size]
                svfloat32_t v_B = svdup_n_f32(B[index_B]);                      // load and broadcast value, B: [L_size,B_size]  
                //---construct discrete_A---
                svfloat32_t v_dA = svmul_f32_m(pg, v_dt, v_A);                  // v_dt * v_A
                v_dA = exp_ps_sve(pg, v_dA);                                    // v_dA = exp(v_dA)
                //---construct discrete_B---
                svfloat32_t v_x = svld1_f32(pg, &hidden_states[index_n]);
                svfloat32_t v_x_prev = svld1_f32(pg, &hidden_states[index]);
                svfloat32_t v_W = svsub_f32_m(pg, v_x, v_x_prev);
                svfloat32_t v_M = svmul_f32_m(pg, v_E, v_W);                    // v_E * v_W
                svfloat32_t v_dB = svmad_f32_m(pg, v_B, v_dt, v_M);             // v_B * v_dt + v_M
                svfloat32_t v_ddB = svmul_f32_m(pg, v_dB, v_hs);                // v_dB*v_hs
                //---main iterate---
                v_ssm = svmad_f32_m(pg, v_dA, v_ssm, v_ddB);                    // iterate
                //------
                svst1_f32(pg, &scan_output[index], v_ssm);                      // store v_ssm
            }
        } 
    }
}