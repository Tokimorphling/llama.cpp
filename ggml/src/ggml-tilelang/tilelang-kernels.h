#pragma once

#include <stdint.h>

#ifdef  __cplusplus
extern "C" {
#endif

void tilelang_w8a8_quant_x(
    const float * x_f32,
    const float * smooth_scale,
    int8_t * x_q,
    float * x_scale,
    int K,
    void * stream
);

void tilelang_w8a8_dot(
    const int8_t * w_q,
    const float * w_scale,
    const int8_t * x_q,
    const float * x_scale,
    float * y_f32,
    int K,
    int N,
    void * stream
);

void tilelang_w8a8_gemv(
    const int8_t * w_q,
    const float * w_scale,
    const float * smooth_scale,
    const float * x_f32,
    float * y_f32,
    int8_t * x_q_scratch,
    float * x_scale_scratch,
    int K,
    int N,
    void * stream
);

void tilelang_w8a8_quant_x_rows(
    const float * x_f32,
    const float * smooth_scale,
    int8_t * x_q,
    float * x_scale,
    int M,
    int K,
    void * stream
);

void tilelang_w8a8_quant_from_partial_max_rows(
    const float * x_scaled_f32,
    const float * partial_max_f32,
    int8_t * x_q,
    float * x_scale,
    int M,
    int K,
    int P,
    void * stream
);

void tilelang_w8a8_silu_mul_quant_rows(
    const float * gate_f32,
    const float * up_f32,
    const float * smooth_scale,
    int8_t * x_q,
    float * x_scale,
    int M,
    int K,
    void * stream
);

void tilelang_w8a8_gemm_dot(
    const int8_t * w_q,
    const float * w_scale,
    const int8_t * x_q,
    const float * x_scale,
    float * y_f32,
    int M,
    int K,
    int N,
    void * stream
);

int tilelang_w8a8_gate_up_block_n(void);

void tilelang_w8a8_gate_up_silu_scaled_gemm_dot_partial_max(
    const int8_t * w_up_q,
    const float * w_up_scale,
    const int8_t * w_gate_q,
    const float * w_gate_scale,
    const int8_t * x_q,
    const float * x_scale,
    const float * smooth_scale,
    float * y_scaled_f32,
    float * partial_max_f32,
    int M,
    int K,
    int N,
    void * stream
);

void tilelang_w8a8_gemm(
    const int8_t * w_q,
    const float * w_scale,
    const float * smooth_scale,
    const float * x_f32,
    float * y_f32,
    int8_t * x_q_scratch,
    float * x_scale_scratch,
    int M,
    int K,
    int N,
    void * stream
);

#ifdef  __cplusplus
}
#endif
