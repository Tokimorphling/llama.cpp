#pragma once

#ifdef  __cplusplus
extern "C" {
#endif

void tilelang_f16_gemv(
    const void * w_f16,
    const float * x_f32,
    float * y_f32,
    int K,
    int N,
    void * stream
);

void tilelang_q8_0_gemv(
    const void * w_q8_0,
    const float * x_f32,
    float * y_f32,
    int K,
    int N,
    void * stream
);

void tilelang_cuda_synchronize(void * stream);
void * tilelang_cuda_event_create(void);
void tilelang_cuda_event_destroy(void * event);
void tilelang_cuda_event_record(void * event, void * stream);
float tilelang_cuda_event_elapsed_ms(void * start, void * stop);

#ifdef  __cplusplus
}
#endif
