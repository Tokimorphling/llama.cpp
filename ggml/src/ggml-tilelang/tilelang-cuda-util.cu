#include "tilelang-kernels.h"

#include <cuda_runtime.h>

extern "C" void tilelang_cuda_synchronize(void * stream) {
    cudaStream_t cu_stream = (cudaStream_t) stream;
    if (cu_stream != nullptr) {
        cudaStreamSynchronize(cu_stream);
    } else {
        cudaDeviceSynchronize();
    }
}

extern "C" void * tilelang_cuda_event_create(void) {
    cudaEvent_t event = nullptr;
    cudaEventCreate(&event);
    return event;
}

extern "C" void tilelang_cuda_event_destroy(void * event) {
    if (event != nullptr) {
        cudaEventDestroy((cudaEvent_t) event);
    }
}

extern "C" void tilelang_cuda_event_record(void * event, void * stream) {
    if (event == nullptr) {
        return;
    }

    cudaEventRecord((cudaEvent_t) event, (cudaStream_t) stream);
}

extern "C" float tilelang_cuda_event_elapsed_ms(void * start, void * stop) {
    if (start == nullptr || stop == nullptr) {
        return 0.0f;
    }

    float ms = 0.0f;
    cudaEventSynchronize((cudaEvent_t) stop);
    cudaEventElapsedTime(&ms, (cudaEvent_t) start, (cudaEvent_t) stop);
    return ms;
}
