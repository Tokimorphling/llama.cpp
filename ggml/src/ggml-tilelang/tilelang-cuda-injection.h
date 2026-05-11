#pragma once

#include "ggml.h"

bool ggml_tilelang_cuda_supports_op(const ggml_tensor * op);

bool ggml_tilelang_cuda_try_custom(
        int device,
        void * stream,
        ggml_tensor * dst);

bool ggml_tilelang_cuda_preload_device(int device);

void ggml_tilelang_cuda_release_device(int device);
