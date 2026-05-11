#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

//
// Backend
//

GGML_API ggml_backend_t ggml_backend_tilelang_init(int device);

GGML_API bool ggml_backend_is_tilelang(ggml_backend_t backend);

//
// Registry
//

GGML_API ggml_backend_reg_t ggml_backend_tilelang_reg(void);

#ifdef  __cplusplus
}
#endif
