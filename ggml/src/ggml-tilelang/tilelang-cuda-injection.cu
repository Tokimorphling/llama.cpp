#include "tilelang-cuda-injection.h"
#include "tilelang-kernels.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "gguf.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

static constexpr int TILELANG_CUDA_MAX_DEVICES = 16;

static bool tilelang_cuda_valid_device(int device) {
    return device >= 0 && device < TILELANG_CUDA_MAX_DEVICES;
}

static bool tilelang_cuda_check(cudaError_t err, const char * what) {
    if (err == cudaSuccess) {
        return true;
    }
    GGML_LOG_WARN("ggml-tilelang: %s failed: %s\n", what, cudaGetErrorString(err));
    (void) cudaGetLastError();
    return false;
}

static void tilelang_cuda_set_device(int device) {
    int current = -1;
    if (cudaGetDevice(&current) == cudaSuccess && current == device) {
        return;
    }
    (void) tilelang_cuda_check(cudaSetDevice(device), "cudaSetDevice");
}

static bool tilelang_cuda_tensor_on_device(const ggml_tensor * tensor, int device) {
    if (tensor == nullptr || tensor->buffer == nullptr) {
        return false;
    }
    return ggml_backend_cuda_get_buffer_type_device(ggml_backend_buffer_get_type(tensor->buffer)) == device;
}

static std::atomic<size_t> g_cuda_tilelang_w8a8_qwen35_ffn_calls;

static bool tilelang_cuda_w8a8_env_enabled(const char * name) {
    const char * env = std::getenv(name);
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

static bool tilelang_cuda_w8a8_enabled() {
    return tilelang_cuda_w8a8_env_enabled("GGML_TILELANG_W8A8_ENABLE");
}

static bool tilelang_cuda_w8a8_debug_enabled() {
    return tilelang_cuda_w8a8_env_enabled("GGML_TILELANG_W8A8_DEBUG");
}

static bool tilelang_cuda_w8a8_preload_enabled() {
    return tilelang_cuda_w8a8_env_enabled("GGML_TILELANG_W8A8_PRELOAD");
}

struct tilelang_cuda_w8a8_tensor {
    std::string name;
    std::string smooth_group_id;
    int K = 0;
    int N = 0;
    size_t w_q_nbytes = 0;
    size_t w_scale_offset = 0;
    size_t w_scale_nbytes = 0;
    size_t smooth_offset = 0;
    size_t smooth_nbytes = 0;

    int8_t * d_w_q[TILELANG_CUDA_MAX_DEVICES] = {};
    float * d_w_scale[TILELANG_CUDA_MAX_DEVICES] = {};
    float * d_smooth[TILELANG_CUDA_MAX_DEVICES] = {};
};

struct tilelang_cuda_w8a8_scratch {
    int8_t * x_q = nullptr;
    float * x_scale = nullptr;
    float * tmp_a = nullptr;
    float * tmp_b = nullptr;
    float * partial_max = nullptr;
    size_t tmp_elements = 0;
    size_t partial_max_elements = 0;
    int K = 0;
    int M = 0;
};

struct tilelang_cuda_w8a8_state {
    std::mutex mutex;
    bool loaded = false;
    bool load_failed = false;
    std::string path;
    std::string source_format;
    std::string source_model;
    std::string recipe;
    std::string smooth;
    size_t total_blob_bytes = 0;
    float alpha = 0.0f;
    std::vector<tilelang_cuda_w8a8_tensor> tensors;
    std::unordered_map<std::string, size_t> by_name;
    tilelang_cuda_w8a8_scratch scratch[TILELANG_CUDA_MAX_DEVICES];
    bool preload_done[TILELANG_CUDA_MAX_DEVICES] = {};
    size_t uploaded_tensors[TILELANG_CUDA_MAX_DEVICES] = {};
    size_t uploaded_bytes[TILELANG_CUDA_MAX_DEVICES] = {};
    size_t reused_w_q_tensors[TILELANG_CUDA_MAX_DEVICES] = {};
    size_t reused_w_q_bytes[TILELANG_CUDA_MAX_DEVICES] = {};
};

static bool tilelang_cuda_qwen35_ffn_op(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_CUSTOM) {
        return false;
    }

    struct ggml_custom_op_params params;
    std::memcpy(&params, op->op_params, sizeof(params));

    static constexpr uintptr_t QWEN35_TILELANG_FFN_MAGIC = 0x51333546u;
    return reinterpret_cast<uintptr_t>(params.userdata) == QWEN35_TILELANG_FFN_MAGIC;
}

static tilelang_cuda_w8a8_state & tilelang_cuda_w8a8_state_get() {
    static tilelang_cuda_w8a8_state state;
    return state;
}

static bool tilelang_cuda_w8a8_read_blob_locked(
        const tilelang_cuda_w8a8_state & state,
        size_t offset,
        size_t nbytes,
        std::vector<uint8_t> & out) {
    std::ifstream in(state.path, std::ios::binary);
    if (!in) {
        return false;
    }

    in.seekg((std::streamoff) offset, std::ios::beg);
    out.resize(nbytes);
    in.read((char *) out.data(), (std::streamsize) nbytes);
    return in && (size_t) in.gcount() == nbytes;
}

struct tilelang_cuda_w8a8_gguf_deleter {
    void operator()(gguf_context * ctx) const {
        gguf_free(ctx);
    }
};

static bool tilelang_cuda_w8a8_get_gguf_tensor_blob(
        const gguf_context * gguf,
        const char * path,
        const char * tensor_name,
        enum ggml_type expected_type,
        size_t expected_nbytes,
        size_t * offset,
        size_t & nbytes) {
    const int64_t tensor_id = gguf_find_tensor(gguf, tensor_name);
    if (tensor_id < 0) {
        GGML_LOG_WARN("ggml-tilelang: W8A8 GGUF tensor missing: %s in %s\n", tensor_name, path);
        return false;
    }

    const enum ggml_type type = gguf_get_tensor_type(gguf, tensor_id);
    if (type != expected_type) {
        GGML_LOG_WARN("ggml-tilelang: W8A8 GGUF tensor %s has type %s, expected %s\n",
            tensor_name, ggml_type_name(type), ggml_type_name(expected_type));
        return false;
    }

    nbytes = gguf_get_tensor_size(gguf, tensor_id);
    if (nbytes != expected_nbytes) {
        GGML_LOG_WARN("ggml-tilelang: W8A8 GGUF tensor %s has %zu bytes, expected %zu\n",
            tensor_name, nbytes, expected_nbytes);
        return false;
    }

    if (offset != nullptr) {
        *offset = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, tensor_id);
    }
    return true;
}

static bool tilelang_cuda_w8a8_load_gguf_tensors_locked(
        tilelang_cuda_w8a8_state & state,
        const char * path) {
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    std::unique_ptr<gguf_context, tilelang_cuda_w8a8_gguf_deleter> gguf(gguf_init_from_file(path, params));
    if (!gguf) {
        state.load_failed = true;
        GGML_LOG_WARN("ggml-tilelang: failed to open W8A8 GGUF: %s\n", path);
        return false;
    }

    const int version_key = gguf_find_key(gguf.get(), "tilelang.w8a8.version");
    const int manifest_key = gguf_find_key(gguf.get(), "tilelang.w8a8.manifest");
    if (version_key < 0 || manifest_key < 0) {
        state.load_failed = true;
        GGML_LOG_WARN("ggml-tilelang: W8A8 GGUF metadata missing in %s\n", path);
        return false;
    }

    const char * manifest_cstr = gguf_get_val_str(gguf.get(), manifest_key);
    if (manifest_cstr == nullptr || manifest_cstr[0] == '\0') {
        state.load_failed = true;
        GGML_LOG_WARN("ggml-tilelang: W8A8 GGUF manifest is empty in %s\n", path);
        return false;
    }

    std::string json_text(manifest_cstr);
    try {
        const nlohmann::ordered_json manifest = nlohmann::ordered_json::parse(json_text);
        state.path = path;
        state.source_format = manifest.value("source_format", std::string("gguf_w8a8_tensors"));
        if (state.source_format != "gguf_w8a8_replacement") {
            state.load_failed = true;
            GGML_LOG_WARN("ggml-tilelang: W8A8 GGUF is not a replacement model: %s source_format=%s\n",
                path, state.source_format.c_str());
            return false;
        }
        state.source_model = manifest.value("source_model", std::string(path));
        state.total_blob_bytes = 0;

        if (manifest.contains("quantization")) {
            const nlohmann::ordered_json & quant = manifest.at("quantization");
            state.recipe = quant.value("recipe", std::string("unknown"));
            state.smooth = quant.value("smooth", std::string("unknown"));
            state.alpha = quant.value("alpha", 0.0f);
        } else {
            state.recipe = "unknown";
            state.smooth = "unknown";
            state.alpha = 0.0f;
        }

        for (const nlohmann::ordered_json & item : manifest.at("tensors")) {
            tilelang_cuda_w8a8_tensor tensor;
            tensor.name = item.at("name").get<std::string>();
            tensor.smooth_group_id = item.value("smooth_group_id", std::string());
            tensor.K = item.at("K").get<int>();
            tensor.N = item.at("N").get<int>();

            const nlohmann::ordered_json & names = item.at("gguf_tensors");
            const std::string w_q_name = names.at("w_q").get<std::string>();
            const std::string w_scale_name = names.at("w_scale").get<std::string>();
            const std::string smooth_name = names.at("smooth_scale").get<std::string>();
            if (w_q_name != tensor.name) {
                state.load_failed = true;
                GGML_LOG_WARN("ggml-tilelang: W8A8 replacement tensor %s stores w_q as %s\n",
                    tensor.name.c_str(), w_q_name.c_str());
                return false;
            }

            if (!tilelang_cuda_w8a8_get_gguf_tensor_blob(
                    gguf.get(), path, w_q_name.c_str(), GGML_TYPE_I8,
                    (size_t) tensor.K * (size_t) tensor.N, nullptr, tensor.w_q_nbytes) ||
                !tilelang_cuda_w8a8_get_gguf_tensor_blob(
                    gguf.get(), path, w_scale_name.c_str(), GGML_TYPE_F32,
                    (size_t) tensor.N * sizeof(float), &tensor.w_scale_offset, tensor.w_scale_nbytes) ||
                !tilelang_cuda_w8a8_get_gguf_tensor_blob(
                    gguf.get(), path, smooth_name.c_str(), GGML_TYPE_F32,
                    (size_t) tensor.K * sizeof(float), &tensor.smooth_offset, tensor.smooth_nbytes)) {
                state.load_failed = true;
                return false;
            }

            state.total_blob_bytes += tensor.w_q_nbytes + tensor.w_scale_nbytes + tensor.smooth_nbytes;
            const size_t tensor_index = state.tensors.size();
            state.by_name.emplace(tensor.name, tensor_index);
            state.tensors.push_back(std::move(tensor));
        }
    } catch (const std::exception & e) {
        state.load_failed = true;
        GGML_LOG_WARN("ggml-tilelang: failed to parse W8A8 GGUF manifest %s: %s\n", path, e.what());
        return false;
    }

    state.loaded = true;
    GGML_LOG_INFO(
        "ggml-tilelang: TileLang W8A8 GGUF tensors loaded tensors=%zu recipe=%s smooth=%s alpha=%.3g "
        "path=%s total_blob_bytes=%zu\n",
        state.tensors.size(), state.recipe.c_str(), state.smooth.c_str(), (double) state.alpha, state.path.c_str(),
        state.total_blob_bytes);
    return true;
}

static bool tilelang_cuda_w8a8_load_locked(tilelang_cuda_w8a8_state & state) {
    if (state.loaded) {
        return true;
    }
    if (state.load_failed) {
        return false;
    }

    const char * gguf_env = std::getenv("GGML_TILELANG_W8A8_GGUF");
    if (gguf_env != nullptr && gguf_env[0] != '\0') {
        return tilelang_cuda_w8a8_load_gguf_tensors_locked(state, gguf_env);
    }

    state.load_failed = true;
    GGML_LOG_WARN("ggml-tilelang: GGML_TILELANG_W8A8_ENABLE=1 but GGML_TILELANG_W8A8_GGUF is not set\n");
    return false;
}

static bool tilelang_cuda_w8a8_upload_tensor_locked(
        tilelang_cuda_w8a8_state & state,
        tilelang_cuda_w8a8_tensor & tensor,
        int device,
        bool log_upload) {
    const bool need_w_scale = tensor.d_w_scale[device] == nullptr;
    const bool need_smooth = tensor.d_smooth[device] == nullptr;
    if (!need_w_scale && !need_smooth) {
        return true;
    }

    const size_t tensor_bytes =
        (need_w_scale ? tensor.w_scale_nbytes : 0) +
        (need_smooth ? tensor.smooth_nbytes : 0);

    std::vector<uint8_t> w_scale;
    std::vector<uint8_t> smooth;
    if ((need_w_scale && !tilelang_cuda_w8a8_read_blob_locked(state, tensor.w_scale_offset, tensor.w_scale_nbytes, w_scale)) ||
        (need_smooth && !tilelang_cuda_w8a8_read_blob_locked(state, tensor.smooth_offset, tensor.smooth_nbytes, smooth))) {
        GGML_LOG_WARN("ggml-tilelang: failed to read W8A8 GGUF tensor blobs for %s\n", tensor.name.c_str());
        return false;
    }

    tilelang_cuda_set_device(device);
    bool allocated_w_scale = false;
    bool allocated_smooth = false;
    cudaError_t err = cudaSuccess;
    if (err == cudaSuccess && need_w_scale) {
        err = cudaMalloc((void **) &tensor.d_w_scale[device], tensor.w_scale_nbytes);
        allocated_w_scale = err == cudaSuccess;
    }
    if (err == cudaSuccess && need_smooth) {
        err = cudaMalloc((void **) &tensor.d_smooth[device], tensor.smooth_nbytes);
        allocated_smooth = err == cudaSuccess;
    }
    if (err != cudaSuccess) {
        GGML_LOG_WARN("ggml-tilelang: W8A8 GGUF tensor upload allocation failed for %s on device %d: %s\n",
            tensor.name.c_str(), device, cudaGetErrorString(err));
        (void) cudaGetLastError();
        if (allocated_w_scale && tensor.d_w_scale[device] != nullptr) {
            (void) tilelang_cuda_check(cudaFree(tensor.d_w_scale[device]), "cudaFree");
            tensor.d_w_scale[device] = nullptr;
        }
        if (allocated_smooth && tensor.d_smooth[device] != nullptr) {
            (void) tilelang_cuda_check(cudaFree(tensor.d_smooth[device]), "cudaFree");
            tensor.d_smooth[device] = nullptr;
        }
        return false;
    }

    if ((need_w_scale && !tilelang_cuda_check(cudaMemcpy(tensor.d_w_scale[device], w_scale.data(), w_scale.size(), cudaMemcpyHostToDevice), "cudaMemcpy")) ||
        (need_smooth && !tilelang_cuda_check(cudaMemcpy(tensor.d_smooth[device], smooth.data(), smooth.size(), cudaMemcpyHostToDevice), "cudaMemcpy"))) {
        return false;
    }
    state.uploaded_tensors[device]++;
    state.uploaded_bytes[device] += tensor_bytes;
    if (log_upload || tilelang_cuda_w8a8_debug_enabled()) {
        GGML_LOG_INFO("ggml-tilelang: uploaded TileLang W8A8 tensor %s K=%d N=%d bytes=%zu uploaded=%zu\n",
            tensor.name.c_str(), tensor.K, tensor.N, tensor_bytes, state.uploaded_bytes[device]);
    }
    return true;
}

static bool tilelang_cuda_w8a8_bind_replacement_weight_locked(
        tilelang_cuda_w8a8_state & state,
        tilelang_cuda_w8a8_tensor & tensor,
        int device,
        const ggml_tensor * weight) {
    if (weight == nullptr ||
        weight->type != GGML_TYPE_I8 ||
        weight->data == nullptr ||
        weight->ne[0] != tensor.K ||
        weight->ne[1] != tensor.N ||
        weight->ne[2] != 1 ||
        weight->ne[3] != 1 ||
        ggml_nbytes(weight) != tensor.w_q_nbytes ||
        !tilelang_cuda_tensor_on_device(weight, device)) {
        GGML_LOG_WARN("ggml-tilelang: invalid replacement W8A8 tensor binding for %s on device %d\n",
            tensor.name.c_str(), device);
        return false;
    }

    int8_t * w_q = (int8_t *) weight->data;
    if (tensor.d_w_q[device] == w_q) {
        return true;
    }
    if (tensor.d_w_q[device] != nullptr) {
        GGML_LOG_WARN("ggml-tilelang: replacement W8A8 tensor %s changed CUDA pointer on device %d\n",
            tensor.name.c_str(), device);
        return false;
    }

    tensor.d_w_q[device] = w_q;
    state.reused_w_q_tensors[device]++;
    state.reused_w_q_bytes[device] += tensor.w_q_nbytes;
    if (tilelang_cuda_w8a8_debug_enabled()) {
        GGML_LOG_INFO("ggml-tilelang: bound replacement W8A8 tensor %s K=%d N=%d bytes=%zu reused=%zu\n",
            tensor.name.c_str(), tensor.K, tensor.N, tensor.w_q_nbytes, state.reused_w_q_bytes[device]);
    }
    return true;
}

static bool tilelang_cuda_w8a8_ensure_scratch_locked(
        tilelang_cuda_w8a8_state & state,
        int device,
        int K,
        int M) {
    tilelang_cuda_w8a8_scratch & scratch = state.scratch[device];
    if (scratch.x_q != nullptr && scratch.K >= K && scratch.M >= M) {
        return true;
    }

    tilelang_cuda_set_device(device);
    if (scratch.x_q != nullptr) {
        (void) tilelang_cuda_check(cudaFree(scratch.x_q), "cudaFree");
        scratch.x_q = nullptr;
    }
    if (scratch.x_scale != nullptr) {
        (void) tilelang_cuda_check(cudaFree(scratch.x_scale), "cudaFree");
        scratch.x_scale = nullptr;
    }

    const size_t x_q_nbytes = (size_t) M * (size_t) K;
    const size_t x_scale_nbytes = (size_t) M * sizeof(float);
    cudaError_t err = cudaMalloc((void **) &scratch.x_q, x_q_nbytes);
    if (err == cudaSuccess) {
        err = cudaMalloc((void **) &scratch.x_scale, x_scale_nbytes);
    }
    if (err != cudaSuccess) {
        GGML_LOG_WARN("ggml-tilelang: W8A8 scratch allocation failed on device %d M=%d K=%d: %s\n",
            device, M, K, cudaGetErrorString(err));
        (void) cudaGetLastError();
        if (scratch.x_q != nullptr) {
            (void) tilelang_cuda_check(cudaFree(scratch.x_q), "cudaFree");
            scratch.x_q = nullptr;
        }
        if (scratch.x_scale != nullptr) {
            (void) tilelang_cuda_check(cudaFree(scratch.x_scale), "cudaFree");
            scratch.x_scale = nullptr;
        }
        scratch.K = 0;
        scratch.M = 0;
        return false;
    }

    scratch.K = K;
    scratch.M = M;
    return true;
}

static bool tilelang_cuda_w8a8_ensure_ffn_scratch_locked(
        tilelang_cuda_w8a8_state & state,
        int device,
        int K,
        int M,
        int Nff) {
    if (!tilelang_cuda_w8a8_ensure_scratch_locked(state, device, std::max(K, Nff), M)) {
        return false;
    }

    tilelang_cuda_w8a8_scratch & scratch = state.scratch[device];
    const size_t tmp_elements = (size_t) M * (size_t) Nff;
    if (scratch.tmp_a != nullptr && scratch.tmp_b != nullptr && scratch.tmp_elements >= tmp_elements) {
        const int gate_up_block_n = std::max(1, tilelang_w8a8_gate_up_block_n());
        const int partial_cols = (Nff + gate_up_block_n - 1) / gate_up_block_n;
        const size_t partial_elements = (size_t) M * (size_t) partial_cols;
        if (scratch.partial_max != nullptr && scratch.partial_max_elements >= partial_elements) {
            return true;
        }
    }

    tilelang_cuda_set_device(device);
    if (scratch.tmp_a != nullptr) {
        (void) tilelang_cuda_check(cudaFree(scratch.tmp_a), "cudaFree");
        scratch.tmp_a = nullptr;
    }
    if (scratch.tmp_b != nullptr) {
        (void) tilelang_cuda_check(cudaFree(scratch.tmp_b), "cudaFree");
        scratch.tmp_b = nullptr;
    }
    if (scratch.partial_max != nullptr) {
        (void) tilelang_cuda_check(cudaFree(scratch.partial_max), "cudaFree");
        scratch.partial_max = nullptr;
    }
    scratch.tmp_elements = 0;
    scratch.partial_max_elements = 0;

    cudaError_t err = cudaMalloc((void **) &scratch.tmp_a, tmp_elements * sizeof(float));
    if (err == cudaSuccess) {
        err = cudaMalloc((void **) &scratch.tmp_b, tmp_elements * sizeof(float));
    }
    const int gate_up_block_n = std::max(1, tilelang_w8a8_gate_up_block_n());
    const int partial_cols = (Nff + gate_up_block_n - 1) / gate_up_block_n;
    const size_t partial_elements = (size_t) M * (size_t) partial_cols;
    if (err == cudaSuccess) {
        err = cudaMalloc((void **) &scratch.partial_max, partial_elements * sizeof(float));
    }
    if (err != cudaSuccess) {
        GGML_LOG_WARN("ggml-tilelang: W8A8 FFN scratch allocation failed on device %d M=%d Nff=%d: %s\n",
            device, M, Nff, cudaGetErrorString(err));
        (void) cudaGetLastError();
        if (scratch.tmp_a != nullptr) {
            (void) tilelang_cuda_check(cudaFree(scratch.tmp_a), "cudaFree");
            scratch.tmp_a = nullptr;
        }
        if (scratch.tmp_b != nullptr) {
            (void) tilelang_cuda_check(cudaFree(scratch.tmp_b), "cudaFree");
            scratch.tmp_b = nullptr;
        }
        if (scratch.partial_max != nullptr) {
            (void) tilelang_cuda_check(cudaFree(scratch.partial_max), "cudaFree");
            scratch.partial_max = nullptr;
        }
        return false;
    }

    scratch.tmp_elements = tmp_elements;
    scratch.partial_max_elements = partial_elements;
    return true;
}

static bool tilelang_cuda_w8a8_preload_device(int device) {
    if (!tilelang_cuda_w8a8_enabled() || !tilelang_cuda_w8a8_preload_enabled()) {
        return true;
    }
    if (!tilelang_cuda_valid_device(device)) {
        GGML_LOG_WARN("ggml-tilelang: unsupported CUDA device index for W8A8 preload: %d\n", device);
        return false;
    }

    tilelang_cuda_w8a8_state & state = tilelang_cuda_w8a8_state_get();
    std::lock_guard<std::mutex> lock(state.mutex);

    if (!tilelang_cuda_w8a8_load_locked(state)) {
        return false;
    }
    if (state.preload_done[device]) {
        return true;
    }

    size_t selected = 0;
    size_t failed = 0;
    int max_K = 0;
    const size_t uploaded_before = state.uploaded_tensors[device];
    const size_t bytes_before = state.uploaded_bytes[device];

    for (tilelang_cuda_w8a8_tensor & tensor : state.tensors) {
        selected++;
        max_K = std::max(max_K, tensor.K);
        if (!tilelang_cuda_w8a8_upload_tensor_locked(state, tensor, device, false)) {
            failed++;
        }
    }

    if (max_K > 0 && !tilelang_cuda_w8a8_ensure_scratch_locked(state, device, max_K, 1)) {
        failed++;
    }

    state.preload_done[device] = true;
    GGML_LOG_INFO(
        "ggml-tilelang: TileLang W8A8 preload device=%d selected=%zu uploaded_now=%zu "
        "uploaded_total=%zu bytes_now=%zu bytes_total=%zu failed=%zu\n",
        device,
        selected,
        state.uploaded_tensors[device] - uploaded_before,
        state.uploaded_tensors[device],
        state.uploaded_bytes[device] - bytes_before,
        state.uploaded_bytes[device],
        failed);

    return failed == 0;
}

static bool tilelang_cuda_w8a8_tensor_matches(
        const tilelang_cuda_w8a8_tensor & tensor,
        int K,
        int N) {
    return tensor.K == K &&
        tensor.N == N &&
        tensor.w_q_nbytes == (size_t) tensor.K * (size_t) tensor.N &&
        tensor.w_scale_nbytes == (size_t) tensor.N * sizeof(float) &&
        tensor.smooth_nbytes == (size_t) tensor.K * sizeof(float);
}

static bool tilelang_cuda_qwen35_ffn_validate_shape(const ggml_tensor * dst) {
    if (!tilelang_cuda_qwen35_ffn_op(dst) || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * up = dst->src[1];
    const ggml_tensor * gate = dst->src[2];
    const ggml_tensor * down = dst->src[3];
    if (x == nullptr || up == nullptr || gate == nullptr || down == nullptr) {
        return false;
    }

    if (x->type != GGML_TYPE_F32 ||
        up->type != GGML_TYPE_I8 ||
        gate->type != GGML_TYPE_I8 ||
        down->type != GGML_TYPE_I8) {
        return false;
    }

    if (!ggml_is_contiguous(x) ||
        !ggml_is_contiguous(up) ||
        !ggml_is_contiguous(gate) ||
        !ggml_is_contiguous(down) ||
        !ggml_is_contiguous(dst)) {
        return false;
    }

    const int64_t K = x->ne[0];
    const int64_t M = x->ne[1];
    const int64_t Nff = up->ne[1];
    if (K <= 0 || M <= 0 || Nff <= 0 ||
        K > (int64_t) std::numeric_limits<int>::max() ||
        M > (int64_t) std::numeric_limits<int>::max() ||
        Nff > (int64_t) std::numeric_limits<int>::max()) {
        return false;
    }

    if (x->ne[2] != 1 || x->ne[3] != 1 ||
        up->ne[0] != K || up->ne[2] != 1 || up->ne[3] != 1 ||
        gate->ne[0] != K || gate->ne[1] != Nff || gate->ne[2] != 1 || gate->ne[3] != 1 ||
        down->ne[0] != Nff || down->ne[1] != K || down->ne[2] != 1 || down->ne[3] != 1 ||
        dst->ne[0] != K || dst->ne[1] != M || dst->ne[2] != 1 || dst->ne[3] != 1) {
        return false;
    }

    if (M > 1 && ((K % 64) != 0 || (Nff % 64) != 0)) {
        return false;
    }

    return true;
}

static void tilelang_cuda_w8a8_launch_quantized_linear(
        int device,
        void * stream,
        const tilelang_cuda_w8a8_tensor & tensor,
        const int8_t * x_q,
        const float * x_scale,
        int M,
        float * y) {
    if (M == 1) {
        tilelang_w8a8_dot(
            tensor.d_w_q[device],
            tensor.d_w_scale[device],
            x_q,
            x_scale,
            y,
            tensor.K,
            tensor.N,
            stream);
    } else {
        tilelang_w8a8_gemm_dot(
            tensor.d_w_q[device],
            tensor.d_w_scale[device],
            x_q,
            x_scale,
            y,
            M,
            tensor.K,
            tensor.N,
            stream);
    }
}

static void tilelang_cuda_w8a8_launch_linear(
        int device,
        void * stream,
        const tilelang_cuda_w8a8_tensor & tensor,
        const float * x,
        int M,
        int8_t * x_q,
        float * x_scale,
        float * y) {
    if (M == 1) {
        tilelang_w8a8_gemv(
            tensor.d_w_q[device],
            tensor.d_w_scale[device],
            tensor.d_smooth[device],
            x,
            y,
            x_q,
            x_scale,
            tensor.K,
            tensor.N,
            stream);
    } else {
        tilelang_w8a8_gemm(
            tensor.d_w_q[device],
            tensor.d_w_scale[device],
            tensor.d_smooth[device],
            x,
            y,
            x_q,
            x_scale,
            M,
            tensor.K,
            tensor.N,
        stream);
    }
}

static bool tilelang_cuda_qwen35_ffn_try(
        int device,
        void * stream,
        ggml_tensor * dst) {
    if (!tilelang_cuda_w8a8_enabled() || !tilelang_cuda_qwen35_ffn_validate_shape(dst)) {
        return false;
    }

    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * weights[] = {
        dst->src[1],
        dst->src[2],
        dst->src[3],
    };
    if (!tilelang_cuda_tensor_on_device(x, device) ||
        !tilelang_cuda_tensor_on_device(weights[0], device) ||
        !tilelang_cuda_tensor_on_device(weights[1], device) ||
        !tilelang_cuda_tensor_on_device(weights[2], device) ||
        !tilelang_cuda_tensor_on_device(dst, device)) {
        return false;
    }

    const int K = (int) x->ne[0];
    const int M = (int) x->ne[1];
    const int Nff = (int) weights[0]->ne[1];

    tilelang_cuda_w8a8_state & state = tilelang_cuda_w8a8_state_get();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!tilelang_cuda_w8a8_load_locked(state)) {
        return false;
    }

    tilelang_cuda_w8a8_tensor * up = nullptr;
    tilelang_cuda_w8a8_tensor * gate = nullptr;
    tilelang_cuda_w8a8_tensor * down = nullptr;
    tilelang_cuda_w8a8_tensor ** tensors[] = { &up, &gate, &down };
    const int expect_K[] = { K, K, Nff };
    const int expect_N[] = { Nff, Nff, K };
    for (int i = 0; i < 3; ++i) {
        const auto it = state.by_name.find(weights[i]->name);
        if (it == state.by_name.end()) {
            return false;
        }

        *tensors[i] = &state.tensors[it->second];
        if (!tilelang_cuda_w8a8_tensor_matches(**tensors[i], expect_K[i], expect_N[i])) {
            return false;
        }

        if (!tilelang_cuda_w8a8_bind_replacement_weight_locked(state, **tensors[i], device, weights[i])) {
            return false;
        }

        if (!tilelang_cuda_w8a8_upload_tensor_locked(state, **tensors[i], device, true)) {
            return false;
        }
    }

    if (!tilelang_cuda_w8a8_ensure_ffn_scratch_locked(state, device, K, M, Nff)) {
        return false;
    }

    tilelang_cuda_w8a8_scratch & scratch = state.scratch[device];
    const bool shared_gate_up_smooth =
        !gate->smooth_group_id.empty() &&
        gate->smooth_group_id == up->smooth_group_id;

    g_cuda_tilelang_w8a8_qwen35_ffn_calls++;

    float * up_out = scratch.tmp_a;
    float * gate_out = scratch.tmp_b;
    bool hidden_quantized = false;
    if (shared_gate_up_smooth) {
        tilelang_w8a8_quant_x_rows(
            (const float *) x->data,
            gate->d_smooth[device],
            scratch.x_q,
            scratch.x_scale,
            M,
            K,
            stream);
        if (M == 1) {
            tilelang_cuda_w8a8_launch_quantized_linear(
                device, stream, *up, scratch.x_q, scratch.x_scale, M, up_out);
            tilelang_cuda_w8a8_launch_quantized_linear(
                device, stream, *gate, scratch.x_q, scratch.x_scale, M, gate_out);
        } else {
            tilelang_w8a8_gate_up_silu_scaled_gemm_dot_partial_max(
                up->d_w_q[device],
                up->d_w_scale[device],
                gate->d_w_q[device],
                gate->d_w_scale[device],
                scratch.x_q,
                scratch.x_scale,
                down->d_smooth[device],
                up_out,
                scratch.partial_max,
                M,
                K,
                Nff,
                stream);
            const int gate_up_block_n = std::max(1, tilelang_w8a8_gate_up_block_n());
            const int partial_cols = (Nff + gate_up_block_n - 1) / gate_up_block_n;
            tilelang_w8a8_quant_from_partial_max_rows(
                up_out,
                scratch.partial_max,
                scratch.x_q,
                scratch.x_scale,
                M,
                Nff,
                partial_cols,
                stream);
            hidden_quantized = true;
        }
    } else {
        tilelang_cuda_w8a8_launch_linear(
            device, stream, *up, (const float *) x->data, M, scratch.x_q, scratch.x_scale, up_out);
        tilelang_cuda_w8a8_launch_linear(
            device, stream, *gate, (const float *) x->data, M, scratch.x_q, scratch.x_scale, gate_out);
    }

    if (!hidden_quantized) {
        tilelang_w8a8_silu_mul_quant_rows(
            gate_out,
            up_out,
            down->d_smooth[device],
            scratch.x_q,
            scratch.x_scale,
            M,
            Nff,
            stream);
    }
    tilelang_cuda_w8a8_launch_quantized_linear(
        device, stream, *down, scratch.x_q, scratch.x_scale, M, (float *) dst->data);
    return true;
}

static void tilelang_cuda_w8a8_release_device(int device) {
    if (!tilelang_cuda_valid_device(device)) {
        return;
    }

    tilelang_cuda_w8a8_state & state = tilelang_cuda_w8a8_state_get();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.loaded) {
        return;
    }

    tilelang_cuda_set_device(device);
    for (tilelang_cuda_w8a8_tensor & tensor : state.tensors) {
        if (tensor.d_w_q[device] != nullptr) {
            tensor.d_w_q[device] = nullptr;
        }
        if (tensor.d_w_scale[device] != nullptr) {
            (void) tilelang_cuda_check(cudaFree(tensor.d_w_scale[device]), "cudaFree");
            tensor.d_w_scale[device] = nullptr;
        }
        if (tensor.d_smooth[device] != nullptr) {
            (void) tilelang_cuda_check(cudaFree(tensor.d_smooth[device]), "cudaFree");
            tensor.d_smooth[device] = nullptr;
        }
    }

    tilelang_cuda_w8a8_scratch & scratch = state.scratch[device];
    if (scratch.x_q != nullptr) {
        (void) tilelang_cuda_check(cudaFree(scratch.x_q), "cudaFree");
        scratch.x_q = nullptr;
    }
    if (scratch.x_scale != nullptr) {
        (void) tilelang_cuda_check(cudaFree(scratch.x_scale), "cudaFree");
        scratch.x_scale = nullptr;
    }
    if (scratch.tmp_a != nullptr) {
        (void) tilelang_cuda_check(cudaFree(scratch.tmp_a), "cudaFree");
        scratch.tmp_a = nullptr;
    }
    if (scratch.tmp_b != nullptr) {
        (void) tilelang_cuda_check(cudaFree(scratch.tmp_b), "cudaFree");
        scratch.tmp_b = nullptr;
    }
    if (scratch.partial_max != nullptr) {
        (void) tilelang_cuda_check(cudaFree(scratch.partial_max), "cudaFree");
        scratch.partial_max = nullptr;
    }
    scratch.K = 0;
    scratch.M = 0;
    scratch.tmp_elements = 0;
    scratch.partial_max_elements = 0;
    state.uploaded_tensors[device] = 0;
    state.uploaded_bytes[device] = 0;
    state.reused_w_q_tensors[device] = 0;
    state.reused_w_q_bytes[device] = 0;
    state.preload_done[device] = false;
}

static void tilelang_cuda_w8a8_log_summary() {
    const size_t qwen35_ffn_calls = g_cuda_tilelang_w8a8_qwen35_ffn_calls.load();

    if (!tilelang_cuda_w8a8_enabled() && qwen35_ffn_calls == 0 && !tilelang_cuda_w8a8_debug_enabled()) {
        return;
    }

    tilelang_cuda_w8a8_state & state = tilelang_cuda_w8a8_state_get();
    std::lock_guard<std::mutex> lock(state.mutex);

    size_t uploaded_tensors = 0;
    size_t uploaded_bytes = 0;
    size_t reused_w_q_tensors = 0;
    size_t reused_w_q_bytes = 0;
    for (int i = 0; i < TILELANG_CUDA_MAX_DEVICES; ++i) {
        uploaded_tensors += state.uploaded_tensors[i];
        uploaded_bytes += state.uploaded_bytes[i];
        reused_w_q_tensors += state.reused_w_q_tensors[i];
        reused_w_q_bytes += state.reused_w_q_bytes[i];
    }

    GGML_LOG_INFO(
        "ggml-tilelang: TileLang injection W8A8 summary enabled=%d preload=%d "
        "gguf_loaded=%d uploaded_tensors=%zu uploaded_bytes=%zu "
        "reused_w_q_tensors=%zu reused_w_q_bytes=%zu\n",
        tilelang_cuda_w8a8_enabled() ? 1 : 0,
        tilelang_cuda_w8a8_preload_enabled() ? 1 : 0,
        state.loaded ? 1 : 0,
        uploaded_tensors,
        uploaded_bytes,
        reused_w_q_tensors,
        reused_w_q_bytes);

    if (state.loaded) {
        GGML_LOG_INFO(
            "ggml-tilelang: TileLang injection W8A8 gguf path=%s source=%s source_format=%s "
            "recipe=%s smooth=%s alpha=%.3g tensor_count=%zu total_blob_bytes=%zu\n",
            state.path.c_str(),
            state.source_model.c_str(),
            state.source_format.c_str(),
            state.recipe.c_str(),
            state.smooth.c_str(),
            (double) state.alpha,
            state.tensors.size(),
            state.total_blob_bytes);
    }

    GGML_LOG_INFO(
        "ggml-tilelang: TileLang injection W8A8 qwen35_ffn calls=%zu\n",
        qwen35_ffn_calls);
}


bool ggml_tilelang_cuda_supports_op(const ggml_tensor * op) {
    return tilelang_cuda_w8a8_enabled() &&
        tilelang_cuda_qwen35_ffn_validate_shape(op);
}

bool ggml_tilelang_cuda_try_custom(
        int device,
        void * stream,
        ggml_tensor * dst) {
    return tilelang_cuda_qwen35_ffn_try(device, stream, dst);
}

bool ggml_tilelang_cuda_preload_device(int device) {
    return tilelang_cuda_w8a8_preload_device(device);
}

void ggml_tilelang_cuda_release_device(int device) {
    tilelang_cuda_w8a8_log_summary();
    tilelang_cuda_w8a8_release_device(device);
}
