#include "ggml-tilelang.h"
#include "tilelang-kernels.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <mutex>
#include <utility>
#include <vector>

#ifdef GGML_TILELANG_CUDA
#include "ggml-cuda.h"
#endif

// MVP context
struct tilelang_context {
    int device_id;
    ggml_backend_t cuda_backend = nullptr;
    ggml_backend_dev_t cuda_dev = nullptr;
    void * stream = nullptr;
    bool stream_logged = false;
    bool timing_enabled = false;
    std::vector<std::pair<void *, void *>> q8_0_timing_events;
    size_t n_mul_mat_f16 = 0;
    size_t n_mul_mat_q8_0 = 0;
    size_t n_mul_mat_q8_0_timed = 0;
    size_t n_cuda_delegate_runs = 0;
    size_t n_cuda_delegate_nodes = 0;
    double q8_0_kernel_ms = 0.0;
};

enum tilelang_mul_mat_reject_reason {
    TILELANG_REJECT_DISABLED = 0,
    TILELANG_REJECT_UNSUPPORTED_TYPE,
    TILELANG_REJECT_NON_CONTIGUOUS,
    TILELANG_REJECT_BATCH,
    TILELANG_REJECT_K_NOT_MULTIPLE_32,
    TILELANG_REJECT_SHAPE_MISMATCH,
    TILELANG_REJECT_DEVICE_MISMATCH,
    TILELANG_REJECT_COUNT,
};

static std::atomic<size_t> g_mul_mat_rejects[TILELANG_REJECT_COUNT];
static std::atomic<size_t> g_shape_hit_dump_count{0};
static std::atomic<size_t> g_shape_reject_dump_count{0};
static std::mutex g_shape_hist_mutex;
static std::map<std::pair<int64_t, int64_t>, size_t> g_q8_0_hit_shapes;
static std::map<std::pair<int64_t, int64_t>, size_t> g_q8_0_reject_shapes;

#ifdef GGML_TILELANG_USE_AOT_KERNELS
static const char * tilelang_kernel_path() {
    return "AOT TileLang";
}
#else
static const char * tilelang_kernel_path() {
    return "CUDA reference stub";
}
#endif

static bool tilelang_runtime_enabled() {
    return std::getenv("GGML_TILELANG_ENABLE") != nullptr;
}

static bool tilelang_shape_dump_enabled() {
    const char * env = std::getenv("GGML_TILELANG_DUMP_SHAPES");
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

static bool tilelang_timing_enabled() {
    const char * env = std::getenv("GGML_TILELANG_TIMING");
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

static bool tilelang_cuda_delegate_enabled() {
    const char * env = std::getenv("GGML_TILELANG_CUDA_DELEGATE");
    return env == nullptr || env[0] == '\0' || std::strcmp(env, "0") != 0;
}

static bool tilelang_native_enabled() {
    const char * env = std::getenv("GGML_TILELANG_NATIVE");
    return env == nullptr || env[0] == '\0' || std::strcmp(env, "0") != 0;
}

static const char * tilelang_native_policy_name() {
    const char * env = std::getenv("GGML_TILELANG_NATIVE_POLICY");
    if (env != nullptr && env[0] != '\0') {
        return env;
    }

    // In composite mode, claiming every decode MUL_MAT breaks CUDA-side fusion
    // opportunities. Keep the default conservative and use "all" for stress tests.
    return tilelang_cuda_delegate_enabled() ? "large_n" : "all";
}

static const char * tilelang_q8_0_variant_name() {
    const char * env = std::getenv("GGML_TILELANG_Q8_0_VARIANT");
    if (env == nullptr || env[0] == '\0') {
        return "thread";
    }
    if (std::strcmp(env, "block") == 0 || std::strcmp(env, "row1") == 0) {
        return "block";
    }
    if (std::strcmp(env, "warp") == 0) {
        return "warp";
    }
    return "thread";
}

#ifdef GGML_TILELANG_CUDA
static ggml_backend_dev_t tilelang_cuda_delegate_dev(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    return ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device);
}

static void * tilelang_get_cuda_stream(struct tilelang_context * ctx) {
    if (ctx->cuda_backend != nullptr) {
        ctx->stream = ggml_backend_cuda_get_stream(ctx->cuda_backend);
    }

#ifdef GGML_TILELANG_EXPERIMENTAL_STREAM_INTEROP
    if (ctx->stream == nullptr) {
        static std::once_flag interop_once;
        std::call_once(interop_once, [] {
            GGML_LOG_INFO("ggml-tilelang: using experimental CUDA stream interop\n");
        });

        void * stream = ggml_backend_cuda_get_device_stream(ctx->device_id);
        if (stream != nullptr) {
            ctx->stream = stream;
        }
    }
#endif

    if (!ctx->stream_logged) {
        GGML_LOG_INFO(
            "ggml-tilelang: CUDA device=%d stream=%p%s\n",
            ctx->device_id,
            ctx->stream,
            ctx->stream == nullptr ? " (fallback null stream)" : ""
        );
        ctx->stream_logged = true;
    }

    return ctx->stream;
}
#else
static ggml_backend_dev_t tilelang_cuda_delegate_dev(int device) {
    GGML_UNUSED(device);
    return nullptr;
}

static void * tilelang_get_cuda_stream(struct tilelang_context * ctx) {
    GGML_UNUSED(ctx);
    return nullptr;
}
#endif

#ifdef GGML_TILELANG_CUDA
static int tilelang_tensor_cuda_device(const ggml_tensor * t) {
    if (t == nullptr || t->buffer == nullptr) {
        return -1;
    }

    return ggml_backend_cuda_get_buffer_type_device(ggml_backend_buffer_get_type(t->buffer));
}

static int tilelang_cuda_stream_device(const void * stream) {
    return ggml_backend_cuda_get_stream_device(const_cast<void *>(stream));
}
#else
static int tilelang_tensor_cuda_device(const ggml_tensor * t) {
    GGML_UNUSED(t);
    return -1;
}

static int tilelang_cuda_stream_device(const void * stream) {
    GGML_UNUSED(stream);
    return -1;
}
#endif

static void tilelang_record_q8_0_shape(std::map<std::pair<int64_t, int64_t>, size_t> & hist, const ggml_tensor * op) {
    const ggml_tensor * w = op ? op->src[0] : nullptr;
    if (w == nullptr || w->type != GGML_TYPE_Q8_0) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_shape_hist_mutex);
    hist[{w->ne[0], w->ne[1]}]++;
}

static void tilelang_dump_tensor_shape(const char * label, const ggml_tensor * t) {
    if (!t) {
        GGML_LOG_INFO("ggml-tilelang: %s=<null>\n", label);
        return;
    }

    GGML_LOG_INFO(
        "ggml-tilelang: %s type=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]\n",
        label,
        ggml_type_name(t->type),
        (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
        t->nb[0], t->nb[1], t->nb[2], t->nb[3]
    );
}

static void tilelang_dump_mul_mat_shape_limited(
        const char * event,
        const char * variant,
        const ggml_tensor * op,
        int tilelang_device,
        const void * stream) {
    if (!tilelang_shape_dump_enabled()) {
        return;
    }

    constexpr size_t max_dumps = 64;
    std::atomic<size_t> * counter = std::strcmp(event, "hit") == 0 ?
        &g_shape_hit_dump_count :
        &g_shape_reject_dump_count;
    const size_t dump_idx = counter->fetch_add(1);
    if (dump_idx >= max_dumps) {
        return;
    }

    const ggml_tensor * w = op ? op->src[0] : nullptr;
    const ggml_tensor * x = op ? op->src[1] : nullptr;
    const int64_t K = w ? w->ne[0] : 0;
    const int64_t N = w ? w->ne[1] : 0;
    const int stream_device = tilelang_cuda_stream_device(stream);
    const int src0_device = tilelang_tensor_cuda_device(w);
    const int src1_device = tilelang_tensor_cuda_device(x);
    const int dst_device = tilelang_tensor_cuda_device(op);

    GGML_LOG_INFO(
        "ggml-tilelang: %s %s stream=%p tilelang_device=%d stream_device=%d src0_device=%d src1_device=%d dst_device=%d K=%lld N=%lld dump=%zu/%zu\n",
        event,
        variant,
        stream,
        tilelang_device,
        stream_device,
        src0_device,
        src1_device,
        dst_device,
        (long long) K,
        (long long) N,
        dump_idx + 1,
        max_dumps
    );
    tilelang_dump_tensor_shape("src0", w);
    tilelang_dump_tensor_shape("src1", x);
    tilelang_dump_tensor_shape("dst", op);
}

static void tilelang_reject_mul_mat(const ggml_tensor * op, enum tilelang_mul_mat_reject_reason reason, const char * detail) {
    g_mul_mat_rejects[reason]++;
    tilelang_record_q8_0_shape(g_q8_0_reject_shapes, op);
    tilelang_dump_mul_mat_shape_limited("reject", detail, op, -1, nullptr);
}

static bool tilelang_validate_tensor_device(const char * label, const ggml_tensor * tensor, int device_id) {
    const int tensor_device = tilelang_tensor_cuda_device(tensor);
    if (tensor_device != device_id) {
        GGML_LOG_ERROR(
            "ggml-tilelang: %s CUDA buffer device mismatch: tilelang_device=%d tensor_device=%d\n",
            label,
            device_id,
            tensor_device
        );
        return false;
    }

    return true;
}

static bool tilelang_validate_cuda_devices(struct tilelang_context * ctx, const ggml_tensor * op, const void * stream) {
    const ggml_tensor * w = op ? op->src[0] : nullptr;
    const ggml_tensor * x = op ? op->src[1] : nullptr;

    if (!tilelang_validate_tensor_device("src0", w, ctx->device_id) ||
        !tilelang_validate_tensor_device("src1", x, ctx->device_id) ||
        !tilelang_validate_tensor_device("dst", op, ctx->device_id)) {
        tilelang_reject_mul_mat(op, TILELANG_REJECT_DEVICE_MISMATCH, "CUDA buffer device mismatch");
        return false;
    }

    const int stream_device = tilelang_cuda_stream_device(stream);
    if (stream != nullptr && stream_device >= 0 && stream_device != ctx->device_id) {
        GGML_LOG_ERROR(
            "ggml-tilelang: CUDA stream device mismatch: tilelang_device=%d stream_device=%d\n",
            ctx->device_id,
            stream_device
        );
        tilelang_reject_mul_mat(op, TILELANG_REJECT_DEVICE_MISMATCH, "CUDA stream device mismatch");
        return false;
    }

    return true;
}

static bool tilelang_supports_mul_mat(const ggml_tensor * op);

//
// backend interface
//

static const char * tilelang_backend_get_name(ggml_backend_t /*backend*/) {
    return "TileLang";
}

static void tilelang_log_q8_0_shape_hist(const char * label, const std::map<std::pair<int64_t, int64_t>, size_t> & hist) {
    if (hist.empty()) {
        return;
    }

    GGML_LOG_INFO("ggml-tilelang: Q8_0 %s shape histogram:\n", label);
    size_t n_printed = 0;
    for (const auto & it : hist) {
        if (n_printed++ >= 16) {
            GGML_LOG_INFO("ggml-tilelang:   ...\n");
            break;
        }
        GGML_LOG_INFO(
            "ggml-tilelang:   K=%lld N=%lld count=%zu\n",
            (long long) it.first.first,
            (long long) it.first.second,
            it.second
        );
    }
}

static void tilelang_backend_free(ggml_backend_t backend) {
    struct tilelang_context * ctx = (struct tilelang_context *)backend->context;
#ifdef GGML_TILELANG_CUDA
    for (const auto & events : ctx->q8_0_timing_events) {
        ctx->q8_0_kernel_ms += tilelang_cuda_event_elapsed_ms(events.first, events.second);
        ctx->n_mul_mat_q8_0_timed++;
        tilelang_cuda_event_destroy(events.first);
        tilelang_cuda_event_destroy(events.second);
    }
    ctx->q8_0_timing_events.clear();
#endif

    const size_t n_rejected =
        g_mul_mat_rejects[TILELANG_REJECT_DISABLED].load() +
        g_mul_mat_rejects[TILELANG_REJECT_UNSUPPORTED_TYPE].load() +
        g_mul_mat_rejects[TILELANG_REJECT_NON_CONTIGUOUS].load() +
        g_mul_mat_rejects[TILELANG_REJECT_BATCH].load() +
        g_mul_mat_rejects[TILELANG_REJECT_K_NOT_MULTIPLE_32].load() +
        g_mul_mat_rejects[TILELANG_REJECT_SHAPE_MISMATCH].load() +
        g_mul_mat_rejects[TILELANG_REJECT_DEVICE_MISMATCH].load();

    const size_t n_supported = ctx->n_mul_mat_f16 + ctx->n_mul_mat_q8_0;
    GGML_LOG_INFO("ggml-tilelang: MUL_MAT_F16 calls = %zu\n", ctx->n_mul_mat_f16);
    GGML_LOG_INFO("ggml-tilelang: MUL_MAT_Q8_0 calls = %zu\n", ctx->n_mul_mat_q8_0);
    GGML_LOG_INFO("ggml-tilelang: CUDA delegate runs = %zu nodes = %zu\n", ctx->n_cuda_delegate_runs, ctx->n_cuda_delegate_nodes);
    GGML_LOG_INFO("ggml-tilelang: MUL_MAT supported=%zu rejected=%zu\n", n_supported, n_rejected);
    if (ctx->n_mul_mat_q8_0_timed > 0) {
        GGML_LOG_INFO(
            "ggml-tilelang: MUL_MAT_Q8_0 kernel time total=%.3f ms calls=%zu avg=%.6f ms\n",
            ctx->q8_0_kernel_ms,
            ctx->n_mul_mat_q8_0_timed,
            ctx->q8_0_kernel_ms / (double) ctx->n_mul_mat_q8_0_timed
        );
    }
    GGML_LOG_INFO(
        "ggml-tilelang: MUL_MAT rejected: disabled=%zu unsupported_type=%zu non_contiguous=%zu batch=%zu k_not_multiple_32=%zu shape_mismatch=%zu device_mismatch=%zu\n",
        g_mul_mat_rejects[TILELANG_REJECT_DISABLED].load(),
        g_mul_mat_rejects[TILELANG_REJECT_UNSUPPORTED_TYPE].load(),
        g_mul_mat_rejects[TILELANG_REJECT_NON_CONTIGUOUS].load(),
        g_mul_mat_rejects[TILELANG_REJECT_BATCH].load(),
        g_mul_mat_rejects[TILELANG_REJECT_K_NOT_MULTIPLE_32].load(),
        g_mul_mat_rejects[TILELANG_REJECT_SHAPE_MISMATCH].load(),
        g_mul_mat_rejects[TILELANG_REJECT_DEVICE_MISMATCH].load()
    );

    if (n_supported > 0) {
        std::lock_guard<std::mutex> lock(g_shape_hist_mutex);
        tilelang_log_q8_0_shape_hist("hit", g_q8_0_hit_shapes);
        tilelang_log_q8_0_shape_hist("reject", g_q8_0_reject_shapes);
    }

    if (ctx->cuda_backend != nullptr) {
        ggml_backend_free(ctx->cuda_backend);
        ctx->cuda_backend = nullptr;
    }

    delete ctx;
    delete backend;
}

static void tilelang_backend_synchronize(ggml_backend_t backend) {
    struct tilelang_context * ctx = (struct tilelang_context *)backend->context;
#ifdef GGML_TILELANG_CUDA
    if (ctx->cuda_backend != nullptr) {
        ggml_backend_synchronize(ctx->cuda_backend);
    } else {
        tilelang_cuda_synchronize(ctx->stream);
    }
#else
    GGML_UNUSED(ctx);
#endif
}

static enum ggml_status tilelang_op_mul_mat_f16(struct tilelang_context * ctx, ggml_tensor * op) {
    static std::once_flag once;
    std::call_once(once, [] {
        GGML_LOG_INFO("ggml-tilelang: executing F16 MUL_MAT via TileLang\n");
    });
    void * stream = tilelang_get_cuda_stream(ctx);
    if (!tilelang_validate_cuda_devices(ctx, op, stream)) {
        return GGML_STATUS_FAILED;
    }
    tilelang_dump_mul_mat_shape_limited("hit", "MUL_MAT_F16", op, ctx->device_id, stream);
    ctx->n_mul_mat_f16++;

    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    ggml_tensor * dst = op;

    const void * w_f16 = w->data;
    const float * x_f32 = (const float *)x->data;
    float * y_f32 = (float *)dst->data;

    int K = w->ne[0];
    int N = w->ne[1];

    tilelang_f16_gemv(w_f16, x_f32, y_f32, K, N, stream);

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status tilelang_op_mul_mat_q8_0(struct tilelang_context * ctx, ggml_tensor * op) {
    static std::once_flag once;
    std::call_once(once, [] {
        GGML_LOG_INFO("ggml-tilelang: executing Q8_0 MUL_MAT via TileLang variant=%s\n", tilelang_q8_0_variant_name());
    });
    void * stream = tilelang_get_cuda_stream(ctx);
    if (!tilelang_validate_cuda_devices(ctx, op, stream)) {
        return GGML_STATUS_FAILED;
    }
    tilelang_record_q8_0_shape(g_q8_0_hit_shapes, op);
    tilelang_dump_mul_mat_shape_limited("hit", "MUL_MAT_Q8_0", op, ctx->device_id, stream);
    ctx->n_mul_mat_q8_0++;

    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    ggml_tensor * dst = op;

    const void * w_q8_0 = w->data;
    const float * x_f32 = (const float *)x->data;
    float * y_f32 = (float *)dst->data;

    int K = w->ne[0];
    int N = w->ne[1];

    void * timing_start = nullptr;
    void * timing_stop = nullptr;
#ifdef GGML_TILELANG_CUDA
    if (ctx->timing_enabled) {
        timing_start = tilelang_cuda_event_create();
        timing_stop = tilelang_cuda_event_create();
        tilelang_cuda_event_record(timing_start, stream);
    }
#endif
    tilelang_q8_0_gemv(w_q8_0, x_f32, y_f32, K, N, stream);
#ifdef GGML_TILELANG_CUDA
    if (ctx->timing_enabled) {
        tilelang_cuda_event_record(timing_stop, stream);
        ctx->q8_0_timing_events.emplace_back(timing_start, timing_stop);
    }
#endif

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status tilelang_op_mul_mat(struct tilelang_context * ctx, ggml_tensor * op) {
    const ggml_tensor * w = op->src[0];
    if (!w) {
        return GGML_STATUS_FAILED;
    }

    switch (w->type) {
        case GGML_TYPE_F16:
            return tilelang_op_mul_mat_f16(ctx, op);
        case GGML_TYPE_Q8_0:
            return tilelang_op_mul_mat_q8_0(ctx, op);
        default:
            GGML_LOG_ERROR("ggml-tilelang: unsupported MUL_MAT weight type %s\n", ggml_type_name(w->type));
            return GGML_STATUS_FAILED;
    }
}

static bool tilelang_is_noop(const ggml_tensor * op) {
    return ggml_is_empty(op) ||
        op->op == GGML_OP_NONE ||
        op->op == GGML_OP_RESHAPE ||
        op->op == GGML_OP_VIEW ||
        op->op == GGML_OP_PERMUTE ||
        op->op == GGML_OP_TRANSPOSE;
}

static bool tilelang_native_policy_allows(const ggml_tensor * op) {
    if (!tilelang_native_enabled()) {
        return false;
    }

    const char * policy = tilelang_native_policy_name();
    if (std::strcmp(policy, "all") == 0) {
        return true;
    }
    if (std::strcmp(policy, "none") == 0) {
        return false;
    }

    if (std::strcmp(policy, "large_n") == 0 || std::strcmp(policy, "output") == 0) {
        const ggml_tensor * w = op ? op->src[0] : nullptr;
        return w != nullptr && w->ne[1] >= 8192;
    }

    static std::once_flag once;
    std::call_once(once, [policy] {
        GGML_LOG_WARN("ggml-tilelang: unknown GGML_TILELANG_NATIVE_POLICY=%s, using large_n\n", policy);
    });
    const ggml_tensor * w = op ? op->src[0] : nullptr;
    return w != nullptr && w->ne[1] >= 8192;
}

static bool tilelang_native_supports_op(const ggml_tensor * op) {
    return op != nullptr &&
        op->op == GGML_OP_MUL_MAT &&
        tilelang_native_policy_allows(op) &&
        tilelang_supports_mul_mat(op);
}

static enum ggml_status tilelang_flush_cuda_delegate_run(
        struct tilelang_context * ctx,
        ggml_cgraph * graph,
        int i_start,
        int i_end) {
    if (i_start >= i_end) {
        return GGML_STATUS_SUCCESS;
    }

    if (ctx->cuda_backend == nullptr) {
        GGML_LOG_ERROR("ggml-tilelang: CUDA delegate backend is not initialized\n");
        return GGML_STATUS_FAILED;
    }

    ggml_cgraph subgraph = ggml_graph_view(graph, i_start, i_end);
    subgraph.uid = ggml_graph_next_uid();
    ctx->n_cuda_delegate_runs++;
    ctx->n_cuda_delegate_nodes += subgraph.n_nodes;

    return ggml_backend_graph_compute_async(ctx->cuda_backend, &subgraph);
}

static enum ggml_status tilelang_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    struct tilelang_context * ctx = (struct tilelang_context *)backend->context;

    if (ctx->cuda_backend != nullptr) {
        ctx->stream = ggml_backend_cuda_get_stream(ctx->cuda_backend);
    }

    int delegate_start = 0;
    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * op = graph->nodes[i];

        if (tilelang_is_noop(op) || (op->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        if (tilelang_native_supports_op(op)) {
            enum ggml_status status = tilelang_flush_cuda_delegate_run(ctx, graph, delegate_start, i);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }

            status = tilelang_op_mul_mat(ctx, op);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }

            delegate_start = i + 1;
            continue;
        }

        if (ctx->cuda_dev == nullptr || !ggml_backend_dev_supports_op(ctx->cuda_dev, op)) {
            GGML_LOG_ERROR("ggml-tilelang: unsupported op %s\n", ggml_op_name(op->op));
            return GGML_STATUS_FAILED;
        }
    }

    return tilelang_flush_cuda_delegate_run(ctx, graph, delegate_start, graph->n_nodes);
}

static const struct ggml_backend_i tilelang_backend_i = {
    /* .get_name                = */ tilelang_backend_get_name,
    /* .free                    = */ tilelang_backend_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ tilelang_backend_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ tilelang_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

//
// device interface
//

static const char * tilelang_device_get_name(ggml_backend_dev_t /*dev*/) {
    return "TileLang";
}

static const char * tilelang_device_get_description(ggml_backend_dev_t /*dev*/) {
    return "TileLang Backend";
}

static void tilelang_device_get_memory(ggml_backend_dev_t /*dev*/, size_t * free, size_t * total) {
#ifdef GGML_TILELANG_CUDA
    ggml_backend_dev_t cuda_dev = tilelang_cuda_delegate_dev(0);
    if (cuda_dev != nullptr) {
        ggml_backend_dev_memory(cuda_dev, free, total);
        return;
    }
#endif

    if (free) {
        *free = 0;
    }
    if (total) {
        *total = 0;
    }
}

static enum ggml_backend_dev_type tilelang_device_get_type(ggml_backend_dev_t /*dev*/) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void tilelang_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
#ifdef GGML_TILELANG_CUDA
    ggml_backend_dev_t cuda_dev = tilelang_cuda_delegate_dev(0);
    if (cuda_dev != nullptr) {
        ggml_backend_dev_get_props(cuda_dev, props);
    }
#endif

    props->name        = tilelang_device_get_name(dev);
    props->description = tilelang_device_get_description(dev);
    tilelang_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->type        = tilelang_device_get_type(dev);
    props->caps.async                 = false;
    props->caps.host_buffer           = false;
    props->caps.buffer_from_host_ptr  = false;
    props->caps.events                = false;
}

static ggml_guid_t ggml_backend_tilelang_guid() {
    static ggml_guid guid = { 'T', 'i', 'l', 'e', 'L', 'a', 'n', 'g', 0, 0, 0, 0, 0, 0, 0, 0 };
    return &guid;
}

static ggml_backend_t tilelang_device_init_backend(ggml_backend_dev_t dev, const char * /*params*/) {
    static std::once_flag once;
    std::call_once(once, [] {
        GGML_LOG_INFO("ggml-tilelang: kernel path = %s\n", tilelang_kernel_path());
    });

    struct tilelang_context * ctx = new tilelang_context();
    ctx->device_id = 0;
    ctx->timing_enabled = tilelang_timing_enabled();
#ifdef GGML_TILELANG_CUDA
    GGML_LOG_INFO(
        "ggml-tilelang: native kernels %s policy=%s\n",
        tilelang_native_enabled() ? "enabled" : "disabled",
        tilelang_native_policy_name()
    );
    if (tilelang_cuda_delegate_enabled()) {
        ctx->cuda_dev = tilelang_cuda_delegate_dev(ctx->device_id);
        ctx->cuda_backend = ggml_backend_cuda_init(ctx->device_id);
        if (ctx->cuda_backend == nullptr) {
            GGML_LOG_ERROR("ggml-tilelang: failed to initialize CUDA delegate backend for device %d\n", ctx->device_id);
            delete ctx;
            return nullptr;
        }
        ctx->stream = ggml_backend_cuda_get_stream(ctx->cuda_backend);
        GGML_LOG_INFO("ggml-tilelang: CUDA delegate backend enabled device=%d stream=%p\n", ctx->device_id, ctx->stream);
    }
    if (ctx->timing_enabled) {
        GGML_LOG_INFO("ggml-tilelang: CUDA event timing enabled\n");
    }
#endif

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_tilelang_guid(),
        /* .iface     = */ tilelang_backend_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };
    return backend;
}

static ggml_backend_buffer_type_t tilelang_device_get_buffer_type(ggml_backend_dev_t /*dev*/) {
#ifdef GGML_TILELANG_CUDA
    return ggml_backend_cuda_buffer_type(0); // MVP: always use device 0
#else
    return nullptr;
#endif
}

static bool tilelang_supports_mul_mat_f16(const ggml_tensor * op) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];

    if (!w || !x) {
        tilelang_reject_mul_mat(op, TILELANG_REJECT_SHAPE_MISMATCH, "missing src");
        return false;
    }

    // MVP: only F16 weights × F32 activations → F32 output
    if (x->type != GGML_TYPE_F32)  {
        tilelang_reject_mul_mat(op, TILELANG_REJECT_UNSUPPORTED_TYPE, "MUL_MAT_F16 unsupported src1 type");
        return false;
    }
    if (op->type != GGML_TYPE_F32) {
        tilelang_reject_mul_mat(op, TILELANG_REJECT_UNSUPPORTED_TYPE, "MUL_MAT_F16 unsupported dst type");
        return false;
    }

    // require contiguous layout
    if (!ggml_is_contiguous(w) || !ggml_is_contiguous(x) || !ggml_is_contiguous(op)) {
        tilelang_reject_mul_mat(op, TILELANG_REJECT_NON_CONTIGUOUS, "MUL_MAT_F16 non-contiguous");
        return false;
    }

    const int64_t K = w->ne[0];
    const int64_t N = w->ne[1];

    if (K <= 0 || N <= 0) {
        tilelang_reject_mul_mat(op, TILELANG_REJECT_SHAPE_MISMATCH, "MUL_MAT_F16 empty shape");
        return false;
    }

    if (x->ne[0] != K || op->ne[0] != N) {
        tilelang_reject_mul_mat(op, TILELANG_REJECT_SHAPE_MISMATCH, "MUL_MAT_F16 shape mismatch");
        return false;
    }

    // stub kernel only handles simple 2D GEMV: y[N] = W[N,K] × x[K]
    // no batching (ne[2], ne[3] must be 1)
    if (x->ne[1] != 1 || op->ne[1] != 1 ||
        w->ne[2] != 1 || w->ne[3] != 1 ||
        x->ne[2] != 1 || x->ne[3] != 1 ||
        op->ne[2] != 1 || op->ne[3] != 1) {
        tilelang_reject_mul_mat(op, TILELANG_REJECT_BATCH, "MUL_MAT_F16 batch != 1");
        return false;
    }

    return true;
}

static bool tilelang_supports_mul_mat_q8_0(const ggml_tensor * op) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];

    if (!w || !x) {
        tilelang_reject_mul_mat(op, TILELANG_REJECT_SHAPE_MISMATCH, "missing src");
        return false;
    }
    if (op->op != GGML_OP_MUL_MAT) {
        tilelang_reject_mul_mat(op, TILELANG_REJECT_SHAPE_MISMATCH, "not MUL_MAT");
        return false;
    }

    if (x->type  != GGML_TYPE_F32)  {
        tilelang_reject_mul_mat(op, TILELANG_REJECT_UNSUPPORTED_TYPE, "MUL_MAT_Q8_0 unsupported src1 type");
        return false;
    }
    if (op->type != GGML_TYPE_F32)  {
        tilelang_reject_mul_mat(op, TILELANG_REJECT_UNSUPPORTED_TYPE, "MUL_MAT_Q8_0 unsupported dst type");
        return false;
    }

    if (!ggml_is_contiguous(w) || !ggml_is_contiguous(x) || !ggml_is_contiguous(op)) {
        tilelang_reject_mul_mat(op, TILELANG_REJECT_NON_CONTIGUOUS, "MUL_MAT_Q8_0 non-contiguous");
        return false;
    }

    const int64_t K = w->ne[0];
    const int64_t N = w->ne[1];

    if (K <= 0 || N <= 0) {
        tilelang_reject_mul_mat(op, TILELANG_REJECT_SHAPE_MISMATCH, "MUL_MAT_Q8_0 empty shape");
        return false;
    }
    if (K % 32 != 0) {
        tilelang_reject_mul_mat(op, TILELANG_REJECT_K_NOT_MULTIPLE_32, "MUL_MAT_Q8_0 K % 32 != 0");
        return false;
    }

    if (x->ne[0] != K || op->ne[0] != N) {
        tilelang_reject_mul_mat(op, TILELANG_REJECT_SHAPE_MISMATCH, "MUL_MAT_Q8_0 shape mismatch");
        return false;
    }

    if (x->ne[1] != 1 || op->ne[1] != 1 ||
        w->ne[2]  != 1 || w->ne[3]  != 1 ||
        x->ne[2]  != 1 || x->ne[3]  != 1 ||
        op->ne[2] != 1 || op->ne[3] != 1) {
        tilelang_reject_mul_mat(op, TILELANG_REJECT_BATCH, "MUL_MAT_Q8_0 batch != 1");
        return false;
    }

    return true;
}

static bool tilelang_supports_mul_mat(const ggml_tensor * op) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];

    if (!w || !x) {
        tilelang_reject_mul_mat(op, TILELANG_REJECT_SHAPE_MISMATCH, "missing src");
        return false;
    }

    switch (w->type) {
        case GGML_TYPE_F16:
            return tilelang_supports_mul_mat_f16(op);
        case GGML_TYPE_Q8_0:
            return tilelang_supports_mul_mat_q8_0(op);
        default:
            tilelang_reject_mul_mat(op, TILELANG_REJECT_UNSUPPORTED_TYPE, "unsupported weight type");
            return false;
    }
}

static bool tilelang_device_supports_op(ggml_backend_dev_t /*dev*/, const struct ggml_tensor * op) {
    if (!tilelang_runtime_enabled()) {
        if (op->op == GGML_OP_MUL_MAT) {
            tilelang_reject_mul_mat(op, TILELANG_REJECT_DISABLED, "disabled");
        }
        return false;
    }

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_MUL_MAT:
            if (tilelang_native_enabled() && tilelang_native_policy_allows(op) && tilelang_supports_mul_mat(op)) {
                return true;
            }
            break;
        default:
            break;
    }

#ifdef GGML_TILELANG_CUDA
    if (tilelang_cuda_delegate_enabled()) {
        ggml_backend_dev_t cuda_dev = tilelang_cuda_delegate_dev(0);
        return cuda_dev != nullptr && ggml_backend_dev_supports_op(cuda_dev, op);
    }
#endif

    return false;
}

static bool tilelang_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft == tilelang_device_get_buffer_type(dev);
}

static bool tilelang_device_offload_op(ggml_backend_dev_t /*dev*/, const struct ggml_tensor * op) {
#ifdef GGML_TILELANG_CUDA
    if (tilelang_cuda_delegate_enabled()) {
        ggml_backend_dev_t cuda_dev = tilelang_cuda_delegate_dev(0);
        return cuda_dev != nullptr && ggml_backend_dev_offload_op(cuda_dev, op);
    }
#else
    GGML_UNUSED(op);
#endif
    return false;
}

static const struct ggml_backend_device_i tilelang_device_i = {
    /* .get_name             = */ tilelang_device_get_name,
    /* .get_description      = */ tilelang_device_get_description,
    /* .get_memory           = */ tilelang_device_get_memory,
    /* .get_type             = */ tilelang_device_get_type,
    /* .get_props            = */ tilelang_device_get_props,
    /* .init_backend         = */ tilelang_device_init_backend,
    /* .get_buffer_type      = */ tilelang_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ tilelang_device_supports_op,
    /* .supports_buft        = */ tilelang_device_supports_buft,
    /* .offload_op           = */ tilelang_device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

//
// backend reg interface
//

static const char * tilelang_reg_get_name(ggml_backend_reg_t /*reg*/) {
    return "TileLang";
}

static size_t tilelang_reg_get_device_count(ggml_backend_reg_t /*reg*/) {
    return 1;
}

static ggml_backend_dev_t tilelang_reg_get_device(ggml_backend_reg_t reg, size_t /*index*/) {
    static ggml_backend_device dev = {
        /* .iface   = */ tilelang_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };
    return &dev;
}

static void * tilelang_reg_get_proc_address(ggml_backend_reg_t /*reg*/, const char * /*name*/) {
    return nullptr;
}

static const struct ggml_backend_reg_i tilelang_reg_i = {
    /* .get_name         = */ tilelang_reg_get_name,
    /* .get_device_count = */ tilelang_reg_get_device_count,
    /* .get_device       = */ tilelang_reg_get_device,
    /* .get_proc_address = */ tilelang_reg_get_proc_address,
};

GGML_API ggml_backend_reg_t ggml_backend_tilelang_reg(void) {
    static struct ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ tilelang_reg_i,
        /* .context     = */ nullptr,
    };
    return &reg;
}

GGML_API ggml_backend_t ggml_backend_tilelang_init(int device) {
    ggml_backend_dev_t dev = tilelang_reg_get_device(ggml_backend_tilelang_reg(), device);
    return ggml_backend_dev_init(dev, nullptr);
}

GGML_API bool ggml_backend_is_tilelang(ggml_backend_t backend) {
    return backend != NULL && std::string(ggml_backend_dev_name(ggml_backend_get_device(backend))) == "TileLang";
}

// Dynamic loading exports
static int tilelang_backend_score() {
#ifdef GGML_TILELANG_CUDA
    return ggml_backend_cuda_get_device_count() > 0 ? 100 : 0;
#else
    return 0;
#endif
}

GGML_BACKEND_DL_IMPL(ggml_backend_tilelang_reg)
GGML_BACKEND_DL_SCORE_IMPL(tilelang_backend_score)
