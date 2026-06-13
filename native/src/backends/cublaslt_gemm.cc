#include "backends/cublaslt_gemm.h"

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <mutex>

#include "ayaka/check.h"
#include "ayaka/dtype.h"

namespace {

ayaka_status_t lt_status(cublasStatus_t status) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return ayaka_status_ok();
    }
    return ayaka_status_make(AYAKA_STATUS_INTERNAL_ERROR,
                             cublasGetStatusString(status));
}

#define AYAKA_LT_RETURN_IF_ERROR(expr) AYAKA_RETURN_IF_ERROR(lt_status(expr))

// cuBLASLt handles are expensive to create but cheap to keep; cache one per
// device for the process lifetime (never destroyed, like the CUDA primary
// context they belong to).
constexpr int kMaxCachedDevices = 64;
std::mutex g_handle_mutex;
cublasLtHandle_t g_handles[kMaxCachedDevices] = {};

ayaka_status_t cached_handle(int32_t device_ordinal, cublasLtHandle_t* out) {
    AYAKA_CHECK(device_ordinal >= 0 && device_ordinal < kMaxCachedDevices,
                "gemm: device ordinal out of cached-handle range");
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    if (g_handles[device_ordinal] == nullptr) {
        const cudaError_t err = cudaSetDevice(device_ordinal);
        if (err != cudaSuccess) {
            return ayaka_status_make(AYAKA_STATUS_DEVICE_ERROR,
                                     cudaGetErrorString(err));
        }
        AYAKA_LT_RETURN_IF_ERROR(cublasLtCreate(&g_handles[device_ordinal]));
    }
    *out = g_handles[device_ordinal];
    return ayaka_status_ok();
}

ayaka_status_t lt_dtype(int32_t dtype, cudaDataType_t* out) {
    switch (dtype) {
        case AYAKA_DTYPE_F32:
            *out = CUDA_R_32F;
            return ayaka_status_ok();
        case AYAKA_DTYPE_F16:
            *out = CUDA_R_16F;
            return ayaka_status_ok();
        case AYAKA_DTYPE_BF16:
            *out = CUDA_R_16BF;
            return ayaka_status_ok();
        default:
            return ayaka_status_make(AYAKA_STATUS_DTYPE_ERROR,
                                     "gemm: dtype must be f32, f16, or bf16");
    }
}

// RAII so every early return frees the cuBLASLt descriptors (no exceptions
// cross this file).
struct MatmulDesc {
    cublasLtMatmulDesc_t desc = nullptr;
    ~MatmulDesc() {
        if (desc) cublasLtMatmulDescDestroy(desc);
    }
};

struct MatrixLayout {
    cublasLtMatrixLayout_t layout = nullptr;
    ~MatrixLayout() {
        if (layout) cublasLtMatrixLayoutDestroy(layout);
    }
};

struct MatmulPreference {
    cublasLtMatmulPreference_t pref = nullptr;
    ~MatmulPreference() {
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
    }
};

// Column-major layout describing the row-major stored buffer of `view`:
// stored [r, c] is read as [c, r] with leading dimension c.
ayaka_status_t row_major_as_col_major(const ayaka_tensor_view_t& view,
                                      cudaDataType_t data_type,
                                      MatrixLayout* layout) {
    return lt_status(cublasLtMatrixLayoutCreate(
        &layout->layout, data_type, static_cast<uint64_t>(view.shape[1]),
        static_cast<uint64_t>(view.shape[0]),
        static_cast<int64_t>(view.shape[1])));
}

}  // namespace

namespace ayaka {

ayaka_status_t gemm_cublaslt(const ayaka_tensor_view_t& out,
                             const ayaka_tensor_view_t& a,
                             const ayaka_tensor_view_t& b,
                             const ayaka_tensor_view_t* bias,
                             bool trans_a,
                             bool trans_b,
                             void* workspace,
                             size_t workspace_bytes,
                             ayaka_stream_t stream) {
    // cuBLASLt is column-major. The row-major out[M, N] is computed as its
    // column-major transpose: out_cm[N, M] = op_b(b)^T @ op_a(a)^T, so b
    // feeds cuBLASLt matrix "A", a feeds matrix "B", and M/N swap roles.
    // With the stored buffers read as column-major ([r, c] -> [c, r]), the
    // user trans flags map directly onto CUBLAS_OP_T on the swapped slots.
    const int64_t m = trans_a ? a.shape[1] : a.shape[0];
    const int64_t n = trans_b ? b.shape[0] : b.shape[1];

    cublasLtHandle_t handle = nullptr;
    AYAKA_RETURN_IF_ERROR(cached_handle(out.device_ordinal, &handle));
    {
        const cudaError_t err = cudaSetDevice(out.device_ordinal);
        if (err != cudaSuccess) {
            return ayaka_status_make(AYAKA_STATUS_DEVICE_ERROR,
                                     cudaGetErrorString(err));
        }
    }

    cudaDataType_t data_type;
    AYAKA_RETURN_IF_ERROR(lt_dtype(out.dtype, &data_type));

    MatmulDesc op;
    AYAKA_LT_RETURN_IF_ERROR(
        cublasLtMatmulDescCreate(&op.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t op_a = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t op_b = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    AYAKA_LT_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
        op.desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)));
    AYAKA_LT_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
        op.desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)));
    if (bias != nullptr) {
        // CUBLASLT_EPILOGUE_BIAS broadcasts a length-N vector over the rows
        // of the column-major result, i.e. over the rows of out.
        const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        AYAKA_LT_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
            op.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
            sizeof(epilogue)));
        const void* bias_ptr = bias->data;
        AYAKA_LT_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
            op.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr,
            sizeof(bias_ptr)));
    }

    MatrixLayout layout_a;  // b's buffer in the swapped "A" slot
    AYAKA_RETURN_IF_ERROR(row_major_as_col_major(b, data_type, &layout_a));
    MatrixLayout layout_b;  // a's buffer in the swapped "B" slot
    AYAKA_RETURN_IF_ERROR(row_major_as_col_major(a, data_type, &layout_b));
    MatrixLayout layout_c;
    AYAKA_LT_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(
        &layout_c.layout, data_type, static_cast<uint64_t>(n),
        static_cast<uint64_t>(m), static_cast<int64_t>(n)));

    MatmulPreference pref;
    AYAKA_LT_RETURN_IF_ERROR(cublasLtMatmulPreferenceCreate(&pref.pref));
    const uint64_t max_workspace = workspace_bytes;
    AYAKA_LT_RETURN_IF_ERROR(cublasLtMatmulPreferenceSetAttribute(
        pref.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace,
        sizeof(max_workspace)));

    cublasLtMatmulHeuristicResult_t heuristic = {};
    int num_results = 0;
    AYAKA_LT_RETURN_IF_ERROR(cublasLtMatmulAlgoGetHeuristic(
        handle, op.desc, layout_a.layout, layout_b.layout, layout_c.layout,
        layout_c.layout, pref.pref, 1, &heuristic, &num_results));
    if (num_results == 0 || heuristic.state != CUBLAS_STATUS_SUCCESS) {
        return ayaka_status_make(
            AYAKA_STATUS_UNSUPPORTED,
            "gemm: no cuBLASLt algorithm for this problem "
            "(a larger workspace may help)");
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    AYAKA_LT_RETURN_IF_ERROR(cublasLtMatmul(
        handle, op.desc, &alpha, b.data, layout_a.layout, a.data,
        layout_b.layout, &beta, out.data, layout_c.layout, out.data,
        layout_c.layout, &heuristic.algo, workspace, workspace_bytes,
        static_cast<cudaStream_t>(stream)));
    return ayaka_status_ok();
}

}  // namespace ayaka
