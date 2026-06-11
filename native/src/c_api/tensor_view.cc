#include "ayaka/tensor_view.h"
#include "ayaka/check.h"

extern "C" AYAKA_API ayaka_status_t
ayaka_tensor_view_validate(const ayaka_tensor_view_t* view) {
    AYAKA_CHECK_NONNULL(view, "tensor_view is null");
    AYAKA_CHECK(view->struct_size == sizeof(ayaka_tensor_view_t),
                "tensor_view struct_size mismatch");
    AYAKA_CHECK(view->abi_version == AYAKA_DESC_VERSION_1,
                "tensor_view unsupported abi_version");
    AYAKA_CHECK_NONNULL(view->data, "tensor_view data is null");
    AYAKA_CHECK(view->rank >= 0 && view->rank <= (int32_t)AYAKA_MAX_RANK,
                "tensor_view rank out of range [0, 8]");
    AYAKA_CHECK(view->dtype >= AYAKA_DTYPE_F32 && view->dtype <= AYAKA_DTYPE_BOOL,
                "tensor_view dtype out of range");
    AYAKA_CHECK(view->device_kind == AYAKA_DEVICE_CPU || view->device_kind == AYAKA_DEVICE_CUDA,
                "tensor_view unknown device kind");
    return ayaka_status_ok();
}
