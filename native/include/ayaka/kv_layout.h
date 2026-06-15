#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Physical layout of the paged KV cache. Mirrors `KvLayout` in
 * crates/ayaka-core/src/layout.rs. NHD is the FlashInfer default; HND
 * coalesces better for FP8 KV (not yet implemented). */
typedef enum {
  /* (.., seq, head, dim) */
  AYAKA_KV_LAYOUT_NHD = 0,
  /* (.., head, seq, dim) */
  AYAKA_KV_LAYOUT_HND = 1,
} ayaka_kv_layout_t;

#ifdef __cplusplus
}
#endif
