#pragma once

// Tensor-parallel collectives ABI (scaffold). The implementation links NCCL and
// is built only with -DAYAKA_ENABLE_NCCL=ON (and the Rust `nccl` feature); it
// is intentionally NOT in the default sources so the base build needs no NCCL.
// The Rust sharding/reduce logic that drives these lives in ayaka_core::tp.

#include <stddef.h>
#include <stdint.h>

#include "ayaka/export.h"
#include "ayaka/status.h"
#include "ayaka/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque NCCL communicator handle (ncclComm_t). */
typedef void* ayaka_comm_t;

/* Reduction op; matches ayaka_core::tp::ReduceOp / ncclRedOp_t. */
typedef enum {
  AYAKA_REDUCE_SUM = 0,
  AYAKA_REDUCE_PROD = 1,
  AYAKA_REDUCE_MAX = 2,
  AYAKA_REDUCE_MIN = 3,
  AYAKA_REDUCE_AVG = 4,
} ayaka_reduce_op_t;

/* A NCCL unique id (128 bytes) broadcast out-of-band to bootstrap a group. */
typedef struct {
  char bytes[128];
} ayaka_comm_unique_id_t;

/* Generate the bootstrap id on rank 0. */
AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_comm_get_unique_id(ayaka_comm_unique_id_t* out_id);

/* Initialize this rank's communicator from the shared id. */
AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_comm_init_rank(ayaka_comm_t* out_comm, const ayaka_comm_unique_id_t* id,
                     int32_t world_size, int32_t rank);

AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_comm_destroy(ayaka_comm_t comm);

/* In-place all-reduce of `count` `dtype` elements at `buffer`, async on stream.
 */
AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t ayaka_comm_all_reduce(
    ayaka_comm_t comm, void* buffer, size_t count, int32_t dtype,
    ayaka_reduce_op_t op, ayaka_stream_t stream);

#ifdef __cplusplus
}
#endif