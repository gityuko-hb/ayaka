#pragma once

#include <stdint.h>

#include "ayaka/export.h"
#include "ayaka/status.h"
#include "ayaka/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque CUDA graph handles (cudaGraph_t / cudaGraphExec_t). */
typedef void* ayaka_graph_t;
typedef void* ayaka_graph_exec_t;

/*
 * Owned CUDA streams for the overlap scheduler (a high-priority compute stream
 * and a separate copy stream). The returned handle is a raw cudaStream_t and
 * must be released with ayaka_stream_destroy.
 */
AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_stream_create(ayaka_stream_t* out_stream, int32_t high_priority);

AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_stream_destroy(ayaka_stream_t stream);

/*
 * CUDA graph capture/replay for fixed-shape decode steps. Usage:
 *   begin_capture(stream) -> enqueue kernels on stream ->
 * end_capture(stream,&g)
 *   -> instantiate(g,&exec) -> launch(exec, stream) [replay N times]
 *   -> exec_destroy(exec) -> destroy(g).
 * All launches are asynchronous on their stream.
 */
AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_stream_begin_capture(ayaka_stream_t stream);

AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_stream_end_capture(ayaka_stream_t stream, ayaka_graph_t* out_graph);

AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_graph_instantiate(ayaka_graph_t graph, ayaka_graph_exec_t* out_exec);

AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_graph_launch(ayaka_graph_exec_t exec, ayaka_stream_t stream);

AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_graph_exec_destroy(ayaka_graph_exec_t exec);

AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_graph_destroy(ayaka_graph_t graph);

#ifdef __cplusplus
}
#endif