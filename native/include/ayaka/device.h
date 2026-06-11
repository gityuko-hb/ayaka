#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    AYAKA_DEVICE_CPU  = 0,
    AYAKA_DEVICE_CUDA = 1,
} ayaka_device_kind_t;

typedef struct {
    ayaka_device_kind_t kind;
    uint16_t ordinal;
} ayaka_device_t;

typedef struct {
    uint8_t major;
    uint8_t minor;
} ayaka_sm_arch_t;

#ifdef __cplusplus
}
#endif
