#pragma once

#include "ayaka/export.h"
#include "ayaka/error_code.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    ayaka_status_code_t code;
    const char* message;
} ayaka_status_t;

static inline ayaka_status_t ayaka_status_ok(void) {
    ayaka_status_t s = {AYAKA_STATUS_OK, NULL};
    return s;
}

static inline ayaka_status_t ayaka_status_make(ayaka_status_code_t code, const char* msg) {
    ayaka_status_t s = {code, msg};
    return s;
}

#ifdef __cplusplus
}
#endif
