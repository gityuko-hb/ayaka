#pragma once

#include <exception>
#include <new>

#include "ayaka/status.h"

#define AYAKA_C_API_GUARD(expr)                                      \
    do {                                                             \
        try {                                                        \
            return (expr);                                           \
        } catch (const std::bad_alloc&) {                            \
            return ayaka_status_make(AYAKA_STATUS_OUT_OF_MEMORY,     \
                                     "native C API allocation failed"); \
        } catch (const std::exception&) {                            \
            return ayaka_status_make(AYAKA_STATUS_INTERNAL_ERROR,    \
                                      "native C API exception");      \
        } catch (...) {                                              \
            return ayaka_status_make(AYAKA_STATUS_INTERNAL_ERROR,    \
                                      "native C API unknown exception"); \
        }                                                            \
    } while (0)
