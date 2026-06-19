#pragma once

#include "ayaka/error.h"
#include <stddef.h>

#define AYAKA_RETURN_IF_ERROR(expr)            \
  do {                                         \
    ayaka_status_t _s = (expr);                \
    if (_s.code != AYAKA_STATUS_OK) return _s; \
  } while (0)

#define AYAKA_CHECK(cond, msg)                                                 \
  do {                                                                         \
    if (!(cond)) return ayaka_status_make(AYAKA_STATUS_INVALID_ARGUMENT, msg); \
  } while (0)

#define AYAKA_CHECK_NONNULL(ptr, msg) AYAKA_CHECK((ptr) != NULL, msg)

#define AYAKA_CHECK_EQ(a, b, msg)                                            \
  do {                                                                       \
    if ((a) != (b)) return ayaka_status_make(AYAKA_STATUS_SHAPE_ERROR, msg); \
  } while (0)

#define AYAKA_CHECK_GE(a, b, msg)                                           \
  do {                                                                      \
    if ((a) < (b)) return ayaka_status_make(AYAKA_STATUS_SHAPE_ERROR, msg); \
  } while (0)

#define AYAKA_CHECK_GT(a, b, msg)                                            \
  do {                                                                       \
    if ((a) <= (b)) return ayaka_status_make(AYAKA_STATUS_SHAPE_ERROR, msg); \
  } while (0)

#define AYAKA_CHECK_LE(a, b, msg)                                           \
  do {                                                                      \
    if ((a) > (b)) return ayaka_status_make(AYAKA_STATUS_SHAPE_ERROR, msg); \
  } while (0)
