#pragma once

#include <stdint.h>

#define AYAKA_DESC_HEADER() \
    uint32_t struct_size;   \
    uint32_t abi_version

#define AYAKA_DESC_VERSION_1 1
