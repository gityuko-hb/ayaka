cmake_minimum_required(VERSION 3.20)
project(ayaka-native LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    # candle-kernels builds its C++/CUDA objects with the static CRT (/MT);
    # match it so Rust test binaries can link both libraries side by side.
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

option(AYAKA_ENABLE_CUDA "Enable CUDA backend" OFF)

set(AYAKA_NATIVE_SOURCES
    src/c_api/tensor_view.cc
)

set(AYAKA_CUDA_HOST_SOURCES
    src/ops/fused_add_rmsnorm.cc
    src/ops/rmsnorm.cc
    src/ops/rope.cc
    src/ops/silu_and_mul.cc
)

set(AYAKA_CUDA_SOURCES
    csrc/activation/swiglu.cu
    csrc/norm/fused_add_rmsnorm.cu
    csrc/norm/rmsnorm.cu
    csrc/pos/rope.cu
)

if(AYAKA_ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES 86)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    list(APPEND AYAKA_NATIVE_SOURCES ${AYAKA_CUDA_HOST_SOURCES} ${AYAKA_CUDA_SOURCES})
endif()

set(AYAKA_NATIVE_HEADERS
    include/ayaka/export.h
    include/ayaka/desc_header.h
    include/ayaka/dtype.h
    include/ayaka/device.h
    include/ayaka/stream.h
    include/ayaka/error_code.h
    include/ayaka/error.h
    include/ayaka/check.h
    include/ayaka/status.h
    include/ayaka/tensor_view.h
)

set(AYAKA_OPS_HEADERS
    include/ayaka/ops/fused_add_rmsnorm.h
    include/ayaka/ops/rmsnorm.h
    include/ayaka/ops/rope.h
    include/ayaka/ops/silu_and_mul.h
)

add_library(ayaka-native STATIC ${AYAKA_NATIVE_SOURCES})

target_include_directories(ayaka-native
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_include_directories(ayaka-native
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(ayaka-native PRIVATE AYAKA_BUILDING_NATIVE)

set_target_properties(ayaka-native PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS ayaka-native
    EXPORT ayaka-native-targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${AYAKA_NATIVE_HEADERS} DESTINATION include/ayaka)
install(FILES ${AYAKA_OPS_HEADERS} DESTINATION include/ayaka/ops)
