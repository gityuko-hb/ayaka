#pragma once

#ifdef __cplusplus
#define AYAKA_EXTERN_C extern "C"
#else
#define AYAKA_EXTERN_C
#endif

#if defined(_WIN32)
  #ifdef AYAKA_BUILDING_NATIVE
    #define AYAKA_API __declspec(dllexport)
  #else
    #define AYAKA_API __declspec(dllimport)
  #endif
#elif defined(__GNUC__) && (__GNUC__ >= 4)
  #define AYAKA_API __attribute__((visibility("default")))
#else
  #define AYAKA_API
#endif

/* Statuses must be checked; dropping one silently swallows kernel errors. */
#if defined(__cplusplus) && __cplusplus >= 201703L
  #define AYAKA_NODISCARD [[nodiscard]]
#elif defined(__GNUC__)
  #define AYAKA_NODISCARD __attribute__((warn_unused_result))
#else
  #define AYAKA_NODISCARD
#endif
