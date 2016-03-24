#pragma once

#ifdef __CUDACC__
#  define __AGENCY_ANNOTATION __host__ __device__
#else
#  define __AGENCY_ANNOTATION
#endif // __AGENCY_ANNOTATION

#if defined(__NVCC__) && !(defined(__clang__) && defined(__CUDA__))
#  ifndef __agency_hd_warning_disable__
#    define __agency_hd_warning_disable__ \
#    pragma hd_warning_disable
#  endif // __agency_hd_warning_disable__
#else
#  define __agency_hd_warning_disable__
#endif // __agency_hd_warning_disable__

