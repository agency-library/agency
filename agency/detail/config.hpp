#pragma once

#ifdef __CUDACC__
#  define __AGENCY_ANNOTATION __host__ __device__
#else
#  define __AGENCY_ANNOTATION
#endif // __AGENCY_ANNOTATION

#if defined(__NVCC__) && !(defined(__clang__) && defined(__CUDA__))
#  ifndef __agency_exec_check_disable__
#    define __agency_exec_check_disable__ \
#    pragma nv_exec_check_disable
#  endif // __agency_exec_check_disable__
#else
#  define __agency_exec_check_disable__
#endif // __agency_exec_check_disable__

