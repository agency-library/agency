#pragma once

#include <exception>
#include <cstdio>
#include "detail/feature_test.hpp"


__host__ __device__
inline void __terminate()
{
#ifdef __CUDA_ARCH__
  asm("trap;");
#else
  std::terminate();
#endif
}


__host__ __device__
inline void __terminate_with_message(const char* message)
{
  printf("%s\n", message);

  __terminate();
}


__host__ __device__
inline void __terminate_on_error(cudaError_t e, const char* message)
{
  if(e)
  {
#if __cuda_lib_has_cudart
    printf("Error after: %s: %s\n", message, cudaGetErrorString(e));
#endif
    __terminate();
  }
}


inline __host__ __device__
void __throw_on_error(cudaError_t e, const char* message)
{
  if(e)
  {
#ifndef __CUDA_ARCH__
    throw thrust::system_error(e, thrust::cuda_category(), message);
#else
#  if (__cuda_lib_has_printf && __cuda_lib_has_cudart)
    printf("Error after %s: %s\n", message, cudaGetErrorString(e));
#  elif __cuda_lib_has_printf
    printf("Error: %s\n", message);
#  endif
    __terminate();
#endif
  }
}



