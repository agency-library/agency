#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/terminate.hpp>
#include <agency/cuda/detail/feature_test.hpp>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>


namespace agency
{
namespace cuda
{
namespace detail
{


__host__ __device__
inline void print_error_message(cudaError_t e, const char* message)
{
#if (__cuda_lib_has_printf && __cuda_lib_has_cudart)
  printf("Error after %s: %s\n", message, cudaGetErrorString(e));
#elif __cuda_lib_has_printf
  printf("Error: %s\n", message);
#endif
}


__host__ __device__
inline void print_error_message_if(cudaError_t e, const char* message)
{
  if(e)
  {
    agency::cuda::detail::print_error_message(e, message);
  }
}


__host__ __device__
inline void terminate_on_error(cudaError_t e, const char* message)
{
  if(e)
  {
    agency::cuda::detail::print_error_message(e, message);

    agency::detail::terminate();
  }
}


inline __host__ __device__
void throw_on_error(cudaError_t e, const char* message)
{
  if(e)
  {
#ifndef __CUDA_ARCH__
    throw thrust::system_error(e, thrust::cuda_category(), message);
#else
    agency::cuda::detail::print_error_message(e, message);

    agency::detail::terminate();
#endif
  }
}


} // end detail
} // cuda
} // end agency

