#pragma once

#include "../feature_test.hpp"
#include "../terminate.hpp"
#include "workaround_unused_variable_warning.hpp"

namespace cuda
{
namespace detail
{


inline void setup_kernel_arguments(size_t){}

template<class Arg1, class... Args>
void setup_kernel_arguments(size_t offset, const Arg1& arg1, const Args&... args)
{
  cudaSetupArgument(arg1, offset);
  setup_kernel_arguments(offset + sizeof(Arg1), args...);
}


template<class T1, class... Ts>
struct first_type
{
  using type = T1;
};


template<class Arg1, class... Args>
__host__ __device__
auto first_parameter(Arg1&& arg1, Args&&... args)
  -> decltype(std::forward<Arg1>(arg1))
{
  return std::forward<Arg1>(arg1);
}


template<typename... Args>
__host__ __device__
cudaError_t triple_chevrons(void* kernel, uint2 shape, int shared_memory_size, cudaStream_t stream, const Args&... args)
{
  // reference the kernel to encourage the compiler not to optimize it away
  workaround_unused_variable_warning(kernel);

#if __cuda_lib_has_cudart
#  ifndef __CUDA_ARCH__
  cudaConfigureCall(dim3(shape.x), dim3(shape.y), shared_memory_size, stream);
  setup_kernel_arguments(0, args...);
  return cudaLaunch(kernel);
#  else
  // XXX generalize to multiple arguments
  if(sizeof...(Args) != 1)
  {
    return cudaErrorNotSupported;
  }

  using Arg = typename first_type<Args...>::type;

  void *param_buffer = cudaGetParameterBuffer(std::alignment_of<Arg>::value, sizeof(Arg));
  std::memcpy(param_buffer, &first_parameter(args...), sizeof(Arg));
  return cudaLaunchDevice(kernel, param_buffer, dim3(shape.x), dim3(shape.y), shared_memory_size, stream);
#  endif // __CUDA_ARCH__
#else // __cuda_lib_has_cudart
  return cudaErrorNotSupported;
#endif
}


template<class... Args>
__host__ __device__
cudaError_t launch_kernel(void* kernel, uint2 shape, int shared_memory_size, cudaStream_t stream, const Args&... args)
{
  struct workaround
  {
    __host__ __device__
    static cudaError_t supported_path(void* kernel, uint2 shape, int shared_memory_size, cudaStream_t stream, const Args&... args)
    {
      // reference the kernel to encourage the compiler not to optimize it away
      workaround_unused_variable_warning(kernel);

      return triple_chevrons(kernel, shape, shared_memory_size, stream, args...);
    }

    __host__ __device__
    static cudaError_t unsupported_path(void* kernel, uint2, int, cudaStream_t, const Args&...)
    {
      // reference the kernel to encourage the compiler not to optimize it away
      workaround_unused_variable_warning(kernel);

      return cudaErrorNotSupported;
    }
  };

#if __cuda_lib_has_cudart
  cudaError_t result = workaround::supported_path(kernel, shape, shared_memory_size, stream, args...);
#else
  cudaError_t result = workaround::unsupported_path(kernel, shape, shared_memory_size, stream, args...);
#endif

  return result;
}


template<class... Args>
__host__ __device__
void checked_launch_kernel(void* kernel, uint2 shape, int shared_memory_size, cudaStream_t stream, const Args&... args)
{
  // the error message we return depends on how the program was compiled
  const char* error_message = 
#if __cuda_lib_has_cudart
   // we have access to CUDART, so something went wrong during the kernel
#  ifndef __CUDA_ARCH__
   "cuda::detail::checked_launch_kernel(): CUDA error after cudaLaunch()"
#  else
   "cuda::detail::checked_launch_kernel(): CUDA error after cudaLaunchDevice()"
#  endif // __CUDA_ARCH__
#else // __cuda_lib_has_cudart
   // we don't have access to CUDART, so output a useful error message explaining why it's unsupported
#  ifndef __CUDA_ARCH__
   "cuda::detail::checked_launch_kernel(): CUDA kernel launch from host requires nvcc"
#  else
   "cuda::detail::checked_launch_kernel(): CUDA kernel launch from device requires arch=sm_35 or better and rdc=true"
#  endif // __CUDA_ARCH__
#endif
  ;

  __throw_on_error(launch_kernel(kernel, shape, shared_memory_size, stream, args...), error_message);
}


template<class... Args>
__host__ __device__
void checked_launch_kernel_on_device(void* kernel, uint2 shape, int shared_memory_size, cudaStream_t stream, int device, const Args&... args)
{
#if __cuda_lib_has_cudart
  // record the current device
  int current_device = 0;
  __throw_on_error(cudaGetDevice(&current_device), "cuda::detail::checked_launch_kernel_on_device(): cudaGetDevice()");
  if(current_device != device)
  {
#  ifndef __CUDA_ARCH__
    __throw_on_error(cudaSetDevice(device), "cuda::detail::checked_launch_kernel_on_device(): cudaSetDevice()");
#  else
    __throw_on_error(cudaErrorNotSupported, "cuda::detail::checked_launch_kernel_on_device(): CUDA kernel launch only allowed on the current device in __device__ code");
#  endif // __CUDA_ARCH__
  }
#else
  // the error message we return depends on how the program was compiled
  const char* error_message = 
#  ifndef __CUDA_ARCH__
     "cuda::detail::checked_launch_kernel_on_device(): CUDA kernel launch from host requires nvcc"
#  else
     "cuda::detail::checked_launch_kernel_on_device(): CUDA kernel launch from device requires arch=sm_35 or better and rdc=true"
#  endif
  ;
  __throw_on_error(cudaErrorNotSupported, error_message);
#endif // __cuda_lib_has_cudart

  checked_launch_kernel(kernel, shape, shared_memory_size, stream, args...);

#if __cuda_lib_has_cudart
  // restore the device
#  ifndef __CUDA_ARCH__
  if(current_device != device)
  {
    __throw_on_error(cudaSetDevice(current_device), "cuda::detail::checked_launch_kernel_on_device: cudaSetDevice()");
  }
#  endif // __CUDA_ARCH__
#else
  __throw_on_error(cudaErrorNotSupported, "cuda::detail::checked_launch_kernel_on_device(): cudaSetDevice requires CUDART");
#endif // __cuda_lib_has_cudart
}


} // end detail
} // end cuda

