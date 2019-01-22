#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/cuda/detail/feature_test.hpp>
#include <agency/detail/is_call_possible.hpp>
#include <agency/cuda/device.hpp>
#include <type_traits>


namespace agency
{
namespace cuda
{
namespace detail
{

template<class GlobalFunctionPointer, class... Args,
         __AGENCY_REQUIRES(
           std::is_pointer<GlobalFunctionPointer>::value
         ),
         __AGENCY_REQUIRES(
           agency::detail::is_call_possible<GlobalFunctionPointer, Args...>::value
         )>
cudaError_t launch_cooperative_kernel(GlobalFunctionPointer kernel, ::dim3 grid_dim, ::dim3 block_dim, size_t shared_memory_size, cudaStream_t stream, const Args&... args)
{
#if __cuda_lib_has_cudaLaunchCooperativeKernel
  const void* const_void_function_pointer = reinterpret_cast<const void*>(kernel);
  void* ptrs_to_args[] = {&const_cast<Args&>(args)...};

  return cudaLaunchCooperativeKernel(const_void_function_pointer, grid_dim, block_dim, ptrs_to_args, shared_memory_size, stream);
#else
  return cudaErrorNotSupported;
#endif
}

template<class GlobalFunctionPointer, class... Args,
         __AGENCY_REQUIRES(
           std::is_pointer<GlobalFunctionPointer>::value
         ),
         __AGENCY_REQUIRES(
           agency::detail::is_call_possible<GlobalFunctionPointer, Args...>::value
         )>
void try_launch_cooperative_kernel(GlobalFunctionPointer kernel, ::dim3 grid_dim, ::dim3 block_dim, size_t shared_memory_size, cudaStream_t stream, const Args&... args)
{
  agency::cuda::detail::throw_on_error(launch_cooperative_kernel(kernel, grid_dim, block_dim, shared_memory_size, stream, args...), "cuda::detail::try_launch_cooperative_kernel(): CUDA error after cudaLaunchCooperativeKernel()");
}


template<class GlobalFunctionPointer, class... Args,
         __AGENCY_REQUIRES(
           std::is_pointer<GlobalFunctionPointer>::value
         ),
         __AGENCY_REQUIRES(
           agency::detail::is_call_possible<GlobalFunctionPointer, Args...>::value
         )>
void try_launch_cooperative_kernel_on_device(GlobalFunctionPointer kernel, ::dim3 grid_dim, ::dim3 block_dim, size_t shared_memory_size, cudaStream_t stream, int device, const Args&... args)
{
  scoped_device scope(device);

  try_launch_cooperative_kernel(kernel, grid_dim, block_dim, shared_memory_size, stream, args...);
}

} // end detail
} // end cuda
} // end agency

