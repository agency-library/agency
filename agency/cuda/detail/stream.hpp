#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/detail/feature_test.hpp>
#include <agency/cuda/detail/terminate.hpp>
#include <agency/cuda/gpu.hpp>

namespace agency
{
namespace cuda
{
namespace detail
{


class stream
{
  public:
    // creates a new stream associated with the current gpu
    inline __host__ __device__
    stream()
      : gpu_(current_gpu())
    {
#if __cuda_lib_has_cudart
      detail::throw_on_error(cudaStreamCreate(&s_), "cudaStreamCreate in cuda::detail::stream ctor");
#else
      detail::terminate_with_message("cuda::detail::stream ctor requires CUDART");
#endif
    }

    // creates a new stream associated with the given gpu
    inline __host__ __device__
    stream(gpu_id g)
      : gpu_(g)
    {
      auto old_gpu = current_gpu();

#if __cuda_lib_has_cudart
#  ifdef __CUDA_ARCH__
      assert(g == old_gpu);
#  else
      detail::throw_on_error(cudaSetDevice(gpu().native_handle()), "cudaSetDevice in cuda::detail::stream ctor");
#  endif

      detail::throw_on_error(cudaStreamCreate(&s_), "cudaStreamCreate in cuda::detail::stream ctor");
#  ifdef __CUDA_ARCH__
      detail::throw_on_error(cudaSetDevice(old_gpu.native_handle()), "cudaSetDevice in cuda::detail::stream ctor");
#  endif
#else
      detail::terminate_with_message("cuda::detail::stream ctor requires CUDART");
#endif
    }

    inline __host__ __device__
    stream(stream&& other)
      : gpu_(other.gpu()), s_{}
    {
      s_ = other.s_;
      other.s_ = 0;
    }

    inline __host__ __device__
    ~stream()
    {
#if __cuda_lib_has_cudart
      if(s_ != 0)
      {
        detail::terminate_on_error(cudaStreamDestroy(s_), "cudaStreamDestroy in cuda::detail::stream dtor");
      }
#else
      detail::terminate_with_message("cuda::detail::stream dtor requires CUDART");
#endif
    }

    inline __host__ __device__
    gpu_id gpu() const
    {
      return gpu_;
    }

    inline __host__ __device__
    cudaStream_t native_handle() const
    {
      return s_;
    }

    inline __host__ __device__
    void swap(stream& other)
    {
      gpu_id tmp1 = gpu_;
      gpu_ = other.gpu_;
      other.gpu_ = tmp1;

      cudaStream_t tmp2 = s_;
      s_ = other.s_;
      other.s_ = tmp2;
    }

  private:
    gpu_id gpu_;
    cudaStream_t s_;
};


} // end detail
} // end cuda
} // end agency

