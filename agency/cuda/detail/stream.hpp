#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/detail/feature_test.hpp>
#include <agency/cuda/detail/terminate.hpp>

namespace agency
{
namespace cuda
{
namespace detail
{


class stream
{
  public:
    inline __host__ __device__
    stream()
    {
#if __cuda_lib_has_cudart
      detail::throw_on_error(cudaStreamCreate(&s_), "cudaStreamCreate in cuda::detail::stream ctor");
#else
      detail::terminate_with_message("cuda::detail::stream ctor requires CUDART");
#endif
    }

    inline __host__ __device__
    stream(stream&& other)
      : s_{}
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
    cudaStream_t native_handle() const
    {
      return s_;
    }

    inline __host__ __device__
    void swap(stream& other)
    {
      cudaStream_t tmp = s_;
      s_ = other.s_;
      other.s_ = tmp;
    }

  private:
    cudaStream_t s_;
};


} // end detail
} // end cuda
} // end agency

