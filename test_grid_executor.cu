#include <agency/cuda/grid_executor.hpp>
#include <iostream>
#include <cstdio>

struct hello_world
{
  __device__
  void operator()(agency::uint2 index)
  {
    printf("Hello world from block %d, thread %d\n", index[0], index[1]);
  }
};


struct with_shared_arg
{
  __device__
  void operator()(agency::uint2 index, int& outer_shared, int& inner_shared)
  {
    atomicAdd(&outer_shared, 1);
    atomicAdd(&inner_shared, 1);

    __syncthreads();

    if(index[1] == 0)
    {
      printf("outer_shared: %d\n", outer_shared);
      printf("inner_shared: %d\n", inner_shared);
    }
  }
};


__host__ __device__
void launch_nested_kernel()
{
  agency::cuda::grid_executor ex;
  bulk_invoke(ex, agency::uint2{2,2}, hello_world());
}


__global__ void kernel()
{
  launch_nested_kernel();
}


template<typename T>
__host__ __device__
void maybe_launch_nested_kernel()
{
#if __cuda_lib_has_cudart
  launch_nested_kernel();
#else
  printf("sorry, can't launch a kernel\n");
#endif
}


template<typename T>
__global__ void kernel_template()
{
  printf("kernel_template\n");
  maybe_launch_nested_kernel<T>();
}


template<class T>
struct factory
{
  __host__ __device__
  T operator()() const
  {
    return value_;
  }

  T value_;
};

template<class T>
__host__ __device__
factory<T> make_factory(const T& value)
{
  return factory<T>{value};
}


int main()
{
  agency::cuda::grid_executor ex;

  std::cout << "Testing bulk_invoke on host" << std::endl;
  bulk_invoke(ex, agency::uint2{2,2}, hello_world());
  cudaDeviceSynchronize();

  std::cout << "Testing bulk_invoke with shared arg on host" << std::endl;
  ex.execute(with_shared_arg(), agency::uint2{2,2}, make_factory(7), make_factory(13));
  cudaDeviceSynchronize();

  std::cout << "Testing bulk_invoke() on device" << std::endl;
  kernel<<<1,1>>>();
  cudaDeviceSynchronize();

  std::cout << "Testing bulk_invoke() on device from kernel template" << std::endl;
  kernel_template<int><<<1,1>>>();
  cudaDeviceSynchronize();

  return 0;
}

