#include "cuda_executor.hpp"

struct hello_world
{
  __device__
  void operator()(uint2 index)
  {
    printf("Hello world from block %d, thread %d\n", index.x, index.y);
  }
};


__host__ __device__
void launch_nested_kernel()
{
  cuda_executor ex;
  bulk_invoke(ex, make_uint2(2,2), hello_world());
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

int main()
{
  cuda_executor ex;

  std::cout << "Testing bulk_invoke on host" << std::endl;

  bulk_invoke(ex, make_uint2(2,2), hello_world());

  cudaDeviceSynchronize();

  std::cout << "Testing bulk_invoke() on device" << std::endl;

  kernel<<<1,1>>>();

  cudaDeviceSynchronize();

  std::cout << "Testing bulk_invoke() on device from kernel template" << std::endl;

  kernel_template<int><<<1,1>>>();

  cudaDeviceSynchronize();

  return 0;
}

