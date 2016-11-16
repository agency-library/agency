#include <agency/agency.hpp>
#include <agency/cuda.hpp>
#include <iostream>

__managed__ int global_variable;

__global__ void get_global_variable(int* result)
{
  *result = global_variable;
}

int main()
{
  using namespace agency;

  std::vector<int, cuda::managed_allocator<int>> result(1, 13);

  // global_variable starts out at zero
  global_variable = 0;

  cuda::async_future<void> f = bulk_async(cuda::par(1), [] __device__ (parallel_agent& self)
  {
    global_variable = 7;
  });

  // launch a kernel but make its launch dependent on f's completion
  cudaStream_t stream = cuda::experimental::make_dependent_stream(f);

  get_global_variable<<<1,1,0,stream>>>(result.data());

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  assert(result[0] == 7);

  std::cout << "OK" << std::endl;

  return 0;
}

