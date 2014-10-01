#include <iostream>
#include <cassert>
#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include "cuda/grid_executor.hpp"
#include "cuda/execution_policy.hpp"


struct fill_functor
{
  int* x;

  __host__ __device__
  fill_functor(int* x_)
    : x(x_)
  {}

  __device__
  void operator()(cuda::parallel_agent& self)
  {
    int i = self.index();
    x[i] = 13;
  }
};


int main()
{
  size_t n = 1 << 16;
  thrust::device_vector<int> x(n);

  auto gpu = cuda::grid_executor();

  fill_functor func(raw_pointer_cast(x.data()));

  std::bulk_invoke(std::par(n).on(gpu), func);

  assert(thrust::all_of(x.begin(), x.end(), thrust::placeholders::_1 == 13));

  std::cout << "OK" << std::endl;

  return 0;
}

