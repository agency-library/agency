#include <agency/cuda/execution_policy.hpp>
#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <cassert>
#include <iostream>

struct saxpy_functor
{
  float a;
  float *x, *y, *z;

  __device__
  void operator()(agency::cuda::parallel_agent& self)
  {
    int i = self.index();
    z[i] = a * x[i] + y[i];
  }
};

int main()
{
  size_t n = 1 << 16;
  thrust::device_vector<float> x(n, 1), y(n, 2), z(n);
  float a = 13.;

  auto gpu = agency::cuda::grid_executor();

  auto f = saxpy_functor{
    a,
    raw_pointer_cast(x.data()),
    raw_pointer_cast(y.data()),
    raw_pointer_cast(z.data())
  };

  agency::bulk_invoke(agency::par(n).on(gpu), f);

  float expected  = a * 1. + 2.;
  assert(thrust::all_of(z.begin(), z.end(), thrust::placeholders::_1 == expected));

  std::cout << "OK" << std::endl;

  return 0;
}

