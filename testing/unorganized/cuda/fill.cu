#include <iostream>
#include <cassert>
#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <agency/agency.hpp>
#include <agency/cuda.hpp>


struct fill_functor
{
  int* x;

  __host__ __device__
  fill_functor(int* x_)
    : x(x_)
  {}

  __device__
  void operator()(agency::parallel_agent& self)
  {
    int i = self.index();
    x[i] = 13;
  }
};


int main()
{
  size_t n = 1 << 16;
  thrust::device_vector<int> x(n);

  fill_functor func(raw_pointer_cast(x.data()));

  agency::bulk_invoke(agency::cuda::par(n), func);

  assert(thrust::all_of(x.begin(), x.end(), thrust::placeholders::_1 == 13));

  std::cout << "OK" << std::endl;

  return 0;
}

