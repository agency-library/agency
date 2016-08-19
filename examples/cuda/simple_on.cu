#include <agency/agency.hpp>
#include <agency/cuda.hpp>
#include <vector>
#include <cassert>
#include <iostream>
#include <algorithm>

int main()
{
  using namespace agency;

  // allocate data in GPU memory
  using vector = std::vector<float, cuda::managed_allocator<float>>;

  size_t n = 1 << 20;
  float a = 13;
  vector x(n, 1);
  vector y(n, 2);
  vector z(n, 0);

  vector reference(n, 13 * 1 + 2);

  float* x_ptr = x.data();
  float* y_ptr = y.data();
  float* z_ptr = z.data();


  // execute sequentially in the current thread
  bulk_invoke(seq(n), [=](sequenced_agent& self)
  {
    int i = self.index();
    z_ptr[i] = a * x_ptr[i] + y_ptr[i];
  });

  assert(z == reference);
  std::fill(z.begin(), z.end(), 0);


  // execute in parallel on the CPU
  bulk_invoke(par(n), [=](parallel_agent& self)
  {
    int i = self.index();
    z_ptr[i] = a * x_ptr[i] + y_ptr[i];
  });

  assert(z == reference);
  std::fill(z.begin(), z.end(), 0);


  // execute in parallel on a GPU
  cuda::grid_executor gpu;
  bulk_invoke(par(n).on(gpu), [=] __device__ (parallel_agent& self)
  {
    int i = self.index();
    z_ptr[i] = a * x_ptr[i] + y_ptr[i];
  });

  assert(z == reference);
  std::fill(z.begin(), z.end(), 0);


  // execute in parallel on all GPUs in the system
  cuda::multidevice_executor all_gpus;
  bulk_invoke(par(n).on(all_gpus), [=] __device__ (parallel_agent& self)
  {
    int i = self.index();
    z_ptr[i] = a * x_ptr[i] + y_ptr[i];
  });

  assert(z == reference);
  std::fill(z.begin(), z.end(), 0);


  std::cout << "OK" << std::endl;
  return 0;
}

