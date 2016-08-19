What is Agency?
===============

Agency is an experimental C++ template library for parallel programming. Unlike
higher-level parallel algorithms libraries like [Thrust](thrust.github.io),
Agency provides **lower-level** primitives for **creating execution**. Agency
interoperates with standard components like **execution policies** and
**executors** to enable the creation of **portable** parallel algorithms.

# Examples

Agency is best-explained through examples. The following program implements a parallel sum.

~~~~{.cpp}
#include <agency/agency.hpp>
#include <agency/experimental.hpp>
#include <vector>
#include <numeric>
#include <iostream>
#include <cassert>

int parallel_sum(int* data, int n)
{
  // create a view of the input
  agency::experimental::span<int> input(data, n);

  // divide the input into 8 tiles
  int num_agents = 8;
  auto tiles = agency::experimental::tile_evenly(input, num_agents);

  // create 8 agents to sum each tile in parallel
  auto partial_sums = agency::bulk_invoke(agency::par(num_agents), [=](agency::parallel_agent& self)
  {
    // get this parallel agent's tile
    auto this_tile = tiles[self.index()];

    // return the sum of this tile
    return std::accumulate(this_tile.begin(), this_tile.end(), 0);
  });

  // return the sum of partial sums
  return std::accumulate(partial_sums.begin(), partial_sums.end(), 0);
}

int main()
{
  // create a large vector filled with 1s
  std::vector<int> vec(32 << 20, 1);

  int sum = parallel_sum(vec.data(), vec.size());

  std::cout << "sum is " << sum << std::endl;

  assert(sum == vec.size());

  return 0;
}
~~~~

-----------


This code example implements a vector sum operation and executes it sequentially, in parallel, in parallel on a single GPU, and finally multiple GPUs:

~~~~{.cpp}
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
~~~~


# Discover the Library

* Refer to Agency's [Quick Start Guide](http://github.com/agency-library/agency/wiki/Quick-Start-Guide) for further information and examples.
* See Agency in action in the [collection of example programs](http://github.com/agency-library/agency/tree/master/examples).
* Browse Agency's [online API documentation](http://agency-library.github.io/modules.html).

Agency is an [NVIDIA Research](http://research.nvidia.com) project.

