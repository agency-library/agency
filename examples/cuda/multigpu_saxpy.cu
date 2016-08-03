#include <cassert>
#include <iostream>
#include <chrono>
#include <agency/agency.hpp>
#include <agency/cuda.hpp>
#include <agency/cuda/experimental.hpp>
#include <agency/experimental.hpp>


template<class Executor, class View>
void saxpy(Executor& exec, size_t n, float a, View x, View y, View z)
{
  using namespace agency;

  bulk_invoke(par(n).on(exec), [=] __host__ __device__ (agency::parallel_agent& self)
  {
    int i = self.index();
    z[i] = a * x[i] + y[i];
  });
}


template<class Container, class Executor>
double time_saxpy()
{
  using agency::experimental::all;

  Executor exec;

  size_t n = 32 << 20;
  Container x(n, 1.f), y(n, 2.f), z(n);
  float a = 13.;

  saxpy(exec, n, a, all(x), all(y), all(z));

  {
    // put this in its own scope so that ref's storage gets deallocated
    // before we measure performance
    Container ref(n, a * 1.f + 2.f);
    assert(ref == z);
  }

  // ensure everything migrates back to the GPU before timing
  saxpy(exec, n, a, all(x), all(y), all(z));

  // time a number of trials
  size_t num_trials = 20;

  auto start = std::chrono::high_resolution_clock::now();
  for(size_t i = 0; i < num_trials; ++i)
  {
    saxpy(exec, n, a, all(x), all(y), all(z));
  }
  std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;

  double seconds = elapsed.count() / num_trials;
  double gigabytes = double(3 * n * sizeof(float)) / (1 << 30);
  return gigabytes / seconds;
}


int main()
{
  // the right choice of container can make a huge difference to performance

  // a contiguous container such as std::vector is laid out in a single contiguous allocation
  using contiguous_container = std::vector<float, agency::cuda::managed_allocator<float>>;

  // a segmented container such as multidevice_array is pieced together from separate, contiguous allocations
  using segmented_container  = agency::cuda::experimental::multidevice_array<float>;

  // let's first measure the performance of SAXPY on a single GPU using a contiguous container
  using gpu_executor = agency::cuda::grid_executor;

  std::cout << "Measuring the performance of gpu SAXPY using contiguous containers..." << std::endl;
  auto singlegpu_bandwidth = time_saxpy<contiguous_container, gpu_executor>();
  std::cout << "SAXPY Bandwidth: " << singlegpu_bandwidth << " GB/s" << std::endl;

  std::cout << std::endl;

  // now let's use multiple gpus at once to execute SAXPY and see how the choice of container impacts performance
  using multigpu_executor = agency::cuda::multidevice_executor;

  std::cout << "Measuring the performance of multigpu SAXPY using contiguous containers..." << std::endl;
  auto multigpu_contiguous_bandwidth = time_saxpy<contiguous_container, multigpu_executor>();
  std::cout << "SAXPY Bandwidth: " << multigpu_contiguous_bandwidth << " GB/s" << std::endl;

  std::cout << std::endl;

  std::cout << "Measuring the performance of multigpu SAXPY using segmented containers..." << std::endl;
  auto multigpu_segmented_bandwidth = time_saxpy<segmented_container, multigpu_executor>();
  std::cout << "SAXPY Bandwidth: " << multigpu_segmented_bandwidth << " GB/s" << std::endl;

  std::cout << std::endl;

  std::cout << "Multigpu speedup: " <<  multigpu_segmented_bandwidth / singlegpu_bandwidth << "x" << std::endl;

  return 0;
}

