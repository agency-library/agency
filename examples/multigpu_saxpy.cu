#include <cassert>
#include <iostream>
#include <chrono>
#include <agency/execution_policy.hpp>
#include <agency/cuda/execution_policy.hpp>
#include <agency/cuda/multidevice_executor.hpp>
#include <agency/cuda/experimental/multidevice_array.hpp>
#include <agency/experimental/view.hpp>


template<class View>
void saxpy(agency::cuda::multidevice_executor& exec, size_t n, float a, View x, View y, View z)
{
  using namespace agency;

  bulk_invoke(cuda::par(n).on(exec), [=] __host__ __device__ (agency::parallel_agent& self)
  {
    int i = self.index();
    z[i] = a * x[i] + y[i];
  });
}


template<class Container>
void time_saxpy()
{
  using namespace agency;
  using namespace agency::experimental;

  cuda::multidevice_executor exec;

  size_t n = 32 << 20;
  Container x(n, 1.f), y(n, 2.f), z(n);
  float a = 13.;

  saxpy(exec, n, a, view(x), view(y), view(z));

  {
    // put this in its own scope so that ref's storage gets deallocated
    // before we measure performance
    Container ref(n, a * 1.f + 2.f);
    assert(ref == z);
  }

  // ensure everything migrates back to the GPU before timing
  saxpy(exec, n, a, view(x), view(y), view(z));

  // time a number of trials
  size_t num_trials = 20;

  auto start = std::chrono::high_resolution_clock::now();
  for(size_t i = 0; i < num_trials; ++i)
  {
    saxpy(exec, n, a, view(x), view(y), view(z));
  }
  std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;

  double seconds = elapsed.count() / num_trials;
  double gigabytes = double(3 * n * sizeof(float)) / (1 << 30);
  double bandwidth = gigabytes / seconds;

  std::cout << "SAXPY Bandwidth: " << bandwidth << " GB/s" << std::endl;
}

int main()
{
  std::cout << "Measuring the performance of SAXPY using contiguous containers..." << std::endl;
  using contiguous_container = std::vector<float, agency::cuda::managed_allocator<float>>;
  time_saxpy<contiguous_container>();

  std::cout << std::endl;

  std::cout << "Measuring the performance of SAXPY using segmented containers..." << std::endl;
  using segmented_container = agency::cuda::experimental::multidevice_array<float>;
  time_saxpy<segmented_container>();

  return 0;
}

