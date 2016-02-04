#include <cassert>
#include <iostream>
#include <chrono>
#include <agency/execution_policy.hpp>
#include <agency/cuda/execution_policy.hpp>
#include <agency/cuda/multidevice_executor.hpp>


struct functor
{
  float a;
  const float* x;
  const float* y;
  float* z;

  __device__
  void operator()(agency::parallel_agent& self)
  {
    int i = self.index();
    z[i] = a * x[i] + y[i];
  }
};

void saxpy(agency::cuda::multidevice_executor& exec, size_t n, float a, const float* x, const float* y, float* z)
{
  using namespace agency;

  bulk_invoke(cuda::par(n).on(exec), functor{a, x, y, z});
}

int main()
{
  using namespace agency;

  cuda::multidevice_executor exec;

  using container = cuda::multidevice_executor::container<float>;

  size_t n = 16 << 20;
  container x(n, 1.f), y(n, 2.f), z(n);
  float a = 13.;

  saxpy(exec, n, a, x.data(), y.data(), z.data());

  container ref(n, a * 1.f + 2.f);
  assert(ref == z);

  std::cout << "Measuring performance..." << std::endl;

  // time a number of trials
  size_t num_trials = 20;

  auto start = std::chrono::high_resolution_clock::now();
  for(size_t i = 0; i < num_trials; ++i)
  {
    saxpy(exec, n, a, x.data(), y.data(), z.data());
  }
  std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;

  double seconds = elapsed.count() / num_trials;
  double gigabytes = double(3 * n * sizeof(float)) / (1 << 30);
  double bandwidth = gigabytes / seconds;

  std::cout << "SAXPY Bandwidth: " << bandwidth << " GB/s" << std::endl;

  return 0;
}

