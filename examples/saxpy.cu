#include <cassert>
#include <iostream>
#include <chrono>
#include <agency/execution_policy.hpp>
#include <agency/cuda/execution_policy.hpp>
#include <thrust/device_vector.h>


template<class Result, class Function>
struct device_lambda_wrapper
{
  mutable Function f;

  template<class... Args>
  __device__
  Result operator()(Args&&... args) const
  {
    return f(std::forward<Args>(args)...);
  }
};

template<class Result, class Function>
device_lambda_wrapper<Result,Function> wrap_lambda(Function f)
{
  return device_lambda_wrapper<Result,Function>{f};
}

void saxpy(size_t n, float a, const float* x, const float* y, float* z)
{
  using namespace agency;

  bulk_invoke(cuda::par(n), wrap_lambda<void>(
    [=] __device__ (parallel_agent &self)
    {
      int i = self.index();
      z[i] = a * x[i] + y[i];
    }
  ));
}

int main()
{
  size_t n = 8 << 20;
  thrust::device_vector<float> x(n, 1), y(n, 2), z(n);
  float a = 13.;

  saxpy(n, a, raw_pointer_cast(x.data()), raw_pointer_cast(y.data()), raw_pointer_cast(z.data()));

  thrust::device_vector<float> ref(n, a * 1.f + 2.f);
  assert(ref == z);

  std::cout << "Measuring performance..." << std::endl;

  // time a number of trials
  size_t num_trials = 20;

  auto start = std::chrono::high_resolution_clock::now();
  for(size_t i = 0; i < num_trials; ++i)
  {
    saxpy(n, a, raw_pointer_cast(x.data()), raw_pointer_cast(y.data()), raw_pointer_cast(z.data()));
  }
  std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;

  double seconds = elapsed.count() / num_trials;
  double gigabytes = double(3 * n * sizeof(float)) / (1 << 30);
  double bandwidth = gigabytes / seconds;

  std::cout << "SAXPY Bandwidth: " << bandwidth << " GB/s" << std::endl;

  return 0;
}

