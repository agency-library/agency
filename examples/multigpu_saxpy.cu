#include <cassert>
#include <iostream>
#include <chrono>
#include <array>
#include <agency/execution_policy.hpp>
#include <agency/cuda/execution_policy.hpp>
#include <agency/cuda/multidevice_executor.hpp>
#include <agency/experimental/span.hpp>
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


template<class T>
class multidevice_array
{
  public:
    constexpr static size_t num_devices = 2;

    using value_type = T;
    using allocator_type = agency::cuda::managed_allocator<value_type>;

    multidevice_array(size_t n, const value_type& val = value_type{})
      : containers_{container(n/num_devices, val, allocator_type(0)), container(n/num_devices, val, allocator_type(1))}
    {}

    agency::experimental::span<value_type> span(size_t i)
    {
      return agency::experimental::view(containers_[i]);
    }

    agency::experimental::segmented_span<value_type,2> view()
    {
      return agency::experimental::segmented_span<value_type,2>(span(0), span(1));
    }

    value_type& operator[](size_t i)
    {
      return span()[i];
    }

    bool operator==(const multidevice_array& other) const
    {
      return containers_ == other.containers_;
    }

    void clear()
    {
      containers_[0].clear();
      containers_[1].clear();
    }

  private:
    using container = std::vector<value_type, allocator_type>;

    std::array<container,num_devices> containers_;
};


template<class T>
agency::experimental::segmented_span<T,2> view(multidevice_array<T>& a)
{
  return a.view();
}

int main()
{
  using namespace agency;
  using namespace agency::experimental;

  cuda::multidevice_executor exec;

#if GO_FASTER
  using container = multidevice_array<float>;
#else
  using container = std::vector<float, cuda::managed_allocator<float>>;
#endif

  size_t n = 32 << 20;
  container x(n, 1.f), y(n, 2.f), z(n);
  float a = 13.;

  saxpy(exec, n, a, view(x), view(y), view(z));

  {
    // put this in its own scope so that ref's storage gets deallocated
    // before we measure performance
    container ref(n, a * 1.f + 2.f);
    assert(ref == z);
  }

  // ensure everything migrates back to the GPU before timing
  saxpy(exec, n, a, view(x), view(y), view(z));

  std::cout << "Measuring performance..." << std::endl;

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

  return 0;
}

