#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/memory/managed_allocator.hpp>
#include <agency/experimental/view.hpp>
#include <vector>
#include <array>

namespace agency
{
namespace cuda
{
namespace experimental
{


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
      return agency::experimental::all(containers_[i]);
    }

    agency::experimental::segmented_span<value_type,2> all()
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
agency::experimental::segmented_span<T,2> all(multidevice_array<T>& a)
{
  return a.all();
}


} // end experimental
} // end cuda
} // end agency

