#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/memory/allocator/managed_allocator.hpp>
#include <agency/experimental/span.hpp>
#include <agency/experimental/ranges/all.hpp>
#include <agency/experimental/tiled_array.hpp>
#include <vector>
#include <array>

namespace agency
{
namespace cuda
{
namespace experimental
{
namespace detail
{


// this function returns a std::vector of managed_allocators, one
// corresponding to each device in the system
template<class T>
std::vector<managed_allocator<T>> all_devices_managed_allocators()
{
  auto devices = cuda::all_devices();

  std::vector<managed_allocator<T>> result;

  for(auto d : devices)
  {
    result.emplace_back(d);
  }

  return result;
}


} // end detail


template<class T>
class multidevice_array : public agency::experimental::tiled_array<T, managed_allocator, managed_allocator>
{
  private:
    using super_t = agency::experimental::tiled_array<T, managed_allocator, managed_allocator>;

  public:
    multidevice_array(size_t n, const T& val = T{})
      : super_t(n, val, detail::all_devices_managed_allocators<T>())
    {}
};


template<class T>
auto all(multidevice_array<T>& a)
  -> decltype(a.all())
{
  return a.all();
}


} // end experimental
} // end cuda
} // end agency

