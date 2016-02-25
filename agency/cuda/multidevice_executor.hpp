#pragma once

#include <agency/cuda/grid_executor.hpp>
#include <agency/cuda/parallel_executor.hpp>
#include <agency/executor_array.hpp>
#include <agency/flattened_executor.hpp>
#include <numeric>

namespace agency
{
namespace cuda
{
namespace detail
{


template<class Container>
std::vector<grid_executor> devices_to_grid_executors(const Container& devices)
{
  std::vector<grid_executor> result(devices.size());
  std::transform(devices.begin(), devices.end(), result.begin(), [](const device_id& d)
  {
    return grid_executor(d);
  });

  return result;
}


std::vector<grid_executor> all_devices_as_grid_executors()
{
  return detail::devices_to_grid_executors(all_devices());
}


} // end detail


class supergrid_executor : public executor_array<grid_executor, this_thread::parallel_executor>
{
  private:
    using super_t = executor_array<grid_executor, this_thread::parallel_executor>;

  public:
    template<class Container>
    supergrid_executor(const Container& grid_executors)
      : super_t(grid_executors.begin(), grid_executors.end())
    {}

    supergrid_executor()
      : supergrid_executor(detail::all_devices_as_grid_executors())
    {}
};


using spanning_grid_executor = flattened_executor<supergrid_executor>;
static_assert(is_executor<spanning_grid_executor>::value, "spanning_grid_executor is not an executor!");


using multidevice_executor = flattened_executor<spanning_grid_executor>;
static_assert(is_executor<multidevice_executor>::value, "multidevice_executor is not an executor!");


} // end cuda
} // end agency

