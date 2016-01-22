#pragma once

#include <agency/cuda/grid_executor.hpp>
#include <agency/cuda/parallel_executor.hpp>
#include <agency/executor_array.hpp>
#include <agency/flattened_executor.hpp>

namespace agency
{
namespace cuda
{
namespace detail
{


using supergrid_executor = executor_array<
  grid_executor,
  this_thread::parallel_executor
>;


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


class multidevice_executor : public flattened_executor<detail::supergrid_executor>
{
  private:
    using super_t = flattened_executor<detail::supergrid_executor>;

  public:
    multidevice_executor()
      : multidevice_executor(detail::all_devices_as_grid_executors())
    {}

    template<class Container>
    multidevice_executor(const Container& devices)
      : super_t(detail::supergrid_executor(devices.begin(), devices.end()))
    {}
};


} // end agency
} // end agency

