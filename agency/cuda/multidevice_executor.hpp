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


class spanning_grid_executor : public flattened_executor<detail::supergrid_executor>
{
  private:
    using super_t = flattened_executor<detail::supergrid_executor>;

    static constexpr size_t maximum_blocksize = 256;

    static size_t sum_maximum_grid_sizes()
    {
      auto all_devices = detail::all_devices();

      return std::accumulate(all_devices.begin(), all_devices.end(), 0, [](size_t partial_sum, device_id d)
      {
        return partial_sum + detail::maximum_grid_size_x(d);
      });
    };

  public:
    using super_t::super_t;

    template<class Container>
    spanning_grid_executor(const Container& devices)
      : super_t(detail::supergrid_executor(devices.begin(), devices.end()),
                sum_maximum_grid_sizes(),
                maximum_blocksize)
    {}

    spanning_grid_executor()
      : spanning_grid_executor(detail::all_devices_as_grid_executors())
    {}
};


class multidevice_executor : public flattened_executor<spanning_grid_executor>
{
  private:
    using super_t = flattened_executor<spanning_grid_executor>;

  public:
    using super_t::super_t;

    multidevice_executor()
      : super_t(2)
    {}
};


} // end agency
} // end agency

