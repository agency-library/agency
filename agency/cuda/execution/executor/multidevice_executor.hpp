#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/execution/executor/executor_array.hpp>
#include <agency/execution/executor/flattened_executor.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/cuda/execution/executor/grid_executor.hpp>
#include <agency/cuda/execution/executor/parallel_executor.hpp>
#include <numeric>
#include <algorithm>
#include <type_traits>
#include <array>

namespace agency
{
namespace cuda
{
namespace detail
{


template<class Range>
std::vector<grid_executor> devices_to_grid_executors(const Range& devices)
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
    supergrid_executor()
      : supergrid_executor(detail::all_devices_as_grid_executors())
    {}

    template<class Range>
    supergrid_executor(const Range& grid_executors)
      : super_t(grid_executors.begin(), grid_executors.end())
    {}
};


class spanning_grid_executor : public flattened_executor<supergrid_executor>
{
  private:
    using super_t = flattened_executor<supergrid_executor>;

  public:
    spanning_grid_executor() = default;

    template<class Range>
    spanning_grid_executor(const Range& grid_executors)
      : super_t(supergrid_executor(grid_executors))
    {}
};

static_assert(is_executor<spanning_grid_executor>::value, "spanning_grid_executor is not an executor!");


class multidevice_executor : public flattened_executor<spanning_grid_executor>
{
  private:
    using super_t = flattened_executor<spanning_grid_executor>;

  public:
    multidevice_executor() = default;

    template<class Range>
    multidevice_executor(const Range& grid_executors)
      : super_t(grid_executors)
    {}

    size_t size() const
    {
      return this->base_executor().base_executor().size();
    }
};

static_assert(is_executor<multidevice_executor>::value, "multidevice_executor is not an executor!");


// XXX devices() and all_devices() should be moved to device.hpp
//
// These functions should return a container of device_id, and we should provide an overload for
// replace_executor(policy, container_of_device_ids) which would introduce the right type of executor for the policy.
//
// for now, just return a multidevice_executor from these functions

template<class... IntegersOrDeviceIds>
multidevice_executor devices(device_id id0, IntegersOrDeviceIds... ids)
{
  std::array<grid_executor, 1 + sizeof...(IntegersOrDeviceIds)> execs = {{grid_executor(id0), grid_executor(ids)...}};
  return multidevice_executor(execs);
}

template<class Range,
         __AGENCY_REQUIRES(
           !std::is_convertible<const Range&, device_id>::value
         )>
multidevice_executor devices(const Range& integers_or_device_ids)
{
  return multidevice_executor(detail::devices_to_grid_executors(integers_or_device_ids));
}

multidevice_executor all_devices()
{
  return cuda::devices(detail::all_devices());
}


} // end cuda
} // end agency

