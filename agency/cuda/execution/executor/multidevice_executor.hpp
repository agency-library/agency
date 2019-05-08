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
#include <vector>


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


inline std::vector<grid_executor> all_devices_as_grid_executors()
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


// this function returns a spanning_grid_executor associated with a range of device_id
// XXX we might prefer to return a supergrid_executor and allow replace_executor() to flatten that as necessary
template<class Range,
         __AGENCY_REQUIRES(
           detail::is_range_of_device_id<Range>::value
         )>
spanning_grid_executor associated_executor(const Range& devices)
{
  // turn the range of device_id into a vector of grid_executors
  std::vector<grid_executor> grid_executors = agency::cuda::detail::devices_to_grid_executors(devices);
  return spanning_grid_executor(grid_executors);
}


class multidevice_executor : public flattened_executor<spanning_grid_executor>
{
  private:
    using super_t = flattened_executor<spanning_grid_executor>;

  public:
    multidevice_executor() = default;

    template<class Range, __AGENCY_REQUIRES(!detail::is_range_of_device_id<Range>::value)>
    multidevice_executor(const Range& grid_executors)
      : super_t(grid_executors)
    {}

    template<class Range, __AGENCY_REQUIRES(detail::is_range_of_device_id<Range>::value)>
    multidevice_executor(const Range& devices)
      : multidevice_executor(detail::devices_to_grid_executors(devices))
    {}

    size_t size() const
    {
      return this->base_executor().base_executor().size();
    }
};

static_assert(is_executor<multidevice_executor>::value, "multidevice_executor is not an executor!");


} // end cuda
} // end agency

