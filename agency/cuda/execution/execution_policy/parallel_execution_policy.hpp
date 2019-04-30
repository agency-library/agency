#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/execution_policy/basic_execution_policy.hpp>
#include <agency/cuda/execution/executor/parallel_executor.hpp>
#include <agency/cuda/execution/executor/grid_executor.hpp>
#include <agency/cuda/execution/executor/multidevice_executor.hpp>
#include <agency/cuda/device.hpp>


namespace agency
{
namespace cuda
{


class parallel_execution_policy : public basic_execution_policy<parallel_agent, cuda::parallel_executor, parallel_execution_policy>
{
  private:
    using super_t = basic_execution_policy<parallel_agent, cuda::parallel_executor, parallel_execution_policy>;

  public:
    using super_t::basic_execution_policy;
};


const parallel_execution_policy par{};


class parallel_execution_policy_2d : public basic_execution_policy<parallel_agent_2d, cuda::parallel_executor, parallel_execution_policy_2d>
{
  private:
    using super_t = basic_execution_policy<parallel_agent_2d, cuda::parallel_executor, parallel_execution_policy_2d>;

  public:
    using super_t::basic_execution_policy;
};


const parallel_execution_policy_2d par2d{};


template<class Index>
__AGENCY_ANNOTATION
basic_execution_policy<
  agency::detail::basic_execution_agent<agency::bulk_guarantee_t::parallel_t,Index>,
  agency::cuda::parallel_executor
>
  parnd(const agency::lattice<Index>& domain)
{
  using policy_type = agency::basic_execution_policy<
    agency::detail::basic_execution_agent<agency::bulk_guarantee_t::parallel_t,Index>,
    agency::cuda::parallel_executor
  >;

  typename policy_type::param_type param(domain);
  return policy_type{param};
}


template<class Shape>
__AGENCY_ANNOTATION
basic_execution_policy<
  agency::detail::basic_execution_agent<agency::bulk_guarantee_t::parallel_t,Shape>,
  agency::cuda::parallel_executor
>
  parnd(const Shape& shape)
{
  return cuda::parnd(agency::make_lattice(shape));
}


template<class ParallelPolicy,
         __AGENCY_REQUIRES(
           agency::detail::policy_is_parallel<ParallelPolicy>::value
        )>
__AGENCY_ANNOTATION
auto replace_executor(const ParallelPolicy& policy, const grid_executor& exec)
  -> decltype(agency::replace_executor(policy, cuda::parallel_executor(exec)))
{
  // create a parallel_executor from the grid_executor
  return agency::replace_executor(policy, cuda::parallel_executor(exec));
}


// this overload is called on e.g. par.on(device(0))
template<class ParallelPolicy,
         __AGENCY_REQUIRES(
           agency::detail::policy_is_parallel<ParallelPolicy>::value
         )>
__AGENCY_ANNOTATION
auto replace_executor(const ParallelPolicy& policy, device_id device)
  -> decltype(agency::replace_executor(policy, cuda::grid_executor(device)))
{
  // create a grid_executor from the device_id
  return agency::replace_executor(policy, cuda::grid_executor(device));
}


// this overload is called on e.g. par.on(all_devices())
template<class ParallelPolicy,
         class Range,
         __AGENCY_REQUIRES(
           detail::is_range_of_device_id<Range>::value and
           agency::detail::policy_is_parallel<ParallelPolicy>::value
         )>
auto replace_executor(const ParallelPolicy& policy, const Range& devices)
  -> decltype(agency::replace_executor(policy, multidevice_executor(detail::devices_to_grid_executors(devices))))
{
  // turn the range of device_id into a vector of grid_executors
  return agency::replace_executor(policy, multidevice_executor(detail::devices_to_grid_executors(devices)));
}


} // end cuda
} // end agency

