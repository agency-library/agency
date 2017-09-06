#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/execution_policy/basic_execution_policy.hpp>
#include <agency/cuda/execution/execution_policy/parallel_execution_policy.hpp>
#include <agency/cuda/execution/execution_policy/concurrent_execution_policy.hpp>
#include <agency/cuda/execution/executor/grid_executor.hpp>
#include <agency/cuda/execution/executor/multidevice_executor.hpp>
#include <agency/cuda/execution/executor/scoped_executor.hpp>
#include <agency/cuda/device.hpp>


namespace agency
{
namespace cuda
{


// XXX consider making this a global object like the other execution policies
inline auto grid(size_t num_blocks, size_t num_threads) ->
  decltype(
    par(num_blocks, con(num_threads))
  )
{
  return par(num_blocks, con(num_threads));
};


// XXX consider making this a unique type instead of an alias
using grid_agent = parallel_group<concurrent_agent>;


// XXX need to figure out how to make this par(con) select grid_executor_2d automatically
// XXX consider making this a global object like the other execution policies
inline auto grid(size2 grid_dim, size2 block_dim) ->
  decltype(
      par2d(grid_dim, con2d(block_dim)).on(grid_executor_2d())
  )
{
  return par2d(grid_dim, con2d(block_dim)).on(grid_executor_2d());
}

// XXX consider making this a unique type instead of an alias
using grid_agent_2d = parallel_group_2d<concurrent_agent_2d>;


// this overload is called on e.g. par(con).on(device(0))
// XXX this function needs to account for the dimensionality of GridPolicy's agents
template<class GridPolicy,
         __AGENCY_REQUIRES(
           agency::detail::policy_is_scoped_parallel_concurrent<GridPolicy>::value
         )>
__AGENCY_ANNOTATION
basic_execution_policy<cuda::grid_agent, cuda::grid_executor>
  replace_executor(const GridPolicy& policy, device_id device)
{
  // create a grid_executor
  cuda::grid_executor exec(device);
  
  return basic_execution_policy<cuda::grid_agent, cuda::grid_executor>(policy.param(), exec);
}


// this overload is called on e.g. par(con).on(all_devices())
// XXX this function needs to account for the dimensionality of GridPolicy's agents
template<class GridPolicy,
         class Range,
         __AGENCY_REQUIRES(
           detail::is_range_of_device_id<Range>::value and
           agency::detail::policy_is_scoped_parallel_concurrent<GridPolicy>::value
         )>
basic_execution_policy<cuda::grid_agent, cuda::spanning_grid_executor>
  replace_executor(const GridPolicy& policy, const Range& devices)
{
  // turn the range of device_id into a vector of grid_executors
  auto grid_executors = agency::cuda::detail::devices_to_grid_executors(devices);
  spanning_grid_executor exec(grid_executors);

  return basic_execution_policy<cuda::grid_agent, cuda::spanning_grid_executor>(policy.param(), exec);
}




} // end cuda
} // end agency

